"""
MMRDR-OCT Explainability — t-SNE + GradCAM
============================================
Single-task version (3-class DME grading).

Generates:
  1. t-SNE of backbone features colored by DME grade
  2. GradCAM on top-K correct predictions per class (implicit localization)
  3. GradCAM on top-K errors (error analysis)
  4. GradCAM NCI vs CI comparison (clinical question: does model distinguish location?)

Usage:
    python finetune_mmrdr/analyse_mmrdr.py \
        --data_path /home/student/Ungureanu_Daria/MMRDR-OCT \
        --csv /home/student/Ungureanu_Daria/MMRDR-OCTOCT.csv \
        --model_path saved_models/mmrdr_domain_adapted_with_aug/best_model.pth \
        --out_dir results/explainability/mmrdr
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    MMRDRDataset, load_mmrdr_splits, get_eval_transform, CLASS_NAMES,
)
from model import MMRDRModel, load_backbone


def load_model(model_path, device):
    print(f"[INFO] Loading: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    config = ckpt["config"]
    num_classes = ckpt["num_classes"]

    print(f"[INFO] arch={config['arch']}  unfreeze={config['unfreeze_last_n']}  "
          f"epoch={ckpt['epoch']}  val_f1={ckpt['val_f1']:.4f}")

    backbone = load_backbone(config["arch"], config["checkpoint"], device)
    model = MMRDRModel(
        backbone=backbone,
        num_classes=num_classes,
        freeze_backbone=(config["unfreeze_last_n"] < 12),
        unfreeze_last_n=config["unfreeze_last_n"],
        head_hidden=config["head_hidden"],
        head_dropout=config["head_dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config, num_classes


@torch.no_grad()
def extract_features(model, loader, device):
    all_feats, all_labels = [], []
    for images, labels in tqdm(loader, desc="Extracting features"):
        images = images.to(device)
        feats = model.backbone(images)
        if isinstance(feats, dict):
            cls_token = feats["x_norm_clstoken"]
        elif isinstance(feats, tuple):
            cls_token = feats[0]
        else:
            cls_token = feats
        if cls_token.dim() == 3:
            cls_token = cls_token[:, 0, :]
        all_feats.append(cls_token.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


def plot_tsne(features, labels, out_path, perplexity=30):
    print(f"[t-SNE] {len(features)} samples...")
    coords = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(features)

    plt.figure(figsize=(10, 8))
    colors = ["#2196F3", "#FF9800", "#F44336"]
    for grade in range(3):
        mask = labels == grade
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    label=CLASS_NAMES[grade], color=colors[grade],
                    alpha=0.7, s=40, edgecolors="white", linewidths=0.3)

    plt.title("t-SNE — DME Severity Feature Space", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11, title="DME Grade")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path}")


def reshape_transform_vit(tensor):
    """ViT output [B, tokens, C] → [B, C, H, W] for GradCAM."""
    result = tensor[:, 1:, :]  # drop CLS
    grid = int(np.sqrt(result.size(1)))
    result = result.reshape(tensor.size(0), grid, grid, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def denormalize(img_tensor):
    img = img_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1)


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Returns y_true, y_pred, y_conf, sample_indices."""
    softmax = nn.Softmax(dim=1)
    y_true, y_pred, y_conf, indices = [], [], [], []
    idx = 0
    for images, labels in tqdm(loader, desc="Collecting predictions"):
        images = images.to(device)
        logits = model(images)
        probs = softmax(logits)
        conf, pred = torch.max(probs, dim=1)
        bsz = images.size(0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        y_conf.extend(conf.cpu().numpy().tolist())
        indices.extend(range(idx, idx + bsz))
        idx += bsz
    return np.array(y_true), np.array(y_pred), np.array(y_conf), np.array(indices)


def select_samples(y_true, y_pred, y_conf, indices, class_idx=None,
                   correct=False, topk=6):
    """
    Select top-K samples by confidence.
    correct=True: most confident CORRECT predictions
    correct=False: most confident WRONG predictions
    class_idx: filter by true label (None = all classes)
    """
    if correct:
        mask = y_true == y_pred
    else:
        mask = y_true != y_pred

    if class_idx is not None:
        mask = mask & (y_true == class_idx)

    sel_indices = indices[mask]
    sel_conf = y_conf[mask]

    if len(sel_indices) == 0:
        return np.array([], dtype=int)

    order = np.argsort(-sel_conf)
    return sel_indices[order[:topk]]


def generate_gradcam_grid(model, dataset, sample_indices, save_path,
                          device, title="GradCAM"):
    """
    Generate GradCAM for selected samples.
    Each row: Original | CAM for Predicted | CAM for True
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("[ERROR] Install: pip install grad-cam")
        return

    if len(sample_indices) == 0:
        print(f"[SKIP] No samples for: {title}")
        return

    # Enable gradients
    for p in model.backbone.parameters():
        p.requires_grad_(True)

    target_layers = [model.backbone.blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layers,
                  reshape_transform=reshape_transform_vit)

    softmax = nn.Softmax(dim=1)
    n = len(sample_indices)

    fig, axes = plt.subplots(n, 3, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(sample_indices):
        img_tensor, true_label = dataset[idx]
        img_input = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_input)
            probs = softmax(logits)
            pred_class = int(torch.argmax(probs).item())
            pred_conf = float(torch.max(probs).item())

        true_name = CLASS_NAMES[true_label]
        pred_name = CLASS_NAMES[pred_class]
        rgb_img = denormalize(img_input)

        cam_pred = cam(input_tensor=img_input,
                       targets=[ClassifierOutputTarget(pred_class)])[0]
        cam_true = cam(input_tensor=img_input,
                       targets=[ClassifierOutputTarget(true_label)])[0]

        vis_pred = show_cam_on_image(rgb_img, cam_pred, use_rgb=True)
        vis_true = show_cam_on_image(rgb_img, cam_true, use_rgb=True)

        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title(f"TRUE: {true_name}\nPRED: {pred_name} ({pred_conf:.2f})", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(vis_pred)
        axes[i, 1].set_title(f"CAM → Predicted: {pred_name}", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(vis_true)
        axes[i, 2].set_title(f"CAM → True: {true_name}", fontsize=10)
        axes[i, 2].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {save_path}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MMRDR Explainability")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/explainability/mmrdr")
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_tsne", action="store_true")
    parser.add_argument("--skip_gradcam", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model and data
    model, config, num_classes = load_model(args.model_path, device)

    _, _, test_df, _ = load_mmrdr_splits(args.csv, args.data_path)
    eval_transform = get_eval_transform(config["img_size"])
    test_ds = MMRDRDataset(test_df, args.data_path, eval_transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # ── t-SNE ──────────────────────────────────────────────────
    if not args.skip_tsne:
        features, labels = extract_features(model, test_loader, device)
        plot_tsne(features, labels,
                  os.path.join(args.out_dir, "tsne_dme.png"))

    # ── GradCAM ────────────────────────────────────────────────
    if not args.skip_gradcam:
        y_true, y_pred, y_conf, idx = collect_predictions(
            model, test_loader, device,
        )

        # 1. CORRECT predictions — per class (implicit localization)
        #    Shows: "model looks at the right structures when correct"
        for grade in range(3):
            correct_samples = select_samples(
                y_true, y_pred, y_conf, idx,
                class_idx=grade, correct=True, topk=args.topk,
            )
            generate_gradcam_grid(
                model, test_ds, correct_samples, device=device,
                save_path=os.path.join(args.out_dir,
                                       f"gradcam_correct_{CLASS_NAMES[grade]}.png"),
                title=f"GradCAM — Correct {CLASS_NAMES[grade]} Predictions",
            )

        # 2. ERRORS — top confident mistakes (error analysis)
        #    Shows: "where and why the model fails"
        error_samples = select_samples(
            y_true, y_pred, y_conf, idx,
            correct=False, topk=args.topk,
        )
        generate_gradcam_grid(
            model, test_ds, error_samples, device=device,
            save_path=os.path.join(args.out_dir, "gradcam_top_errors.png"),
            title="GradCAM — Top Confident Errors",
        )

        # 3. NCI vs CI confusion specifically
        #    Shows: "when model confuses NCI↔CI, where does it look?"
        nci_as_ci = select_samples(
            y_true, y_pred, y_conf, idx,
            class_idx=1, correct=False, topk=args.topk,
        )
        generate_gradcam_grid(
            model, test_ds, nci_as_ci, device=device,
            save_path=os.path.join(args.out_dir,
                                   "gradcam_errors_NCI_misclassified.png"),
            title="GradCAM — NCI-DME Misclassified (hardest class)",
        )

        ci_as_nci = select_samples(
            y_true, y_pred, y_conf, idx,
            class_idx=2, correct=False, topk=args.topk,
        )
        generate_gradcam_grid(
            model, test_ds, ci_as_nci, device=device,
            save_path=os.path.join(args.out_dir,
                                   "gradcam_errors_CI_misclassified.png"),
            title="GradCAM — CI-DME Misclassified",
        )

    print(f"\n[DONE] All outputs: {args.out_dir}")


if __name__ == "__main__":
    main()