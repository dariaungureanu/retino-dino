"""
Corina Explainability - t-SNE + GradCAM for multi-label biomarker detection.

Usage:
    python finetune_corina/analyse_corina.py \
        --data_path //home/student/Ungureanu_Daria/corina_dataset \
        --csv /home/student/Ungureanu_Daria/corina_dataset/corina_metadata.csv \
        --model_path saved_models/corina_domain_adapted/best_model.pth \
        --out_dir results/explainability/corina
"""

import argparse
import os

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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from dataset import (
    CorinaDataset, load_corina_splits, get_eval_transform, BIOMARKERS, NUM_LABELS,
)
from model import CorinaModel, load_backbone

THRESHOLD = 0.5



def load_model(model_path, device):
    print(f"loading: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    config = ckpt["config"]

    backbone = load_backbone(config["arch"], config["checkpoint"], device)
    model = CorinaModel(
        backbone=backbone,
        num_labels=ckpt["num_labels"],
        freeze_backbone=(config["unfreeze_last_n"] < 12),
        unfreeze_last_n=config["unfreeze_last_n"],
        head_hidden=config["head_hidden"],
        head_dropout=config["head_dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


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
    """t-SNE colored by biomarker combination (string built from the binary vector)."""
    combo_names = []
    for row in labels:
        active = [BIOMARKERS[i] for i in range(NUM_LABELS) if row[i] == 1]
        combo_names.append("+".join(active) if active else "None")

    unique_combos = sorted(set(combo_names))
    combo_to_idx = {c: i for i, c in enumerate(unique_combos)}
    combo_indices = np.array([combo_to_idx[c] for c in combo_names])

    print(f"[t-SNE] {len(features)} samples, {len(unique_combos)} label combinations...")
    coords = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(features)

    plt.figure(figsize=(14, 10))
    palette = sns.color_palette("husl", len(unique_combos))

    for i, combo in enumerate(unique_combos):
        mask = combo_indices == i
        count = mask.sum()
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    label=f"{combo} ({count})", color=palette[i],
                    alpha=0.7, s=40, edgecolors="white", linewidths=0.3)

    plt.title("t-SNE - Biomarker Combination Feature Space", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, title="biomarkers")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"{out_path}")


class BiomarkerWrapper(nn.Module):
    def __init__(self, model, biomarker_idx):
        super().__init__()
        self.model = model
        self.biomarker_idx = biomarker_idx

    def forward(self, x):
        logits = self.model(x)  # [B, 4]
        return logits[:, self.biomarker_idx:self.biomarker_idx+1]  # [B, 1]


def reshape_transform_vit(tensor):
    result = tensor[:, 1:, :]
    grid = int(np.sqrt(result.size(1)))
    result = result.reshape(tensor.size(0), grid, grid, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def denormalize(img_tensor):
    img = img_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1)


@torch.no_grad()
def collect_predictions(model, loader, device):
    all_probs, all_labels, indices = [], [], []
    idx = 0
    for images, labels in tqdm(loader, desc="Collecting predictions"):
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        bsz = images.size(0)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        indices.extend(range(idx, idx + bsz))
        idx += bsz
    return np.concatenate(all_probs), np.concatenate(all_labels), np.array(indices)


def generate_per_biomarker_gradcam(model, dataset, sample_indices,
                                    save_path, device, title="GradCAM"):
    """For each sample show Original + GradCAM for EACH biomarker (side by side)."""

    if len(sample_indices) == 0:
        print(f"no samples for: {title}")
        return

    for p in model.backbone.parameters():
        p.requires_grad_(True)

    n = len(sample_indices)
    n_cols = 1 + NUM_LABELS

    fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    target_layers = [model.backbone.blocks[-1].norm1]

    for i, idx in enumerate(sample_indices):
        img_tensor, true_labels = dataset[idx]
        img_input = img_tensor.unsqueeze(0).to(device)
        rgb_img = denormalize(img_input)

        with torch.no_grad():
            logits = model(img_input)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        true_str = "+".join([BIOMARKERS[j] for j in range(NUM_LABELS) if true_labels[j] == 1]) or "None"
        pred_str = "+".join([f"{BIOMARKERS[j]}({probs[j]:.2f})" for j in range(NUM_LABELS) if probs[j] >= THRESHOLD]) or "None"

        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title(f"TRUE: {true_str}\nPRED: {pred_str}", fontsize=8)
        axes[i, 0].axis("off")

        for j, bm in enumerate(BIOMARKERS):
            wrapper = BiomarkerWrapper(model, j)
            cam = GradCAM(model=wrapper, target_layers=target_layers,
                          reshape_transform=reshape_transform_vit)

            # wrapper output is single-channel
            cam_map = cam(input_tensor=img_input,
                          targets=[ClassifierOutputTarget(0)])[0]

            vis = show_cam_on_image(rgb_img, cam_map, use_rgb=True)

            true_val = "ok" if true_labels[j] == 1 else "no"
            pred_val = f"{probs[j]:.2f}"

            axes[i, j + 1].imshow(vis)
            axes[i, j + 1].set_title(f"{bm}\nTrue:{true_val} Pred:{pred_val}", fontsize=9)
            axes[i, j + 1].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"{save_path}")


def select_samples_multilabel(probs, labels, indices, biomarker_idx=None,
                               correct=True, topk=6):
    """Select samples for GradCAM. "correct" means the specified biomarker matches truth."""
    preds = (probs >= THRESHOLD).astype(int)

    if biomarker_idx is not None:
        if correct:
            mask = preds[:, biomarker_idx] == labels[:, biomarker_idx]
            # Only keep samples where the biomarker is actually present.
            mask = mask & (labels[:, biomarker_idx] == 1)
        else:
            mask = preds[:, biomarker_idx] != labels[:, biomarker_idx]
        conf = probs[:, biomarker_idx]
    else:
        if correct:
            mask = np.all(preds == labels, axis=1)
        else:
            mask = ~np.all(preds == labels, axis=1)
        conf = np.max(probs, axis=1)

    sel = indices[mask]
    sel_conf = conf[mask]

    if len(sel) == 0:
        return np.array([], dtype=int)

    order = np.argsort(-sel_conf)
    return sel[order[:topk]]

def main():
    parser = argparse.ArgumentParser(description="Corina Explainability")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/explainability/corina")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_tsne", action="store_true")
    parser.add_argument("--skip_gradcam", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model, config = load_model(args.model_path, device)

    _, _, test_df = load_corina_splits(args.csv, args.data_path)
    eval_transform = get_eval_transform(config["img_size"])
    test_ds = CorinaDataset(test_df, args.data_path, eval_transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    if not args.skip_tsne:
        features, labels = extract_features(model, test_loader, device)
        plot_tsne(features, labels,
                  os.path.join(args.out_dir, "tsne_biomarkers.png"))

    if not args.skip_gradcam:
        probs, labels, idx = collect_predictions(model, test_loader, device)

        # Per-biomarker correct detections (implicit localization).
        for j, bm in enumerate(BIOMARKERS):
            correct_samples = select_samples_multilabel(
                probs, labels, idx, biomarker_idx=j,
                correct=True, topk=args.topk,
            )
            generate_per_biomarker_gradcam(
                model, test_ds, correct_samples, device=device,
                save_path=os.path.join(args.out_dir, f"gradcam_correct_{bm}.png"),
                title=f"GradCAM - Correct {bm} Detections (per-biomarker view)",
            )

        # Overall top errors (exact-match failures).
        error_samples = select_samples_multilabel(
            probs, labels, idx, correct=False, topk=args.topk,
        )
        generate_per_biomarker_gradcam(
            model, test_ds, error_samples, device=device,
            save_path=os.path.join(args.out_dir, "gradcam_top_errors.png"),
            title="GradCAM - Top Errors (per-biomarker view)",
        )

        # Per-biomarker errors.
        for j, bm in enumerate(BIOMARKERS):
            err_samples = select_samples_multilabel(
                probs, labels, idx, biomarker_idx=j,
                correct=False, topk=args.topk,
            )
            generate_per_biomarker_gradcam(
                model, test_ds, err_samples, device=device,
                save_path=os.path.join(args.out_dir, f"gradcam_errors_{bm}.png"),
                title=f"GradCAM - {bm} Detection Errors",
            )

    print(f"\nall outputs: {args.out_dir}")


if __name__ == "__main__":
    main()