"""

Loads any checkpoint from the fine-tuning pipeline and generates:
  1. t-SNE plots (disease + condition) of backbone features
  2. GradCAM on top-K most confident errors (disease + condition)
  3. GradCAM on per-class errors (optional, specify --gradcam_class)

Usage:
    # Full analysis on Run C
    python finetune_octdl/analyse_explainability.py \
        --data_path /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --checkpoint_dir saved_models/run_C_unfreeze2 \
        --out_dir results/explainability/run_finetuning_C

    # Only t-SNE
    python finetune_octdl/analyse_explainability.py \
        --data_path /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --checkpoint_dir saved_models/run_C_unfreeze2 \
        --out_dir results/explainability/run_finetuning_C \
        --skip_gradcam

    # Only GradCAM for a specific class
    python finetune_octdl/analyse_explainability.py \
        --data_path /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --checkpoint_dir saved_models/run_C_unfreeze2 \
        --out_dir results/explainability/run_finetuning_C \
        --skip_tsne --gradcam_class AMD

"""

import argparse
import os
import sys
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import (
    IGNORE_INDEX, OCTDLMultiTaskDataset, get_data_splits, get_eval_transform,
)
from model import OCTDLMultiTaskModel, load_backbone


def load_model_from_checkpoint(model_path, device):
    print(f"loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device)

    config = ckpt["config"]
    disease_map = ckpt["disease_map"]
    condition_map = ckpt["condition_map"]

    print(f"arch={config['arch']}  unfreeze={config['unfreeze_last_n']}  "
          f"epoch={ckpt['epoch']}  val_f1={ckpt['val_disease_f1']:.4f}")

    backbone = load_backbone(config["arch"], config["checkpoint"], device)
    model = OCTDLMultiTaskModel(
        backbone=backbone,
        num_diseases=len(disease_map),
        num_conditions=len(condition_map),
        freeze_backbone=(config["unfreeze_last_n"] < 12),
        unfreeze_last_n=config["unfreeze_last_n"],
        head_hidden=config["head_hidden"],
        head_dropout=config["head_dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def extract_features_and_labels(model, loader, device):
    """
    Extract CLS-token features from the backbone for all images.
    Returns: features [N, D], disease_labels [N], condition_labels [N]
    """
    all_feats, all_labels_d, all_labels_c = [], [], []

    for images, labels_d, labels_c in tqdm(loader, desc="Extracting features"):
        images = images.to(device, non_blocking=True)

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
        all_labels_d.append(labels_d.numpy())
        all_labels_c.append(labels_c.numpy())

    return (
        np.concatenate(all_feats),
        np.concatenate(all_labels_d),
        np.concatenate(all_labels_c),
    )


def plot_tsne(features, labels, label_map, title, save_path,
              perplexity=30, random_state=42):
    """
    Compute t-SNE projection and plot colored by class.
    """
    # Filter out ignore_index labels (for condition)
    valid_mask = labels != IGNORE_INDEX
    features = features[valid_mask]
    labels = labels[valid_mask]

    if len(features) == 0:
        print(f"no valid samples for t-SNE: {title}")
        return

    print(f"Computing projection for {len(features)} samples...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    coords = tsne.fit_transform(features)

    plt.figure(figsize=(12, 9))
    unique_labels = np.unique(labels)
    palette = sns.color_palette("bright", len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = label_map.get(int(lbl), f"Class {lbl}")
        plt.scatter(
            coords[mask, 0], coords[mask, 1],
            label=name, color=palette[i], alpha=0.7, s=40, edgecolors="white", linewidths=0.3,
        )

    plt.title(title, fontsize=15, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="class", fontsize=10)
    plt.xlabel("t-SNE 1", fontsize=11)
    plt.ylabel("t-SNE 2", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"{save_path}")

class TaskHeadWrapper(nn.Module):
    def __init__(self, model, head_index):
        super().__init__()
        self.model = model
        self.head_index = head_index

    def forward(self, x):
        outputs = self.model(x)
        return outputs[self.head_index]


def reshape_transform_vit(tensor):
    result = tensor[:, 1:, :]  # drop CLS token
    grid_size = int(np.sqrt(result.size(1)))
    result = result.reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))
    result = result.permute(0, 3, 1, 2)  # [B, C, H, W]
    return result


def denormalize(img_tensor):
    """Undo ImageNet normalization, returning a [0,1] RGB numpy array."""
    img = img_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    return np.clip(img, 0, 1)


@torch.no_grad()
def collect_predictions(model, loader, device, head_index=0):
    """
    Run inference and collect predictions for one head.
    """
    model.eval()
    softmax = nn.Softmax(dim=1)
    y_true, y_pred, y_conf, indices = [], [], [], []

    running_idx = 0
    for images, labels_d, labels_c in tqdm(loader, desc=f"Collecting predictions (head={head_index})"):
        images = images.to(device)
        logits_d, logits_c = model(images)
        logits = logits_d if head_index == 0 else logits_c
        labels = labels_d if head_index == 0 else labels_c

        probs = softmax(logits)
        conf, pred = torch.max(probs, dim=1)

        bsz = images.size(0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        y_conf.extend(conf.cpu().numpy().tolist())
        indices.extend(range(running_idx, running_idx + bsz))
        running_idx += bsz

    return np.array(y_true), np.array(y_pred), np.array(y_conf), np.array(indices)


def select_top_errors(y_true, y_pred, y_conf, indices, topk=8, class_idx=None):
    """
    Select top-K most confident wrong predictions.
    If class_idx is given, only errors where true label == class_idx.
    """
    if class_idx is not None:
        mask = (y_true == class_idx) & (y_pred != class_idx)
    else:
        mask = (y_true != y_pred) & (y_true != IGNORE_INDEX)

    wrong_indices = indices[mask]
    wrong_conf = y_conf[mask]

    if len(wrong_indices) == 0:
        return np.array([], dtype=int)

    order = np.argsort(-wrong_conf)  # most confident errors first
    return wrong_indices[order[:topk]]


def select_top_correct(y_true, y_pred, y_conf, indices, topk=8, class_idx=None):
    """Top-K most confident correct predictions, optionally restricted to one class."""
    if class_idx is not None:
        mask = (y_true == class_idx) & (y_pred == class_idx)
    else:
        mask = (y_true == y_pred) & (y_true != IGNORE_INDEX)

    sel = indices[mask]
    confs = y_conf[mask]

    if len(sel) == 0:
        return np.array([], dtype=int)

    order = np.argsort(-confs)
    return sel[order[:topk]]


def generate_gradcam_grid(
    model, dataset, sample_indices, head_index, label_map,
    task_name, save_path, device,
):
    """GradCAM grid: original | CAM for predicted | CAM for true."""

    if len(sample_indices) == 0:
        print(f"no samples for GradCAM: {task_name}")
        return

    for p in model.backbone.parameters():
        p.requires_grad_(True)

    wrapper = TaskHeadWrapper(model, head_index)
    target_layers = [model.backbone.blocks[-1].norm1]
    cam = GradCAM(model=wrapper, target_layers=target_layers,
                  reshape_transform=reshape_transform_vit)

    softmax = nn.Softmax(dim=1)
    n = len(sample_indices)

    fig, axes = plt.subplots(n, 3, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(sample_indices):
        img_tensor, label_d, label_c = dataset[idx]
        img_input = img_tensor.unsqueeze(0).to(device)

        true_label = int(label_d if head_index == 0 else label_c)
        if true_label == IGNORE_INDEX:
            continue

        with torch.no_grad():
            logits = wrapper(img_input)
            probs = softmax(logits)
            pred_class = int(torch.argmax(probs, dim=1).item())
            pred_conf = float(torch.max(probs).item())

        true_name = label_map.get(true_label, f"?{true_label}")
        pred_name = label_map.get(pred_class, f"?{pred_class}")
        rgb_img = denormalize(img_input)

        cam_pred = cam(input_tensor=img_input, targets=[ClassifierOutputTarget(pred_class)])[0]
        cam_true = cam(input_tensor=img_input, targets=[ClassifierOutputTarget(true_label)])[0]

        vis_pred = show_cam_on_image(rgb_img, cam_pred, use_rgb=True)
        vis_true = show_cam_on_image(rgb_img, cam_true, use_rgb=True)

        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title(f"TRUE: {true_name}\nPRED: {pred_name} ({pred_conf:.2f})", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(vis_pred)
        axes[i, 1].set_title(f"CAM -> Predicted: {pred_name}", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(vis_true)
        axes[i, 2].set_title(f"CAM -> True: {true_name}", fontsize=10)
        axes[i, 2].axis("off")

    fig.suptitle(f"GradCAM Error Analysis - {task_name}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"{save_path}")


def main():
    parser = argparse.ArgumentParser(description="Explainability: t-SNE + GradCAM")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="results/explainability")

    parser.add_argument("--skip_tsne", action="store_true")
    parser.add_argument("--skip_gradcam", action="store_true")
    parser.add_argument("--gradcam_topk", type=int, default=8,
                        help="Number of top errors to visualize per task")
    parser.add_argument("--gradcam_class", type=str, default=None,
                        help="Generate per-class error GradCAM for this class name (e.g., AMD)")
    parser.add_argument("--tsne_perplexity", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    if args.model_path:
        model_path = args.model_path
    elif args.checkpoint_dir:
        model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    else:
        raise ValueError("Provide --model_path or --checkpoint_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    model, ckpt = load_model_from_checkpoint(model_path, device)
    config = ckpt["config"]
    disease_map = ckpt["disease_map"]
    condition_map = ckpt["condition_map"]
    inv_disease = {v: k for k, v in disease_map.items()}
    inv_condition = {v: k for k, v in condition_map.items()}

    # Load test data
    csv_path = os.path.join(args.data_path, "OCTDL_clean_metadata.csv")
    _, _, test_df, _, _ = get_data_splits(csv_path)

    eval_transform = get_eval_transform(config["img_size"])
    test_ds = OCTDLMultiTaskDataset(
        test_df, args.data_path, eval_transform, disease_map, condition_map,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"test set: {len(test_ds)} images")

    if not args.skip_tsne:
        print("T-SNE VISUALIZATION")
        features, labels_d, labels_c = extract_features_and_labels(
            model, test_loader, device,
        )

        # Disease t-SNE
        plot_tsne(
            features, labels_d, inv_disease,
            title="t-SNE - Disease Feature Space (Fine-tuned)",
            save_path=os.path.join(args.out_dir, "tsne_disease.png"),
            perplexity=args.tsne_perplexity,
        )

        # Condition t-SNE
        plot_tsne(
            features, labels_c, inv_condition,
            title="t-SNE - Condition Feature Space (Fine-tuned)",
            save_path=os.path.join(args.out_dir, "tsne_condition.png"),
            perplexity=args.tsne_perplexity,
        )

    if not args.skip_gradcam:
        print("gradcam error analysis")
        # Disease head: top errors and top correct predictions
        y_true_d, y_pred_d, y_conf_d, idx_d = collect_predictions(
            model, test_loader, device, head_index=0,
        )
        top_errors_d = select_top_errors(
            y_true_d, y_pred_d, y_conf_d, idx_d, topk=args.gradcam_topk,
        )
        generate_gradcam_grid(
            model, test_ds, top_errors_d, head_index=0, label_map=inv_disease,
            task_name="Disease", device=device,
            save_path=os.path.join(args.out_dir, "gradcam_disease_top_errors.png"),
        )
        top_correct_d = select_top_correct(
            y_true_d, y_pred_d, y_conf_d, idx_d, topk=args.gradcam_topk,
        )
        generate_gradcam_grid(
            model, test_ds, top_correct_d, head_index=0, label_map=inv_disease,
            task_name="Disease - Correct", device=device,
            save_path=os.path.join(args.out_dir, "gradcam_disease_top_correct.png"),
        )

        # Condition head: top errors and top correct predictions
        y_true_c, y_pred_c, y_conf_c, idx_c = collect_predictions(
            model, test_loader, device, head_index=1,
        )
        top_errors_c = select_top_errors(
            y_true_c, y_pred_c, y_conf_c, idx_c, topk=args.gradcam_topk,
        )
        generate_gradcam_grid(
            model, test_ds, top_errors_c, head_index=1, label_map=inv_condition,
            task_name="Condition", device=device,
            save_path=os.path.join(args.out_dir, "gradcam_condition_top_errors.png"),
        )
        top_correct_c = select_top_correct(
            y_true_c, y_pred_c, y_conf_c, idx_c, topk=args.gradcam_topk,
        )
        generate_gradcam_grid(
            model, test_ds, top_correct_c, head_index=1, label_map=inv_condition,
            task_name="Condition - Correct", device=device,
            save_path=os.path.join(args.out_dir, "gradcam_condition_top_correct.png"),
        )

        # Per-class errors (optional)
        if args.gradcam_class:
            cls_name = args.gradcam_class
            if cls_name in disease_map:
                cls_errors = select_top_errors(
                    y_true_d, y_pred_d, y_conf_d, idx_d,
                    topk=args.gradcam_topk, class_idx=disease_map[cls_name],
                )
                generate_gradcam_grid(
                    model, test_ds, cls_errors, head_index=0, label_map=inv_disease,
                    task_name=f"Disease (TRUE={cls_name})", device=device,
                    save_path=os.path.join(args.out_dir, f"gradcam_disease_errors_{cls_name}.png"),
                )
            elif cls_name in condition_map:
                cls_errors = select_top_errors(
                    y_true_c, y_pred_c, y_conf_c, idx_c,
                    topk=args.gradcam_topk, class_idx=condition_map[cls_name],
                )
                generate_gradcam_grid(
                    model, test_ds, cls_errors, head_index=1, label_map=inv_condition,
                    task_name=f"Condition (TRUE={cls_name})", device=device,
                    save_path=os.path.join(args.out_dir, f"gradcam_condition_errors_{cls_name}.png"),
                )
            else:
                print(f"class '{cls_name}' not found in disease or condition maps")

    print(f"\nall outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()