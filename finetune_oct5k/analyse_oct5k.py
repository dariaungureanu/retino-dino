"""

Generates:
  1. t-SNE colored by dominant biomarker
  2. GradCAM per biomarker on correct detections WITH bbox overlay
  3. GradCAM on errors
  4. IoU summary: quantitative measure of heatmap-bbox overlap

Usage:
    python finetune_oct5k/analyse_oct5k.py \
        --data_path /home/student/Ungureanu_Daria/oct5k \
        --csv /home/student/Ungureanu_Daria/oct5k/oct5k_metadata.csv \
        --bbox_csv /home/student/Ungureanu_Daria/oct5k/oct5k_bboxes.csv \
        --model_path saved_models/oct5k_domain_adapted/best_model.pth \
        --out_dir results/explainability/oct5k
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import (
    OCT5kDataset, load_oct5k_splits, get_eval_transform,
    BIOMARKERS, SHORT_NAMES,
)
from model import OCT5kModel, load_backbone

THRESHOLD = 0.5

def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    config = ckpt["config"]
    active_biomarkers = ckpt["biomarkers"]

    backbone = load_backbone(config["arch"], config["checkpoint"], device)
    model = OCT5kModel(
        backbone=backbone,
        num_labels=ckpt["num_labels"],
        freeze_backbone=(config["unfreeze_last_n"] < 12),
        unfreeze_last_n=config["unfreeze_last_n"],
        head_hidden=config["head_hidden"],
        head_dropout=config["head_dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}, val_f1={ckpt['val_f1_macro']:.4f}")
    print(f"Active biomarkers: {[SHORT_NAMES[b] for b in active_biomarkers]}")
    return model, config, active_biomarkers


def load_bboxes_for_image(bbox_df, image_csv_path):
    """
    Get all bounding boxes for a specific image
    """
    rows = bbox_df[bbox_df["image"] == image_csv_path]
    bboxes = {}
    for _, row in rows.iterrows():
        cls = row["class"]
        box = (row["xmin"], row["ymin"], row["xmax"], row["ymax"])
        if cls not in bboxes:
            bboxes[cls] = []
        bboxes[cls].append(box)
    return bboxes


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


def plot_tsne(features, labels, active_biomarkers, out_path):
    combo_names = []
    for row in labels:
        active = [SHORT_NAMES[active_biomarkers[i]]
                  for i in range(len(active_biomarkers)) if row[i] == 1]
        combo_names.append("+".join(active) if active else "None")

    unique = sorted(set(combo_names))
    combo_idx = {c: i for i, c in enumerate(unique)}
    indices = np.array([combo_idx[c] for c in combo_names])

    coords = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1)).fit_transform(features)

    plt.figure(figsize=(14, 10))
    palette = sns.color_palette("husl", len(unique))
    for i, combo in enumerate(unique):
        mask = indices == i
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    label=f"{combo} ({mask.sum()})", color=palette[i],
                    alpha=0.7, s=40, edgecolors="white", linewidths=0.3)

    plt.title("t-SNE - OCT5k Biomarker Feature Space", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, title="Biomarkers")
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
        logits = self.model(x)
        return logits[:, self.biomarker_idx:self.biomarker_idx + 1]


def reshape_transform_vit(tensor):
    result = tensor[:, 1:, :]
    grid = int(np.sqrt(result.size(1)))
    result = result.reshape(tensor.size(0), grid, grid, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def denormalize(img_tensor):
    img = img_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1)


def compute_heatmap_bbox_iou(cam_map, bboxes, orig_w, orig_h, cam_threshold=0.5):
    """
    Compute IoU between thresholded GradCAM heatmap and bounding boxes.

    cam_map: [H, W] heatmap normalized 0-1 (at 224x224 resolution)
    bboxes: list of (xmin, ymin, xmax, ymax) in ORIGINAL image pixel coordinates
    orig_w, orig_h: original image dimensions (for scaling bboxes to cam resolution)

    Returns: IoU score (0-1)
    """
    if not bboxes:
        return 0.0

    cam_h, cam_w = cam_map.shape
    cam_binary = (cam_map >= cam_threshold).astype(np.float32)

    bbox_mask = np.zeros((cam_h, cam_w), dtype=np.float32)

    sx = cam_w / orig_w
    sy = cam_h / orig_h

    for (xmin, ymin, xmax, ymax) in bboxes:
        x1 = max(0, int(xmin * sx))
        y1 = max(0, int(ymin * sy))
        x2 = min(cam_w, int(xmax * sx))
        y2 = min(cam_h, int(ymax * sy))
        bbox_mask[y1:y2, x1:x2] = 1.0

    intersection = (cam_binary * bbox_mask).sum()
    union = ((cam_binary + bbox_mask) > 0).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


@torch.no_grad()
def collect_predictions(model, loader, device, active_biomarkers):
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


def generate_gradcam_with_bbox(
    model, dataset, test_df, bbox_df, sample_indices,
    biomarker_idx, biomarker_name, active_biomarkers,
    save_path, device, title="GradCAM",
):
    """
    GradCAM for one biomarker with expert bounding box overlay + IoU score.
    """
    if len(sample_indices) == 0:
        print(f"No samples for: {title}")
        return []

    for p in model.backbone.parameters():
        p.requires_grad_(True)

    wrapper = BiomarkerWrapper(model, biomarker_idx)
    target_layers = [model.backbone.blocks[-1].norm1]
    cam = GradCAM(model=wrapper, target_layers=target_layers,
                  reshape_transform=reshape_transform_vit)

    n = len(sample_indices)
    all_ious = []

    fig, axes = plt.subplots(n, 3, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(sample_indices):
        img_tensor, true_labels = dataset[idx]
        img_input = img_tensor.unsqueeze(0).to(device)
        rgb_img = denormalize(img_input)

        with torch.no_grad():
            logits = model(img_input)
            prob = torch.sigmoid(logits[0, biomarker_idx]).item()

        true_val = "ok" if true_labels[biomarker_idx] == 1 else "no"

        cam_map = cam(input_tensor=img_input,
                      targets=[ClassifierOutputTarget(0)])[0]
        vis_cam = show_cam_on_image(rgb_img, cam_map, use_rgb=True)

        row = test_df.iloc[idx]
        image_csv = row["image_csv"]
        bboxes = load_bboxes_for_image(bbox_df, image_csv)
        biomarker_bboxes = bboxes.get(biomarker_name, [])

        orig_img = Image.open(os.path.join(dataset.root_dir, row["image"]))
        ow, oh = orig_img.size  # width, height

        iou = compute_heatmap_bbox_iou(cam_map, biomarker_bboxes, ow, oh)
        all_ious.append(iou)

        cam_h, cam_w = rgb_img.shape[:2]
        sx = cam_w / ow
        sy = cam_h / oh

        axes[i, 0].imshow(rgb_img)
        for (xmin, ymin, xmax, ymax) in biomarker_bboxes:
            rect = patches.Rectangle(
                (xmin * sx, ymin * sy), (xmax - xmin) * sx, (ymax - ymin) * sy,
                linewidth=2, edgecolor='lime', facecolor='none',
            )
            axes[i, 0].add_patch(rect)
        axes[i, 0].set_title(f"{SHORT_NAMES[biomarker_name]} True:{true_val} Pred:{prob:.2f}\n"
                              f"BBoxes: {len(biomarker_bboxes)}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(vis_cam)
        axes[i, 1].set_title(f"GradCAM -> {SHORT_NAMES[biomarker_name]}", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(vis_cam)
        for (xmin, ymin, xmax, ymax) in biomarker_bboxes:
            rect = patches.Rectangle(
                (xmin * sx, ymin * sy), (xmax - xmin) * sx, (ymax - ymin) * sy,
                linewidth=2, edgecolor='lime', facecolor='none', linestyle='--',
            )
            axes[i, 2].add_patch(rect)
        iou_color = "green" if iou >= 0.3 else "orange" if iou >= 0.1 else "red"
        axes[i, 2].set_title(f"GradCAM + BBox | IoU={iou:.2f}", fontsize=10,
                              color=iou_color, fontweight="bold")
        axes[i, 2].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"{save_path}")

    if all_ious:
        mean_iou = np.mean(all_ious)
        print(f"  {SHORT_NAMES[biomarker_name]} IoU: mean={mean_iou:.3f}, "
              f"per-image={[f'{x:.2f}' for x in all_ious]}")

    return all_ious


def select_samples(probs, labels, indices, biomarker_idx,
                   correct=True, topk=5):
    preds = (probs >= THRESHOLD).astype(int)

    if correct:
        mask = (preds[:, biomarker_idx] == labels[:, biomarker_idx]) & (labels[:, biomarker_idx] == 1)
    else:
        mask = preds[:, biomarker_idx] != labels[:, biomarker_idx]

    sel = indices[mask]
    sel_conf = probs[:, biomarker_idx][mask]

    if len(sel) == 0:
        return np.array([], dtype=int)

    order = np.argsort(-sel_conf)
    return sel[order[:topk]]


def main():
    parser = argparse.ArgumentParser(description="OCT5k Explainability")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--bbox_csv", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/explainability/oct5k")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_tsne", action="store_true")
    parser.add_argument("--skip_gradcam", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model, config, active_biomarkers = load_model(args.model_path, device)

    _, _, test_df, _ = load_oct5k_splits(
        args.csv, args.data_path,
        drop_rare=config.get("drop_rare", 15),
    )
    eval_transform = get_eval_transform(config["img_size"])
    test_ds = OCT5kDataset(test_df, args.data_path, eval_transform, active_biomarkers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    bbox_df = pd.read_csv(args.bbox_csv)
    print(f"Loaded {len(bbox_df)} bounding boxes")

    # t-SNE
    if not args.skip_tsne:
        features, labels = extract_features(model, test_loader, device)
        plot_tsne(features, labels, active_biomarkers,
                  os.path.join(args.out_dir, "tsne_oct5k.png"))

    # GradCAM with bbox validation
    if not args.skip_gradcam:
        probs, labels, idx = collect_predictions(model, test_loader, device, active_biomarkers)

        all_iou_results = {}

        for j, bm in enumerate(active_biomarkers):
            correct = select_samples(probs, labels, idx,
                                     biomarker_idx=j, correct=True, topk=args.topk)
            ious_correct = generate_gradcam_with_bbox(
                model, test_ds, test_df, bbox_df, correct,
                biomarker_idx=j, biomarker_name=bm,
                active_biomarkers=active_biomarkers,
                save_path=os.path.join(args.out_dir, f"gradcam_correct_{SHORT_NAMES[bm]}.png"),
                device=device,
                title=f"GradCAM + BBox - Correct {SHORT_NAMES[bm]} Detections",
            )

            errors = select_samples(probs, labels, idx,
                                    biomarker_idx=j, correct=False, topk=args.topk)
            ious_errors = generate_gradcam_with_bbox(
                model, test_ds, test_df, bbox_df, errors,
                biomarker_idx=j, biomarker_name=bm,
                active_biomarkers=active_biomarkers,
                save_path=os.path.join(args.out_dir, f"gradcam_errors_{SHORT_NAMES[bm]}.png"),
                device=device,
                title=f"GradCAM + BBox - {SHORT_NAMES[bm]} Errors",
            )

            if ious_correct:
                all_iou_results[f"{SHORT_NAMES[bm]}_correct"] = np.mean(ious_correct)
            if ious_errors:
                all_iou_results[f"{SHORT_NAMES[bm]}_errors"] = np.mean(ious_errors)

        print("  GRADCAM-BBOX IoU SUMMARY")
        for key, val in sorted(all_iou_results.items()):
            print(f"  {key:>20}: IoU = {val:.3f}")
        if all_iou_results:
            overall_mean = np.mean(list(all_iou_results.values()))
            correct_only = [v for k, v in all_iou_results.items() if "correct" in k]
            if correct_only:
                print(f"  {'Mean (correct)':>20}: IoU = {np.mean(correct_only):.3f}")
            print(f"  {'Overall Mean':>20}: IoU = {overall_mean:.3f}")

        iou_path = os.path.join(args.out_dir, "iou_results.json")
        import json
        with open(iou_path, "w") as f:
            json.dump(all_iou_results, f, indent=2)
        print(f"{iou_path}")

    print(f"\n[DONE] All outputs: {args.out_dir}")


if __name__ == "__main__":
    main()