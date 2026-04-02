"""
Method — Patch PCA Visualization for DINOv2 Backbone Analysis
================================================================
Purpose:
    Visualize what a DINOv2 backbone has learned at the patch-token level.
    For each image, we extract all patch embeddings from the last layer,
    project them to 3 dimensions via PCA, and map those to RGB channels.
    This produces a color map showing how the model internally separates
    different spatial regions (e.g., retinal tissue vs background vs fluid).

What this tells me:
    - Whether the backbone distinguishes retinal tissue from background
    - Whether it captures internal layer structure (RPE, ILM, fluid pockets)
    - NOT attention or importance — just representational structure

Usage:
    # Domain-adapted checkpoint
    python analyse_pretrain/method_patch_pca.py \
        --arch dinov2_vits14 \
        --checkpoint /home/student/Ungureanu_Daria/antrenare_oct_rezultate/model_final.rank_0.pth \
        --csv /home/student/Ungureanu_Daria/OCTDL_Cleaned/OCTDL_clean_metadata.csv \
        --image_root /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --split_col label_disease --split AMD \
        --path_col file_name --label_col label_disease \
        --img_size 518 --max_images 5 \
        --out_dir results/patch_pca/domain_adapted

    # Baseline (original ImageNet pretrained — no --checkpoint)
    python analyse_pretrain/method_patch_pca.py \
        --arch dinov2_vits14 \
        --csv /home/student/Ungureanu_Daria/OCTDL_Cleaned/OCTDL_clean_metadata.csv \
        --image_root /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --split_col label_disease --split AMD \
        --path_col file_name --label_col label_disease \
        --img_size 518 --max_images 5 \
        --out_dir results/patch_pca/imagenet_baseline

    # Run multiple diseases in aloop (adjust paths as needed)
    for disease in AMD DME ERM NO RVO VID; do
        python analyse_pretrain/method_patch_pca.py \
            --arch dinov2_vits14 \
            --checkpoint /home/student/Ungureanu_Daria/antrenare_oct_rezultate/model_final.rank_0.pth \
            --csv /home/student/Ungureanu_Daria/OCTDL_Cleaned/OCTDL_clean_metadata.csv \
            --image_root /home/student/Ungureanu_Daria/OCTDL_Cleaned \
            --split_col label_disease --split $disease \
            --max_images 3 \
            --out_dir results/patch_pca/domain_adapted/$disease
    done
"""

import argparse
import json
import os
from typing import List, Dict

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyse_shared import (
    load_model, build_samples, OCTDataset,
    add_common_args, get_device, validate_img_size,
)


# ──────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def get_patch_tokens(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Extract patch-level embeddings from the DINOv2 backbone.
    Returns [B, N, D] where N = number of patches, D = embedding dim.
    """
    feats = model.forward_features(x)

    if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
        return feats["x_norm_patchtokens"]
    elif isinstance(feats, dict) and "x_prenorm" in feats:
        return feats["x_prenorm"][:, 1:, :]  # strip CLS
    elif torch.is_tensor(feats):
        return feats[:, 1:, :]  # strip CLS
    else:
        raise RuntimeError(
            f"Unexpected forward_features output: {type(feats)}, "
            f"keys={feats.keys() if isinstance(feats, dict) else 'N/A'}"
        )


# ──────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────

def save_pca_map(
        raw_img_tensor: torch.Tensor,
        tokens: np.ndarray,
        out_path: str,
        label: str,
        image_name: str,
        grid_h: int,
        grid_w: int,
        overlay_alpha: float = 0.55,
) -> np.ndarray:
    """
    Fit PCA(3) on one image's patch tokens, produce RGB map, save figure.
    Returns explained_variance_ratio_ array.
    """
    n_patches, dim = tokens.shape
    assert n_patches == grid_h * grid_w, (
        f"Token count {n_patches} != grid {grid_h}x{grid_w}={grid_h * grid_w}"
    )

    # PCA projection
    pca = PCA(n_components=3)
    projected = pca.fit_transform(tokens).reshape(grid_h, grid_w, 3)

    # Normalize to [0, 1]
    projected = (projected - projected.min()) / (projected.max() - projected.min() + 1e-8)

    # Raw image as numpy [H, W, 3]
    raw_np = raw_img_tensor.permute(1, 2, 0).cpu().numpy()
    h, w = raw_np.shape[:2]

    # Upsample PCA map to image resolution
    pca_resized = np.asarray(
        Image.fromarray((projected * 255).astype(np.uint8)).resize(
            (w, h), resample=Image.BILINEAR
        )
    ).astype(np.float32) / 255.0

    # Blend
    alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
    overlay = np.clip((1.0 - alpha) * raw_np + alpha * pca_resized, 0.0, 1.0)

    # Variance string
    evr = pca.explained_variance_ratio_
    var_str = f"var: {evr[0]:.1%}, {evr[1]:.1%}, {evr[2]:.1%}"

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(raw_np)
    axes[0].set_title(f"Input — {label}")
    axes[0].axis("off")

    axes[1].imshow(pca_resized)
    axes[1].set_title(f"Patch PCA (RGB)  ({var_str})")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (a={alpha:.2f})")
    axes[2].axis("off")

    fig.suptitle(
        f"{image_name}  |  {grid_h}x{grid_w} patches  |  dim={dim}",
        fontsize=9, color="gray",
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return evr


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Patch PCA visualization for DINOv2 backbone"
    )
    add_common_args(ap)
    ap.add_argument("--overlay_alpha", type=float, default=0.55)
    ap.add_argument("--out_dir", default="results/patch_pca")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = get_device()
    grid_side = validate_img_size(args.img_size)

    # ── Data ──
    samples = build_samples(
        args.csv, args.image_root,
        args.split_col, args.split,
        args.path_col, args.label_col,
    )
    if args.max_images > 0:
        samples = samples[:args.max_images]
        print(f"[DATA] Using first {len(samples)} images")

    ds = OCTDataset(samples, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    # ── Model ──
    model = load_model(args.arch, args.checkpoint, device)

    # Shape check
    with torch.no_grad():
        test_tokens = get_patch_tokens(model, ds[0][0].unsqueeze(0).to(device))
        print(f"[INFO] Patch tokens shape: {test_tokens.shape}  "
              f"(expected [1, {grid_side ** 2}, *])")

    # ── Process ──
    records: List[Dict] = []
    all_variance = []

    for i, (x, raw, paths, labels) in enumerate(tqdm(dl, desc="Patch PCA")):
        x = x.to(device)
        tokens = get_patch_tokens(model, x)[0].cpu().numpy()  # [N, D]

        image_stem = os.path.splitext(os.path.basename(paths[0]))[0]
        out_png = os.path.join(args.out_dir, f"pca_{i:04d}_{labels[0]}_{image_stem}.png")

        evr = save_pca_map(
            raw_img_tensor=raw[0],
            tokens=tokens,
            out_path=out_png,
            label=labels[0],
            image_name=os.path.basename(paths[0]),
            grid_h=grid_side,
            grid_w=grid_side,
            overlay_alpha=args.overlay_alpha,
        )
        all_variance.append(evr)

        records.append({
            "index": i,
            "image_path": paths[0],
            "image_name": os.path.basename(paths[0]),
            "label": labels[0],
            "output_path": out_png,
            "n_patches": int(tokens.shape[0]),
            "embed_dim": int(tokens.shape[1]),
            "explained_variance": [float(v) for v in evr],
        })

    # ── Summary ──
    avg_var = np.mean(all_variance, axis=0)
    print(f"\n[RESULT] Processed {len(records)} images")
    print(f"[RESULT] Avg explained variance (3 PCA components): "
          f"{avg_var[0]:.1%}, {avg_var[1]:.1%}, {avg_var[2]:.1%}  "
          f"(total: {avg_var.sum():.1%})")

    checkpoint_label = args.checkpoint or "ImageNet baseline (no checkpoint)"
    result = {
        "method": "patch_pca",
        "checkpoint": checkpoint_label,
        "arch": args.arch,
        "img_size": args.img_size,
        "split_col": args.split_col,
        "split": args.split,
        "num_images": len(records),
        "avg_explained_variance": [float(v) for v in avg_var],
        "records": records,
    }

    out_json = args.out_json or os.path.join(args.out_dir, "results.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] JSON log saved: {out_json}")


if __name__ == "__main__":
    main()