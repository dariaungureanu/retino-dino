"""
Method — CLS-Patch Cosine Similarity
=======================================
Purpose:
    Measure how much each spatial patch contributes to the global [CLS]
    representation. For each image, we compute the cosine similarity
    between the normalized CLS token and every normalized patch token
    from the last transformer layer.

What this tells me:
    - Which spatial regions drive the global image representation
    - High similarity = that patch is "important" to the CLS summary
    - For OCT: retinal tissue should light up; background should be dark

Usage:
    # Domain-adapted checkpoint
    python analyse_pretrain/method_cls_similarity.py \
        --arch dinov2_vits14 \
        --checkpoint /home/student/Ungureanu_Daria/antrenare_oct_v2/model_final.rank_0.pth \
        --csv /home/student/Ungureanu_Daria/OCTDL_Cleaned/OCTDL_clean_metadata.csv \
        --image_root /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --split_col label_disease --split AMD \
        --path_col file_name --label_col label_disease \
        --img_size 518 --max_images 5 \
        --out_dir results/cls_similarity/domain_adapted

    # Baseline (original ImageNet pretrained — no --checkpoint)
    python analyse_pretrain/method_cls_similarity \
        --arch dinov2_vits14 \
        --csv /home/student/Ungureanu_Daria/OCTDL_Cleaned/OCTDL_clean_metadata.csv \
        --image_root /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --split_col label_disease --split AMD \
        --path_col file_name --label_col label_disease \
        --img_size 518 --max_images 5 \
        --out_dir results/cls_similarity/imagenet_baseline

    # Run multiple diseases in aloop (adjust paths as needed)
    for disease in AMD DME ERM NO RVO VID; do
        python analyse_pretrain/method_cls_similarity\
            --arch dinov2_vits14 \
            --checkpoint /home/student/Ungureanu_Daria/antrenare_oct_v2/model_final.rank_0.pth \
            --csv /home/student/Ungureanu_Daria/OCTDL_Cleaned/OCTDL_clean_metadata.csv \
            --image_root /home/student/Ungureanu_Daria/OCTDL_Cleaned \
            --split_col label_disease --split $disease \
            --max_images 2 \
            --out_dir results/cls_similarity/domain_adapted/$disease
    done

"""

import argparse
import json
import os
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyse_shared import (
    load_model, build_samples, OCTDataset, Sample,
    add_common_args, get_device, validate_img_size,
)


# ──────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_cls_patch_similarity(
    model: torch.nn.Module, x: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarity between the [CLS] token and every patch token.

    Uses forward_features() which returns normalized tokens from the
    last layer. Both CLS and patch tokens are L2-normalized before
    computing dot product (= cosine similarity).

    Args:
        model: DINOv2 backbone (from torch.hub)
        x:     Input tensor [B, 3, H, W]

    Returns:
        Similarity map [B, N] where N = number of patches.
        Values in [-1, 1], higher = patch more similar to CLS.
    """
    feats = model.forward_features(x)

    if not isinstance(feats, dict):
        raise RuntimeError(
            f"Expected dict from forward_features, got {type(feats)}"
        )

    required = {"x_norm_clstoken", "x_norm_patchtokens"}
    if not required.issubset(feats.keys()):
        available = set(feats.keys())
        raise RuntimeError(
            f"forward_features missing required keys.\n"
            f"  Required: {required}\n"
            f"  Available: {available}"
        )

    cls_token = feats["x_norm_clstoken"]       # [B, D]
    patch_tokens = feats["x_norm_patchtokens"]  # [B, N, D]

    # Both are already L2-normalized by DINOv2's head,
    # but we normalize again to be safe (idempotent if already unit-norm)
    cls_token = F.normalize(cls_token, dim=-1)          # [B, D]
    patch_tokens = F.normalize(patch_tokens, dim=-1)    # [B, N, D]

    # Cosine similarity = dot product of unit vectors
    # [B, N, D] × [B, D, 1] -> [B, N, 1] -> [B, N]
    similarity = torch.bmm(
        patch_tokens, cls_token.unsqueeze(-1)
    ).squeeze(-1)  # [B, N]

    return similarity


# ──────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────

def save_similarity_map(
    raw_img_tensor: torch.Tensor,
    similarity: np.ndarray,
    out_path: str,
    label: str,
    image_name: str,
    grid_h: int,
    grid_w: int,
    overlay_alpha: float = 0.45,
):
    """
    Save (input | heatmap | overlay) figure for one image.

    Args:
        similarity: 1D array of shape [N] with cosine similarities
    """
    # Reshape to spatial grid
    sim_grid = similarity.reshape(grid_h, grid_w)

    # Normalize to [0, 1] for visualization
    sim_min, sim_max = sim_grid.min(), sim_grid.max()
    sim_norm = (sim_grid - sim_min) / (sim_max - sim_min + 1e-8)

    # Get raw image as numpy
    raw_np = raw_img_tensor.permute(1, 2, 0).cpu().numpy()
    h, w = raw_np.shape[:2]

    # Upsample similarity map to image resolution
    from PIL import Image
    sim_resized = np.asarray(
        Image.fromarray(np.uint8(255 * sim_norm)).resize((w, h), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    # Apply colormap (jet: blue=low, red=high)
    heat_rgb = plt.get_cmap("jet")(sim_resized)[..., :3]

    # Blend overlay
    alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
    overlay = np.clip((1.0 - alpha) * raw_np + alpha * heat_rgb, 0.0, 1.0)

    # Statistics for title
    mean_sim = float(similarity.mean())
    std_sim = float(similarity.std())

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(raw_np)
    axes[0].set_title(f"Input — {label}")
    axes[0].axis("off")

    axes[1].imshow(heat_rgb)
    axes[1].set_title(f"CLS-Patch Similarity (mean={mean_sim:.3f}, std={std_sim:.3f})")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (α={alpha:.2f})")
    axes[2].axis("off")

    fig.suptitle(
        f"{image_name}  |  {grid_h}×{grid_w} patches  |  "
        f"sim range: [{sim_min:.3f}, {sim_max:.3f}]",
        fontsize=9, color="gray",
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"mean": mean_sim, "std": std_sim, "min": float(sim_min), "max": float(sim_max)}


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="CLS-Patch Cosine Similarity analysis for DINOv2"
    )
    add_common_args(ap)
    ap.add_argument("--overlay_alpha", type=float, default=0.45)
    ap.add_argument("--out_dir", default="results/cls_similarity")
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

    # ── Process ──
    records: List[Dict] = []
    all_stats = []

    for i, (x, raw, paths, labels) in enumerate(tqdm(dl, desc="CLS-Patch Similarity")):
        x = x.to(device)
        similarity = compute_cls_patch_similarity(model, x)  # [1, N]
        sim_np = similarity[0].cpu().numpy()                  # [N]

        image_stem = os.path.splitext(os.path.basename(paths[0]))[0]
        out_png = os.path.join(args.out_dir, f"cls_sim_{i:04d}_{labels[0]}_{image_stem}.png")

        stats = save_similarity_map(
            raw_img_tensor=raw[0],
            similarity=sim_np,
            out_path=out_png,
            label=labels[0],
            image_name=os.path.basename(paths[0]),
            grid_h=grid_side,
            grid_w=grid_side,
            overlay_alpha=args.overlay_alpha,
        )
        all_stats.append(stats)

        records.append({
            "index": i,
            "image_path": paths[0],
            "image_name": os.path.basename(paths[0]),
            "label": labels[0],
            "output_path": out_png,
            **stats,
        })

    # ── Summary ──
    if all_stats:
        avg_mean = np.mean([s["mean"] for s in all_stats])
        avg_std = np.mean([s["std"] for s in all_stats])
        print(f"\n[RESULT] Processed {len(records)} images")
        print(f"[RESULT] Avg cosine similarity: {avg_mean:.4f} ± {avg_std:.4f}")
        print(f"[RESULT] Interpretation:")
        if avg_std > 0.10:
            print(f"[RESULT]   High variance -> CLS is selective (good for classification)")
        elif avg_std > 0.05:
            print(f"[RESULT]   Moderate variance -> CLS has some spatial preference")
        else:
            print(f"[RESULT]   Low variance -> CLS attends broadly (may lack focus)")

    checkpoint_label = args.checkpoint or "ImageNet baseline (no checkpoint)"
    result = {
        "method": "cls_patch_cosine_similarity",
        "checkpoint": checkpoint_label,
        "arch": args.arch,
        "img_size": args.img_size,
        "split_col": args.split_col,
        "split": args.split,
        "num_images": len(records),
        "avg_similarity_mean": float(avg_mean) if all_stats else None,
        "avg_similarity_std": float(avg_std) if all_stats else None,
        "records": records,
    }

    out_json = args.out_json or os.path.join(args.out_dir, "results.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] JSON log saved: {out_json}")


if __name__ == "__main__":
    main()