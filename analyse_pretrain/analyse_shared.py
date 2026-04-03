"""
Shared utilities for DINOv2 pretraining analysis methods.
==========================================================
Contains: model loading (with checkpoint diagnostics), dataset,
sample building, and common constants.

Import in each method script:
    from analyse_shared import load_model, build_samples, OCTDataset, Sample
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# DINOv2 ViT-S/14 canonical eval resolution: 518 = 37 * 14
DEFAULT_IMG_SIZE = 518
DEFAULT_ARCH = "dinov2_vits14"


# ──────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────

@dataclass
class Sample:
    image_path: str
    label: str


def build_samples(
    csv_path: str,
    image_root: str,
    split_col: Optional[str],
    split_name: Optional[str],
    path_col: str,
    label_col: str,
) -> List[Sample]:
    """
    Build list of (image_path, label) from a CSV file.
    Handles OCTDL_Cleaned layout where CSV has bare filenames
    and images are stored in image_root/<disease>/filename.jpg.
    """
    df = pd.read_csv(csv_path)
    print(f"[DATA] CSV loaded: {len(df)} rows from {csv_path}")

    if split_col and split_name:
        df = df[df[split_col] == split_name].copy()
        print(f"[DATA] Filtered {split_col}=={split_name}: {len(df)} rows remain")

    samples = []
    missing = 0
    for _, row in df.iterrows():
        rel = str(row[path_col]).strip()

        # OCTDL_Cleaned layout: bare filename → prepend disease folder
        if "/" not in rel and "\\" not in rel and "disease" in df.columns:
            rel = os.path.join(str(row["disease"]), rel)

        full_path = os.path.join(image_root, rel)
        if not os.path.isfile(full_path):
            missing += 1
            continue

        samples.append(Sample(
            image_path=full_path,
            label=str(row[label_col]),
        ))

    if missing > 0:
        print(f"[DATA] WARNING: {missing} images not found on disk, skipped")
    print(f"[DATA] Built {len(samples)} samples")
    return samples


class OCTDataset(Dataset):
    """
    Returns (normalized_tensor, raw_tensor, path, label) per image.

    normalized_tensor: ImageNet-normalized, for model input
    raw_tensor:        [0,1] range, for visualization
    """

    def __init__(self, samples: List[Sample], img_size: int = DEFAULT_IMG_SIZE):
        self.samples = samples
        self.img_size = img_size
        self.normalize = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.raw = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        return self.normalize(img), self.raw(img), s.image_path, s.label


# ──────────────────────────────────────────────────────────────
# Model loading with full diagnostics
# ──────────────────────────────────────────────────────────────

def load_model(
    arch: str,
    checkpoint: Optional[str],
    device: torch.device,
) -> torch.nn.Module:
    """
    Load DINOv2 backbone from torch.hub, optionally overwriting weights
    from a continual-pretraining checkpoint.

    Args:
        arch:       Hub model name (e.g., "dinov2_vits14")
        checkpoint: Path to .pth checkpoint, or None for ImageNet baseline
        device:     Target device

    Returns:
        Model in eval mode on the specified device.
    """
    print(f"\n{'=' * 60}")
    print(f"  MODEL LOADING DIAGNOSTICS")
    print(f"{'=' * 60}")

    # Step 1 — Load hub model (comes with ImageNet pretrained weights)
    print(f"[MODEL] Architecture: {arch}")
    model = torch.hub.load("facebookresearch/dinov2", arch)
    model_keys = set(model.state_dict().keys())
    print(f"[MODEL] Hub model has {len(model_keys)} parameter tensors")

    # Step 2 — No checkpoint → return ImageNet baseline
    if checkpoint is None:
        print(f"[MODEL] No checkpoint → using ORIGINAL ImageNet weights (baseline)")
        print(f"{'=' * 60}\n")
        model.eval().to(device)
        return model

    # Step 3 — Load checkpoint
    if not os.path.isfile(checkpoint):
        print(f"[FATAL] Checkpoint file not found: {checkpoint}")
        sys.exit(1)

    print(f"[MODEL] Checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu")

    if not isinstance(ckpt, dict):
        print(f"[FATAL] Expected dict, got {type(ckpt)}")
        sys.exit(1)

    print(f"[MODEL] Top-level keys: {list(ckpt.keys())}")

    # Step 4 — Extract the right sub-dict
    # Official DINOv2 repo saves checkpoints in different formats:
    #
    # Format A (FSDP / single-GPU official):
    #   ckpt = {"model": {"student.backbone.cls_token": ...,
    #                      "teacher.backbone.cls_token": ...,
    #                      "dino_loss.center": ..., ...},
    #           "optimizer": ..., "iteration": ...}
    #   → All components flat under "model" with full dotted prefixes
    #
    # Format B (some custom saves):
    #   ckpt = {"teacher": {"backbone.cls_token": ..., "dino_head.mlp.0.weight": ...},
    #           "student": {...}}
    #   → Teacher/student as separate top-level dicts
    #
    # We always want the TEACHER backbone (EMA-smoothed, higher quality).

    if "model" in ckpt:
        st = ckpt["model"]
        print(f"[MODEL] Extracted 'model' sub-dict ({len(st)} keys)")
    elif "teacher" in ckpt:
        st = ckpt["teacher"]
        print(f"[MODEL] Extracted 'teacher' sub-dict ({len(st)} keys)")
    elif "state_dict" in ckpt:
        st = ckpt["state_dict"]
        print(f"[MODEL] Extracted 'state_dict' sub-dict ({len(st)} keys)")
    else:
        st = ckpt
        print(f"[MODEL] No recognized sub-dict, using top-level")

    # Show raw keys for debugging
    raw_keys = list(st.keys())[:10]
    print(f"[MODEL] Raw keys (first 10):")
    for k in raw_keys:
        print(f"[MODEL]   {k}")

    # Step 5 — Extract teacher backbone keys and strip prefix
    # We try multiple known prefix patterns in priority order.
    # The first pattern that yields matches wins.
    PREFIX_PATTERNS = [
        "teacher.backbone.",   # Format A: flat model dict (official FSDP save)
        "backbone.",           # Format B: already inside teacher sub-dict
        "module.backbone.",    # Format B + DDP wrapping
        "module.",             # Bare DDP wrapping (no backbone nesting)
        "",                    # Keys already clean (e.g., manually saved backbone)
    ]

    clean = {}
    matched_prefix = None

    for prefix in PREFIX_PATTERNS:
        candidate = {}
        for k, v in st.items():
            if k.startswith(prefix) and prefix != "":
                stripped = k[len(prefix):]
                candidate[stripped] = v
            elif prefix == "" and not any(k.startswith(p) for p in ["dino_loss", "ibot_patch_loss"]):
                candidate[k] = v

        # Check if this prefix gives us keys that match the model
        if candidate:
            overlap = set(candidate.keys()) & model_keys
            if len(overlap) > len(model_keys) * 0.5:
                clean = candidate
                matched_prefix = prefix
                break

    # If no pattern matched well, fall back to teacher.backbone.* with forced extraction
    if not clean:
        print(f"[MODEL] No prefix pattern matched cleanly. Trying teacher.backbone.* forcefully...")
        for k, v in st.items():
            for prefix in ["teacher.backbone.", "student.backbone.", "backbone.", "module."]:
                if k.startswith(prefix):
                    clean[k[len(prefix):]] = v
                    matched_prefix = prefix
                    break

    print(f"[MODEL] Matched prefix: '{matched_prefix}'")
    print(f"[MODEL] Cleaned keys ({len(clean)} total, first 5): {list(clean.keys())[:5]}")

    # Step 5b — Interpolate pos_embed if resolution mismatch
    # Checkpoint was trained at 224px → pos_embed shape [1, 257, 384] (256 patches + CLS)
    # Hub model may be initialized at 518px → pos_embed shape [1, 1370, 384] (1369 patches + CLS)
    # We interpolate the checkpoint's pos_embed to match the model's expected resolution.
    if "pos_embed" in clean and "pos_embed" in model_keys:
        ckpt_pos = clean["pos_embed"]              # [1, N_ckpt, D]
        model_pos = model.state_dict()["pos_embed"] # [1, N_model, D]

        if ckpt_pos.shape != model_pos.shape:
            print(f"[MODEL] pos_embed shape mismatch: checkpoint {list(ckpt_pos.shape)} "
                  f"vs model {list(model_pos.shape)}")

            # Separate CLS token (position 0) from patch positions
            cls_token_pos = ckpt_pos[:, :1, :]     # [1, 1, D]
            patch_pos = ckpt_pos[:, 1:, :]          # [1, N_patches_ckpt, D]

            n_patches_ckpt = patch_pos.shape[1]
            n_patches_model = model_pos.shape[1] - 1  # subtract CLS

            grid_ckpt = int(n_patches_ckpt ** 0.5)    # e.g., 16 for 224px
            grid_model = int(n_patches_model ** 0.5)   # e.g., 37 for 518px

            print(f"[MODEL] Interpolating pos_embed: {grid_ckpt}×{grid_ckpt} → "
                  f"{grid_model}×{grid_model}")

            # Reshape to 2D grid, interpolate, reshape back
            d = patch_pos.shape[-1]
            patch_pos = patch_pos.reshape(1, grid_ckpt, grid_ckpt, d).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(
                patch_pos.float(),
                size=(grid_model, grid_model),
                mode="bicubic",
                align_corners=False,
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, d)

            # Recombine CLS + interpolated patch positions
            clean["pos_embed"] = torch.cat([cls_token_pos, patch_pos], dim=1)
            print(f"[MODEL] pos_embed interpolated: {list(clean['pos_embed'].shape)}")
        else:
            print(f"[MODEL] pos_embed shape matches: {list(ckpt_pos.shape)}")

    # Step 6 — Load weights and report
    result = model.load_state_dict(clean, strict=False)

    loaded = len(model_keys) - len(result.missing_keys)
    total = len(model_keys)

    print(f"\n[MODEL] ── Result ──")
    print(f"[MODEL] Loaded: {loaded}/{total} keys")

    if result.missing_keys:
        print(f"[MODEL] Missing (first 5): {result.missing_keys[:5]}")
    if result.unexpected_keys:
        # dino_head.*, ibot_head.* are expected — SSL heads not in backbone
        print(f"[MODEL] Unexpected (first 5): {result.unexpected_keys[:5]}")

    # Step 7 — Abort if nothing loaded
    if loaded == 0:
        print(f"\n[FATAL] Zero keys loaded! Model has ImageNet weights, not yours!")
        print(f"[FATAL] Likely cause: key prefix mismatch")
        sys.exit(1)
    elif loaded < total * 0.9:
        print(f"\n[WARN] Only {loaded}/{total} keys — partial load")
    else:
        print(f"\n[MODEL] ✓ Domain-adapted weights loaded successfully")

    print(f"{'=' * 60}\n")
    model.eval().to(device)
    return model


# ──────────────────────────────────────────────────────────────
# Common argument helpers
# ──────────────────────────────────────────────────────────────

def add_common_args(parser):
    """Add arguments shared across all analysis methods."""
    g = parser.add_argument_group("Model")
    g.add_argument("--arch", default=DEFAULT_ARCH,
                   help="DINOv2 hub architecture (must match checkpoint)")
    g.add_argument("--checkpoint", default=None,
                   help="Domain-adapted checkpoint. Omit for ImageNet baseline.")

    g = parser.add_argument_group("Data")
    g.add_argument("--csv", required=True, help="Path to metadata CSV")
    g.add_argument("--image_root", required=True, help="Root image directory")
    g.add_argument("--split_col", default=None, help="CSV column to filter on")
    g.add_argument("--split", default=None, help="Value to filter for")
    g.add_argument("--path_col", default="file_name", help="CSV column with filenames")
    g.add_argument("--label_col", default="label_disease", help="CSV column with labels")

    g = parser.add_argument_group("Processing")
    g.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE,
                   help="Input resolution (518 recommended, must be divisible by 14)")
    g.add_argument("--max_images", type=int, default=10,
                   help="Max images to process (0=all)")

    return parser


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    return device


def validate_img_size(img_size: int, patch_size: int = 14):
    if img_size % patch_size != 0:
        print(f"[WARN] img_size={img_size} not divisible by {patch_size}. "
              f"Positional embeddings will be interpolated.")
    grid = img_size // patch_size
    print(f"[INFO] Resolution {img_size}×{img_size} → {grid}×{grid} = {grid**2} patches")
    return grid