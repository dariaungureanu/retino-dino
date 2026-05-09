"""
Multi-Task DINOv2 Model for OCTDL

Architecture:
    DINOv2 ViT-S/14 backbone (384-dim, 12 blocks)
    ├── Head_Disease  → 7 classes
    └── Head_Condition → 8 classes
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_backbone(
    arch: str,
    checkpoint: Optional[str],
    device: torch.device,
) -> nn.Module:
    """
    Load DINOv2 backbone from torch.hub, optionally overwriting weights
    from a continual-pretraining checkpoint.
    """
    print(f"\n{'='*60}")
    print(f"  MODEL LOADING DIAGNOSTICS")
    print(f"{'='*60}")

    # Step 1 — Load hub architecture with ImageNet weights
    print(f"[MODEL] Architecture: {arch}")
    model = torch.hub.load("facebookresearch/dinov2", arch)
    model_keys = set(model.state_dict().keys())
    print(f"[MODEL] Hub model: {len(model_keys)} parameter tensors")

    if checkpoint is None:
        print(f"[MODEL] No checkpoint → ImageNet baseline")
        print(f"{'='*60}\n")
        return model.to(device)

    # Step 2 — Load checkpoint file
    if not os.path.isfile(checkpoint):
        print(f"[FATAL] Checkpoint not found: {checkpoint}")
        sys.exit(1)

    print(f"[MODEL] Checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu")
    print(f"[MODEL] Top-level keys: {list(ckpt.keys())}")

    # Step 3 — Extract the right sub-dict
    if "model" in ckpt:
        st = ckpt["model"]
        print(f"[MODEL] Using 'model' sub-dict ({len(st)} keys)")
    elif "teacher" in ckpt:
        st = ckpt["teacher"]
        print(f"[MODEL] Using 'teacher' sub-dict ({len(st)} keys)")
    elif "state_dict" in ckpt:
        st = ckpt["state_dict"]
        print(f"[MODEL] Using 'state_dict' sub-dict ({len(st)} keys)")
    else:
        st = ckpt
        print(f"[MODEL] No recognized sub-dict, using top-level")

    # Show first keys for debugging
    print(f"[MODEL] Raw keys (first 5): {list(st.keys())[:5]}")

    # Step 4 — Try prefix patterns to find teacher backbone weights
    PREFIX_PATTERNS = [
        "teacher.backbone.",
        "backbone.",
        "module.backbone.",
        "module.",
        "",
    ]

    clean = {}
    matched_prefix = None

    for prefix in PREFIX_PATTERNS:
        candidate = {}
        for k, v in st.items():
            if prefix and k.startswith(prefix):
                candidate[k[len(prefix):]] = v
            elif prefix == "" and not any(k.startswith(p) for p in
                                          ["dino_loss", "ibot_patch_loss",
                                           "dino_head", "ibot_head",
                                           "student.", "teacher."]):
                candidate[k] = v

        if candidate:
            overlap = set(candidate.keys()) & model_keys
            if len(overlap) > len(model_keys) * 0.5:
                clean = candidate
                matched_prefix = prefix
                break

    # Fallback: force extraction
    if not clean:
        print(f"[MODEL] No prefix matched cleanly. Forcing teacher.backbone.*")
        for k, v in st.items():
            for prefix in ["teacher.backbone.", "student.backbone.", "backbone.", "module."]:
                if k.startswith(prefix):
                    clean[k[len(prefix):]] = v
                    matched_prefix = prefix
                    break

    print(f"[MODEL] Matched prefix: '{matched_prefix}'")
    print(f"[MODEL] Cleaned keys: {len(clean)} (first 3: {list(clean.keys())[:3]})")

    # Step 5 — Interpolate pos_embed if resolution mismatch
    if "pos_embed" in clean and "pos_embed" in model_keys:
        ckpt_pos  = clean["pos_embed"]
        model_pos = model.state_dict()["pos_embed"]

        if ckpt_pos.shape != model_pos.shape:
            print(f"[MODEL] pos_embed mismatch: ckpt {list(ckpt_pos.shape)} "
                  f"vs model {list(model_pos.shape)}")

            cls_pos   = ckpt_pos[:, :1, :]           # [1, 1, D]
            patch_pos = ckpt_pos[:, 1:, :]            # [1, N_ckpt, D]

            n_ckpt  = patch_pos.shape[1]
            n_model = model_pos.shape[1] - 1
            g_ckpt  = int(n_ckpt ** 0.5)
            g_model = int(n_model ** 0.5)
            d       = patch_pos.shape[-1]

            print(f"[MODEL] Interpolating: {g_ckpt}×{g_ckpt} → {g_model}×{g_model}")

            patch_pos = patch_pos.reshape(1, g_ckpt, g_ckpt, d).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(
                patch_pos.float(), size=(g_model, g_model),
                mode="bicubic", align_corners=False,
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, d)
            clean["pos_embed"] = torch.cat([cls_pos, patch_pos], dim=1)
            print(f"[MODEL] pos_embed interpolated: {list(clean['pos_embed'].shape)}")

    # Step 6 — Load weights
    result = model.load_state_dict(clean, strict=False)
    loaded = len(model_keys) - len(result.missing_keys)

    print(f"\n[MODEL] Loaded: {loaded}/{len(model_keys)} keys")
    if result.missing_keys:
        print(f"[MODEL] Missing (first 5): {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"[MODEL] Unexpected (first 5): {result.unexpected_keys[:5]}")

    if loaded == 0:
        print(f"\n[FATAL] Zero keys loaded! Weights are still ImageNet!")
        sys.exit(1)
    elif loaded < len(model_keys) * 0.9:
        print(f"\n[WARN] Partial load: {loaded}/{len(model_keys)}")
    else:
        print(f"\n[MODEL] Domain-adapted weights loaded successfully")

    print(f"{'='*60}\n")
    return model.to(device)


class OCTDLMultiTaskModel(nn.Module):
    """
    DINOv2 backbone + dual classification heads.

    Freeze strategy for ViT-S/14 (12 blocks):
        - freeze_backbone=True, unfreeze_last_n=0  → linear probing
        - freeze_backbone=True, unfreeze_last_n=2  → partial unfreeze
        - freeze_backbone=False                     → full fine-tune

    """

    # ViT-S/14 config
    EMBED_DIM = 384
    NUM_BLOCKS = 12

    def __init__(
        self,
        backbone: nn.Module,
        num_diseases: int = 7,
        num_conditions: int = 8,
        freeze_backbone: bool = True,
        unfreeze_last_n: int = 2,
        head_hidden: int = 256,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n = unfreeze_last_n

        # Apply freeze strategy
        if freeze_backbone:
            self._apply_freeze(unfreeze_last_n)

        # Classification heads
        self.head_disease = self._build_head(self.EMBED_DIM, head_hidden, num_diseases, head_dropout)
        self.head_condition = self._build_head(self.EMBED_DIM, head_hidden, num_conditions, head_dropout)

        self._print_param_summary()

    def _build_head(self, in_dim, hidden, num_classes, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def _apply_freeze(self, unfreeze_last_n: int):
        """
        Freeze entire backbone, then selectively unfreeze:
            - Last N transformer blocks
            - Final LayerNorm (norm.)
        """
        # Freeze everything first
        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_last_n == 0:
            print(f"[MODEL] Backbone fully frozen (linear probing)")
            return

        # Unfreeze last N blocks
        unfreeze_start = self.NUM_BLOCKS - unfreeze_last_n
        unfrozen_names = []

        for name, param in self.backbone.named_parameters():
            should_unfreeze = False

            # Check if parameter belongs to one of the last N blocks
            for block_idx in range(unfreeze_start, self.NUM_BLOCKS):
                if f"blocks.{block_idx}." in name:
                    should_unfreeze = True
                    break

            # Always unfreeze the final norm layer
            if name.startswith("norm."):
                should_unfreeze = True

            if should_unfreeze:
                param.requires_grad = True
                unfrozen_names.append(name)

        print(f"[MODEL] Unfrozen blocks: {unfreeze_start}–{self.NUM_BLOCKS - 1} "
              f"+ norm ({len(unfrozen_names)} params)")

    def _print_param_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"[MODEL] Parameters: {total:,} total | "
              f"{trainable:,} trainable ({100*trainable/total:.1f}%) | "
              f"{frozen:,} frozen")

    def forward(self, x):
        """
        Forward pass. Handles different DINOv2 output formats:
        - dict with 'x_norm_clstoken'
        - raw tensor
        """
        features = self.backbone(x)

        if isinstance(features, dict):
            features = features["x_norm_clstoken"]
        elif isinstance(features, tuple):
            features = features[0]

        logits_disease   = self.head_disease(features)
        logits_condition = self.head_condition(features)

        return logits_disease, logits_condition

    def get_param_groups(self, lr_backbone: float, lr_heads: float, weight_decay: float):
        """
        Differential learning rates
            - Backbone unfrozen params: low LR (e.g., 1e-5)
            - Classification heads: higher LR (e.g., 5e-4)
        """
        backbone_params = []
        backbone_nodecay = []
        head_params = []
        head_nodecay = []

        # Separate backbone vs head, decay vs no-decay
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            is_head = name.startswith("head_")
            no_decay = ("bias" in name or "norm" in name or "bn" in name)

            if is_head:
                if no_decay:
                    head_nodecay.append(param)
                else:
                    head_params.append(param)
            else:
                if no_decay:
                    backbone_nodecay.append(param)
                else:
                    backbone_params.append(param)

        groups = [
            {"params": backbone_params,   "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": backbone_nodecay,  "lr": lr_backbone, "weight_decay": 0.0},
            {"params": head_params,       "lr": lr_heads,    "weight_decay": weight_decay},
            {"params": head_nodecay,      "lr": lr_heads,    "weight_decay": 0.0},
        ]

        # Filter out empty groups
        groups = [g for g in groups if len(g["params"]) > 0]

        for g in groups:
            n = sum(p.numel() for p in g["params"])
            print(f"[OPTIM] lr={g['lr']:.1e}  wd={g['weight_decay']:.1e}  "
                  f"params={n:,}")

        return groups