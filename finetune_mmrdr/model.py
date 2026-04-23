"""
Single-Task DINOv2 Model for MMRDR-OCT (DME grading).

Same backbone loading as OCTDL pipeline.
Single classification head: 384 → 256 → 3 classes.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_backbone(arch, checkpoint, device):
    """
    Load DINOv2 backbone — same logic as OCTDL pipeline.
    Handles FSDP checkpoints, prefix matching, pos_embed interpolation.
    """
    print(f"\n{'='*60}")
    print(f"  MODEL LOADING")
    print(f"{'='*60}")

    model = torch.hub.load("facebookresearch/dinov2", arch)
    model_keys = set(model.state_dict().keys())
    print(f"[MODEL] {arch}: {len(model_keys)} params")

    if checkpoint is None:
        print(f"[MODEL] No checkpoint → ImageNet baseline")
        return model.to(device)

    if not os.path.isfile(checkpoint):
        print(f"[FATAL] Not found: {checkpoint}")
        sys.exit(1)

    ckpt = torch.load(checkpoint, map_location="cpu")
    print(f"[MODEL] Checkpoint: {checkpoint}")

    # Extract sub-dict
    if "model" in ckpt:
        st = ckpt["model"]
    elif "teacher" in ckpt:
        st = ckpt["teacher"]
    elif "state_dict" in ckpt:
        st = ckpt["state_dict"]
    else:
        st = ckpt

    # Try prefix patterns
    PREFIX_PATTERNS = ["teacher.backbone.", "backbone.", "module.backbone.", "module.", ""]
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

    if not clean:
        for k, v in st.items():
            for prefix in ["teacher.backbone.", "student.backbone.", "backbone."]:
                if k.startswith(prefix):
                    clean[k[len(prefix):]] = v
                    matched_prefix = prefix
                    break

    print(f"[MODEL] Prefix: '{matched_prefix}', {len(clean)} keys")

    # pos_embed interpolation
    if "pos_embed" in clean and "pos_embed" in model_keys:
        ckpt_pos = clean["pos_embed"]
        model_pos = model.state_dict()["pos_embed"]
        if ckpt_pos.shape != model_pos.shape:
            cls_pos = ckpt_pos[:, :1, :]
            patch_pos = ckpt_pos[:, 1:, :]
            n_ckpt = patch_pos.shape[1]
            n_model = model_pos.shape[1] - 1
            g_ckpt = int(n_ckpt ** 0.5)
            g_model = int(n_model ** 0.5)
            d = patch_pos.shape[-1]
            patch_pos = patch_pos.reshape(1, g_ckpt, g_ckpt, d).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos.float(), size=(g_model, g_model),
                                      mode="bicubic", align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, d)
            clean["pos_embed"] = torch.cat([cls_pos, patch_pos], dim=1)
            print(f"[MODEL] pos_embed interpolated")

    result = model.load_state_dict(clean, strict=False)
    loaded = len(model_keys) - len(result.missing_keys)

    if loaded == 0:
        print(f"[FATAL] Zero keys loaded!")
        sys.exit(1)

    print(f"[MODEL] Loaded {loaded}/{len(model_keys)} keys ✓")
    print(f"{'='*60}\n")
    return model.to(device)


class MMRDRModel(nn.Module):
    """
    Single-task classifier: DINOv2 backbone → single MLP head → 3 classes.
    Same architecture as OCTDL heads, just one instead of two.
    """
    EMBED_DIM = 384  # ViT-S/14
    NUM_BLOCKS = 12

    def __init__(self, backbone, num_classes=3, freeze_backbone=True,
                 unfreeze_last_n=2, head_hidden=256, head_dropout=0.3):
        super().__init__()
        self.backbone = backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            if unfreeze_last_n > 0:
                unfreeze_start = self.NUM_BLOCKS - unfreeze_last_n
                for name, param in self.backbone.named_parameters():
                    for block_idx in range(unfreeze_start, self.NUM_BLOCKS):
                        if f"blocks.{block_idx}." in name:
                            param.requires_grad = True
                    if name.startswith("norm."):
                        param.requires_grad = True

        self.head = nn.Sequential(
            nn.Linear(self.EMBED_DIM, head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_classes),
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MODEL] {total:,} total | {trainable:,} trainable ({100*trainable/total:.1f}%)")

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, dict):
            features = features["x_norm_clstoken"]
        elif isinstance(features, tuple):
            features = features[0]
        return self.head(features)

    def get_param_groups(self, lr_backbone, lr_heads, weight_decay):
        backbone_params, backbone_nodecay = [], []
        head_params, head_nodecay = [], []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_head = name.startswith("head.")
            no_decay = ("bias" in name or "norm" in name or "bn" in name)
            if is_head:
                (head_nodecay if no_decay else head_params).append(param)
            else:
                (backbone_nodecay if no_decay else backbone_params).append(param)

        groups = [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": backbone_nodecay, "lr": lr_backbone, "weight_decay": 0.0},
            {"params": head_params, "lr": lr_heads, "weight_decay": weight_decay},
            {"params": head_nodecay, "lr": lr_heads, "weight_decay": 0.0},
        ]
        return [g for g in groups if len(g["params"]) > 0]