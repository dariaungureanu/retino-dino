import torch
import torch.nn as nn
import os


def load_custom_backbone(checkpoint_path):
    """
    Încarcă arhitectura DINOv2 direct din 'hub' (internet/cache),
    fără să aibă nevoie de folderul local 'dinov2'.
    """
    print(f"🔄 Loading backbone architecture from Torch Hub...")

    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

    print(f"📂 Loading custom weights from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Error: Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    clean_state_dict = {}

    for key, value in state_dict.items():
        new_key = key.replace("student.", "").replace("backbone.", "").replace("_fsdp_wrapped_module.", "").replace(
            "module.", "")
        clean_state_dict[new_key] = value

    msg = backbone.load_state_dict(clean_state_dict, strict=False)
    print(f"✅ Backbone weights loaded! (Missing keys expected for head: {len(msg.missing_keys)})")

    return backbone


class OCTDLMultiTaskModel(nn.Module):
    def __init__(self, checkpoint_path, num_diseases=7, num_conditions=8, freeze_backbone=True):
        super().__init__()

        self.backbone = load_custom_backbone(checkpoint_path)

        # 2. Freeze Backbone (Protecție)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("❄️ Backbone is FROZEN. Only heads will be trained.")
        else:
            print("🔥 Backbone is UN-FROZEN. Full fine-tuning enabled.")

        # 3. Define Heads
        self.feature_dim = 1024  # ViT-Large are 1024

        # Head 1: Disease
        self.head_disease = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_diseases)
        )

        # Head 2: Condition
        self.head_condition = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_conditions)
        )

    def forward(self, x):
        # 1. Extract Features
        features = self.backbone(x)

        # Gestionare output DINOv2 (uneori e dict, alteori tuple)
        if isinstance(features, dict):
            features = features["x_norm_clstoken"]
        elif isinstance(features, tuple):
            features = features[0]

        # 2. Pass through Heads
        logits_disease = self.head_disease(features)
        logits_condition = self.head_condition(features)

        return logits_disease, logits_condition