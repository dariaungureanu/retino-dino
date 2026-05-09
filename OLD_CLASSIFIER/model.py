import torch
import torch.nn as nn
import os


def load_custom_backbone(checkpoint_path):
    print("Loading backbone architecture from Torch Hub...")

    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

    print(f"Loading custom weights from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Error: Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    clean_state_dict = {}

    for key, value in state_dict.items():
        new_key = key.replace("student.", "").replace("backbone.", "").replace("_fsdp_wrapped_module.", "").replace(
            "module.", "")
        clean_state_dict[new_key] = value

    msg = backbone.load_state_dict(clean_state_dict, strict=False)
    print(f"Backbone weights loaded successfully (Missing keys for head layers: {len(msg.missing_keys)})")

    return backbone


class OCTDLMultiTaskModel(nn.Module):
    def __init__(self, checkpoint_path, num_diseases=7, num_conditions=8, freeze_backbone=True, unfreeze_last_block=False):
        super().__init__()

        self.backbone = load_custom_backbone(checkpoint_path)

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if unfreeze_last_block and ("blocks.23" in name or "norm" in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            if unfreeze_last_block:
                print("Backbone is mostly frozen, BUT the last block (blocks.23) is UNFROZEN.")
            else:
                print("Backbone is fully frozen. Only classification heads will be trained.")
        else:
            print("Backbone is unfrozen. Full fine-tuning enabled.")

        self.feature_dim = 1024

        self.head_disease = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_diseases)
        )

        self.head_condition = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_conditions)
        )

    def forward(self, x):
        features = self.backbone(x)

        if isinstance(features, dict):
            features = features["x_norm_clstoken"]
        elif isinstance(features, tuple):
            features = features[0]

        logits_disease = self.head_disease(features)
        logits_condition = self.head_condition(features)

        return logits_disease, logits_condition