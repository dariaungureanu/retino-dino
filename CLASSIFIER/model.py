import torch
import torch.nn as nn
import os
import sys
# Ensure we can import from the cloned dinov2 repository
# We assume the 'dinov2' folder is in the root of your project
try:
    from dinov2.models.vision_transformer import vit_large
except ImportError:
    print("Error: Could not import 'dinov2'. Make sure the 'dinov2' folder is present.")
    sys.exit(1)


def load_custom_backbone(checkpoint_path):
    """
    Loads the DINOv2 ViT-Large model from a local .pth checkpoint.
    Logic taken from your 'pretrain' script to handle custom keys.
    """
    print(f" Loading custom backbone from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f" Error: Checkpoint not found at {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Clean the state_dict keys (removing 'student.', 'backbone.', etc.)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    clean_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("student.", "").replace("backbone.", "").replace("_fsdp_wrapped_module.", "").replace(
            "module.", "")
        clean_state_dict[new_key] = value

    # Infer img_size from pos_embed if possible, otherwise default to logic
    # (Keeping your script's logic here for safety)
    pos_embed_shape = clean_state_dict.get('pos_embed', None)
    if pos_embed_shape is not None:
        # shape is [1, num_patches + 1, embed_dim]
        num_patches = pos_embed_shape.shape[1] - 1
        patch_size = 16  # Assuming patch 16 based on your provided script
        img_size = int((num_patches ** 0.5) * patch_size)
        print(f"   Detected img_size from checkpoint: {img_size}")
    else:
        img_size = 224
        print(f"   Could not detect size, defaulting to: {img_size}")

    # Instantiate the architecture
    # We use ViT-Large because your checkpoint is 4.8GB (Large)
    model = vit_large(
        patch_size=16,
        img_size=224,
        init_values=1.0,
        block_chunks=0
    )

    # Load weights
    msg = model.load_state_dict(clean_state_dict, strict=False)
    print(f" Backbone loaded successfully! (Msg: {msg})")

    return model


class OCTDLMultiTaskModel(nn.Module):
    def __init__(self, checkpoint_path, num_diseases=7, num_conditions=8, freeze_backbone=True):
        """
        Multi-Task Classifier for OCT images.
        Combines Custom Loading (Script) with Multi-Head Logic (Notebook).
        """
        super().__init__()

        # 1. Load Pre-trained Backbone (Custom .pth)
        self.backbone = load_custom_backbone(checkpoint_path)

        # 2. Freeze Backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("❄ Backbone is FROZEN. Only heads will be trained.")
        else:
            print(" Backbone is UN-FROZEN. Full fine-tuning enabled.")

        # 3. Define Heads
        # ViT-Large has an embedding dimension of 1024 (Base has 768)
        self.feature_dim = 1024

        # Head 1: Disease Classification (Main Task)
        # Structure inspired by your notebook: Linear -> ReLU -> Dropout -> Linear
        self.head_disease = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),  # Added BatchNorm for stability
            nn.ReLU(),
            nn.Dropout(0.3),  # Slightly higher dropout than 0.1 for better regularization
            nn.Linear(512, num_diseases)
        )

        # Head 2: Condition Classification (Secondary Task)
        self.head_condition = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_conditions)
        )

    def forward(self, x):
        # 1. Extract Features from DINOv2
        features = self.backbone(x)

        # Handle different output types of DINOv2
        if isinstance(features, dict):
            features = features["x_norm_clstoken"]
        elif isinstance(features, tuple):
            features = features[0]

        # 2. Pass through Heads
        logits_disease = self.head_disease(features)
        logits_condition = self.head_condition(features)

        # Return both logits. Loss calculation happens in train_classifier.py
        return logits_disease, logits_condition


if __name__ == "__main__":
    # Sanity Check
    print("--- Testing Model Architecture (Dummy Mode) ---")
    try:
        # Create a dummy model without loading weights (to test shapes)
        model = OCTDLMultiTaskModel("dummy_path.pth", freeze_backbone=True)
    except FileNotFoundError:
        print("   (Skipping weight load for test, just checking class definition...)")

        # Manually init a dummy backbone for shape testing
        from dinov2.models.vision_transformer import vit_large

        backbone = vit_large(patch_size=16, img_size=224)


        # Mock class for testing
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.feature_dim = 1024
                self.head_disease = nn.Linear(1024, 7)
                self.head_condition = nn.Linear(1024, 8)

            def forward(self, x):
                f = self.backbone(x)
                return self.head_disease(f), self.head_condition(f)


        model = MockModel()
        dummy_input = torch.randn(2, 3, 224, 224)
        d, c = model(dummy_input)
        print(f" Input: [2, 3, 224, 224]")
        print(f" Disease Output: {d.shape} (Expected [2, 7])")
        print(f" Condition Output: {c.shape} (Expected [2, 8])")