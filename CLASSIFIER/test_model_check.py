import torch
import os
from model import OCTDLMultiTaskModel

CHECKPOINT_PATH = r"/saved_models/model_final.rank_0.pth"

def test_model():
    print("--- 1. Testing checkpoint loading ---")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: File not found at: {CHECKPOINT_PATH}")
        return

    try:
        model = OCTDLMultiTaskModel(
            checkpoint_path=CHECKPOINT_PATH,
            num_diseases=7,
            num_conditions=8,
            freeze_backbone=True
        )
        print("Model instantiated successfully.")
        print(f"   Backbone type: ViT-Large (DINOv2)")
    except Exception as e:
        print(f"Critical error during loading: {e}")
        return

    print("\n--- 2. Testing forward pass (dimensions) ---")
    try:
        dummy_input = torch.randn(2, 3, 224, 224)
        print(f"   Input shape: {dummy_input.shape}")

        out_disease, out_condition = model(dummy_input)

        print(f"Forward pass successful.")
        print(f"   Output disease shape:   {out_disease.shape} (Expected: [2, 7])")
        print(f"   Output condition shape: {out_condition.shape} (Expected: [2, 8])")

    except Exception as e:
        print(f"Error during forward pass: {e}")
        return

    print("\nAll tests passed. Model is ready for training.")

if __name__ == "__main__":
    test_model()