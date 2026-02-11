"""
Quick test to verify all dependencies are installed
Run this before training to check everything is ready
"""

print("🔍 Testing dependencies...\n")

try:
    import torch
    print("PyTorch installed:", torch.__version__)
except ImportError:
    print("PyTorch NOT installed - run: pip install torch torchvision")

try:
    import numpy as np
    print("NumPy installed:", np.__version__)
except ImportError:
    print("NumPy NOT installed - run: pip install numpy")

try:
    from sklearn.metrics import accuracy_score
    import sklearn
    print("scikit-learn installed:", sklearn.__version__)
except ImportError:
    print("scikit-learn NOT installed - run: pip install scikit-learn")

try:
    import wandb
    print("WandB installed:", wandb.__version__)
except ImportError:
    print("WandB NOT installed - run: pip install wandb")

try:
    from torchvision import transforms
    import torchvision
    print("TorchVision installed:", torchvision.__version__)
except ImportError:
    print("TorchVision NOT installed - run: pip install torchvision")

try:
    from PIL import Image
    print("Pillow installed")
except ImportError:
    print("Pillow NOT installed - run: pip install Pillow")

print("\nAll dependencies checked.")
