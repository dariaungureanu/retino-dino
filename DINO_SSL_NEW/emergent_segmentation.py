import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.decomposition import PCA
# --- USER SETTINGS ---
IMAGE_PATH = r"D:\Ungureanu_Daria\OCTDL_Cleaned\DME\dme_1434389_1.jpg"
CHECKPOINT_PATH = r"D:\Ungureanu_Daria\retino-dino\checkpoints_dino_oct_optimized\dinov2_oct_opt_latest.pth"
PATCH_SIZE = 14
IMG_SIZE = 518  # Must be a multiple of 14 (37 * 14 = 518)
GRID_SIZE = IMG_SIZE // PATCH_SIZE

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading base DINOv2 architecture from Meta...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

    #take weights from checkpint + load into model
    print(f"Injecting domain-adapted weights from: {os.path.basename(CHECKPOINT_PATH)}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    teacher_state_dict = checkpoint["teacher"]
    clean_state_dict = {}
    for key, value in teacher_state_dict.items():
        if key.startswith("backbone."):
            clean_key = key.replace("backbone.", "")
            clean_state_dict[clean_key] = value

    #load the weights into the mode
    model.load_state_dict(clean_state_dict, strict=True)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img_pil = Image.open(IMAGE_PATH).convert("RGB")
    img_original = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model.forward_features(img_tensor)

    patch_tokens = outputs['x_norm_patchtokens'][0].cpu().numpy()

    print("Running PCA to reduce 768 dimensions to RGB...")
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(patch_tokens)
    for i in range(3):
        pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / \
                             (pca_features[:, i].max() - pca_features[:, i].min())

    pca_features_grid = pca_features.reshape(GRID_SIZE, GRID_SIZE, 3)
    pca_resized = cv2.resize(pca_features_grid, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_original)
    axes[0].set_title("Original OCT (DME)", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(pca_resized)
    axes[1].set_title("DINOv2 PCA Semantic Features", fontsize=14)
    axes[1].axis("off")

    overlay = cv2.addWeighted(np.array(img_original), 0.4, np.uint8(255 * pca_resized), 0.6, 0)
    axes[2].imshow(overlay)
    axes[2].set_title("Pathology Overlay", fontsize=14)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("thesis_dinov2_pca_dme.png", dpi=300, bbox_inches='tight')
    print("Success! Saved as 'thesis_dinov2_pca_dme.png'.")
    plt.show()


if __name__ == '__main__':
    main()