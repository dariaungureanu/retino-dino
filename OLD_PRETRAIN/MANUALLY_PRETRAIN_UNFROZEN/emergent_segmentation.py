import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.decomposition import PCA
import argparse

IMAGE_PATH = r"D:\Ungureanu_Daria\OCTDL_Cleaned\DME\dme_1434389_1.jpg"
CHECKPOINT_PATH = r"D:\Ungureanu_Daria\retino-dino\checkpoints_dino_oct_optimized\dinov2_oct_opt_latest.pth"
PATCH_SIZE = 14
IMG_SIZE = 518  # Must be a multiple of 14 (37 * 14 = 518)
GRID_SIZE = IMG_SIZE // PATCH_SIZE

def main(image_path=None, output_name=None):
    if image_path is None:
        image_path = IMAGE_PATH
    if output_name is None:
        output_name = "thesis_dinov2_pca.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    print("loading base DINOv2 architecture from Meta...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

    print(f"injecting domain-adapted weights from: {os.path.basename(CHECKPOINT_PATH)}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    teacher_state_dict = checkpoint["teacher"]
    clean_state_dict = {}
    for key, value in teacher_state_dict.items():
        if key.startswith("backbone."):
            clean_key = key.replace("backbone.", "")
            clean_state_dict[clean_key] = value

    model.load_state_dict(clean_state_dict, strict=True)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img_pil = Image.open(image_path).convert("RGB")
    img_original = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model.forward_features(img_tensor)

    patch_tokens = outputs['x_norm_patchtokens'][0].cpu().numpy()

    print("running PCA to reduce 768 dimensions to RGB...")
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(patch_tokens)
    for i in range(3):
        pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / \
                             (pca_features[:, i].max() - pca_features[:, i].min())

    pca_features_grid = pca_features.reshape(GRID_SIZE, GRID_SIZE, 3)
    pca_resized = cv2.resize(pca_features_grid, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_original)
    axes[0].set_title("original OCT", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(pca_resized)
    axes[1].set_title("DINOv2 PCA Semantic Features", fontsize=14)
    axes[1].axis("off")

    overlay = cv2.addWeighted(np.array(img_original), 0.4, np.uint8(255 * pca_resized), 0.6, 0)
    axes[2].imshow(overlay)
    axes[2].set_title("pathology Overlay", fontsize=14)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"saved {output_name}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DINOv2 emergent segmentation visualization")
    parser.add_argument('--image_path', '--image', type=str, default=IMAGE_PATH, dest='image_path',
                        help=f'Path to the image file (default: {IMAGE_PATH})')
    parser.add_argument('--output', type=str, default="thesis_dinov2_pca_dme.png", dest='output_name',
                        help='Output filename for the visualization (default: thesis_dinov2_pca_dme.png)')
    args = parser.parse_args()
    main(image_path=args.image_path, output_name=args.output_name)
