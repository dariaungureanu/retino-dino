import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

# --- USER SETTINGS ---
IMAGE_PATH = r"D:\Ungureanu_Daria\OCTDL_Cleaned\DME\dme_1434389_1.jpg"
CHECKPOINT_PATH = r"D:\Ungureanu_Daria\retino-dino\checkpoints_dino_oct_optimized\dinov2_oct_opt_BEST.pth"


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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img_pil = Image.open(IMAGE_PATH).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    attention_maps = []
    def hook_fn(module, input, output):
        attention_maps.append(input[0])
    hook = model.blocks[-1].attn.attn_drop.register_forward_hook(hook_fn)

    #force memory_efficient off -> i dont use xformers
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        with torch.no_grad():
            model(img_tensor)

    hook.remove()

    attn = attention_maps[0][0] # Shape: (12_heads, 257_tokens, 257_tokens)
    cls_attn = attn[:, 0, 1:] # Shape: (12_heads, 256_patches)

    grid_size = int(np.sqrt(cls_attn.shape[1]))  # Should be 16 (224 / 14)
    img_original = img_pil.resize((224, 224))
    img_orig_np = np.array(img_original)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Plot Original
    axes[0].imshow(img_original)
    axes[0].set_title("Original OCT")
    axes[0].axis("off")

    # Plot Mean Attention
    mean_attn = cls_attn.mean(dim=0).reshape(grid_size, grid_size).cpu().numpy()
    mean_attn = (mean_attn - mean_attn.min()) / (mean_attn.max() - mean_attn.min())
    mean_attn_resized = cv2.resize(mean_attn, (224, 224), interpolation=cv2.INTER_CUBIC)

    heatmap_mean = plt.get_cmap('jet')(mean_attn_resized)[:, :, :3]
    overlay_mean = cv2.addWeighted(img_orig_np, 0.5, np.uint8(255 * heatmap_mean), 0.5, 0)

    axes[1].imshow(overlay_mean)
    axes[1].set_title("Mean Attention")
    axes[1].axis("off")

    heads_to_plot = [0, 4, 8]
    for idx, head_idx in enumerate(heads_to_plot):
        head_attn = cls_attn[head_idx].reshape(grid_size, grid_size).cpu().numpy()
        head_attn = (head_attn - head_attn.min()) / (head_attn.max() - head_attn.min())
        head_attn_resized = cv2.resize(head_attn, (224, 224), interpolation=cv2.INTER_CUBIC)

        heatmap_head = plt.get_cmap('jet')(head_attn_resized)[:, :, :3]
        overlay_head = cv2.addWeighted(img_orig_np, 0.5, np.uint8(255 * heatmap_head), 0.5, 0)

        axes[idx + 2].imshow(overlay_head)
        axes[idx + 2].set_title(f"Attention Head {head_idx}")
        axes[idx + 2].axis("off")

    plt.tight_layout()
    plt.savefig("attention_map_result.png", dpi=300, bbox_inches='tight')
    print("Success! The image was saved as 'attention_map_result.png'.")
    plt.show()


if __name__ == '__main__':
    main()