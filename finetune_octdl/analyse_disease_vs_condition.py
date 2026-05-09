"""
Disease vs Condition GradCAM comparison for OCTDL.

For the SAME image, shows where the disease head looks vs where the condition
head looks - evidence that the model learned different spatial features for
each task. Output: side-by-side [Original] [Disease CAM] [Condition CAM].

Usage:
    python finetune_octdl/analyse_disease_vs_condition.py \
        --data_path /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --checkpoint_dir saved_models/octdl_domain_adapted_unfreeze2 \
        --out_dir results/disease_vs_condition
"""

import argparse
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from dataset import (
    IGNORE_INDEX, OCTDLMultiTaskDataset, get_data_splits, get_eval_transform,
)
from model import OCTDLMultiTaskModel, load_backbone


class TaskHeadWrapper(nn.Module):
    """Output only one head's logits for GradCAM."""
    def __init__(self, model, head_index):
        super().__init__()
        self.model = model
        self.head_index = head_index

    def forward(self, x):
        return self.model(x)[self.head_index]


def reshape_transform_vit(tensor):
    result = tensor[:, 1:, :]
    grid = int(np.sqrt(result.size(1)))
    result = result.reshape(tensor.size(0), grid, grid, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def denormalize(img_tensor):
    img = img_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1)


DISEASE_CONDITION_MAP = {
    "AMD": ["drusen", "MNV", "MNV_suspected"],
    "DME": ["ME"],
    "ERM": ["ERM"],
    "NO": ["NO"],
    "RVO": ["ME", "DRIL"],
    "VID": ["MH"],
    "RAO": ["DRIL"],
}


def main():
    parser = argparse.ArgumentParser(description="Disease vs Condition GradCAM")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="results/disease_vs_condition")
    parser.add_argument("--topk", type=int, default=5,
                        help="Images per disease class")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if args.model_path:
        model_path = args.model_path
    elif args.checkpoint_dir:
        model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    else:
        raise ValueError("Provide --model_path or --checkpoint_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(model_path, map_location=device)
    config = ckpt["config"]
    disease_map = ckpt["disease_map"]
    condition_map = ckpt["condition_map"]
    inv_disease = {v: k for k, v in disease_map.items()}
    inv_condition = {v: k for k, v in condition_map.items()}

    backbone = load_backbone(config["arch"], config["checkpoint"], device)
    model = OCTDLMultiTaskModel(
        backbone=backbone,
        num_diseases=len(disease_map),
        num_conditions=len(condition_map),
        freeze_backbone=(config["unfreeze_last_n"] < 12),
        unfreeze_last_n=config["unfreeze_last_n"],
        head_hidden=config["head_hidden"],
        head_dropout=config["head_dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for p in model.backbone.parameters():
        p.requires_grad_(True)

    csv_path = os.path.join(args.data_path, "OCTDL_clean_metadata.csv")
    _, _, test_df, _, _ = get_data_splits(csv_path)
    eval_transform = get_eval_transform(config["img_size"])
    test_ds = OCTDLMultiTaskDataset(
        test_df, args.data_path, eval_transform, disease_map, condition_map,
    )

    softmax = nn.Softmax(dim=1)
    all_preds_d, all_preds_c, all_labels_d, all_labels_c = [], [], [], []
    all_conf_d = []

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    with torch.no_grad():
        for images, labels_d, labels_c in tqdm(test_loader, desc="Collecting predictions"):
            images = images.to(device)
            logits_d, logits_c = model(images)
            probs_d = softmax(logits_d)
            conf_d, pred_d = torch.max(probs_d, dim=1)
            pred_c = torch.argmax(logits_c, dim=1)

            all_preds_d.extend(pred_d.cpu().numpy())
            all_preds_c.extend(pred_c.cpu().numpy())
            all_labels_d.extend(labels_d.numpy())
            all_labels_c.extend(labels_c.numpy())
            all_conf_d.extend(conf_d.cpu().numpy())

    all_preds_d = np.array(all_preds_d)
    all_preds_c = np.array(all_preds_c)
    all_labels_d = np.array(all_labels_d)
    all_labels_c = np.array(all_labels_c)
    all_conf_d = np.array(all_conf_d)

    target_layers = [model.backbone.blocks[-1].norm1]

    wrapper_disease = TaskHeadWrapper(model, 0)
    wrapper_condition = TaskHeadWrapper(model, 1)

    cam_disease = GradCAM(model=wrapper_disease, target_layers=target_layers,
                          reshape_transform=reshape_transform_vit)
    cam_condition = GradCAM(model=wrapper_condition, target_layers=target_layers,
                            reshape_transform=reshape_transform_vit)

    for disease_idx in range(len(disease_map)):
        disease_name = inv_disease[disease_idx]

        mask = (all_labels_d == disease_idx) & (all_preds_d == disease_idx)
        indices = np.where(mask)[0]
        confs = all_conf_d[indices]

        if len(indices) == 0:
            print(f"no correct predictions for {disease_name}")
            continue

        order = np.argsort(-confs)
        selected = indices[order[:args.topk]]

        n = len(selected)
        fig, axes = plt.subplots(n, 3, figsize=(14, 4 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        expected_conds = DISEASE_CONDITION_MAP.get(disease_name, [])

        for i, idx in enumerate(selected):
            img_tensor, label_d, label_c = test_ds[idx]
            img_input = img_tensor.unsqueeze(0).to(device)
            rgb_img = denormalize(img_input)

            with torch.no_grad():
                logits_d, logits_c = model(img_input)
                prob_d = softmax(logits_d)[0]
                prob_c = softmax(logits_c)[0]
                pred_d = torch.argmax(prob_d).item()
                pred_c = torch.argmax(prob_c).item()
                conf_d_val = prob_d[pred_d].item()
                conf_c_val = prob_c[pred_c].item()

            pred_disease_name = inv_disease[pred_d]
            pred_cond_name = inv_condition[pred_c]
            true_cond_name = inv_condition.get(int(label_c), "IGNORE")

            cam_d = cam_disease(
                input_tensor=img_input,
                targets=[ClassifierOutputTarget(pred_d)]
            )[0]

            cam_c = cam_condition(
                input_tensor=img_input,
                targets=[ClassifierOutputTarget(pred_c)]
            )[0]

            vis_d = show_cam_on_image(rgb_img, cam_d, use_rgb=True)
            vis_c = show_cam_on_image(rgb_img, cam_c, use_rgb=True)

            axes[i, 0].imshow(rgb_img)
            axes[i, 0].set_title(
                f"disease: {pred_disease_name} ({conf_d_val:.0%})\n"
                f"Condition: {pred_cond_name} ({conf_c_val:.0%})\n"
                f"True cond: {true_cond_name}",
                fontsize=9,
            )
            axes[i, 0].axis("off")

            axes[i, 1].imshow(vis_d)
            axes[i, 1].set_title(
                f"disease Head -> {pred_disease_name}",
                fontsize=10, fontweight="bold", color="#1565C0",
            )
            axes[i, 1].axis("off")

            axes[i, 2].imshow(vis_c)
            axes[i, 2].set_title(
                f"condition Head -> {pred_cond_name}",
                fontsize=10, fontweight="bold", color="#C62828",
            )
            axes[i, 2].axis("off")

        expected_str = ", ".join(expected_conds) if expected_conds else "?"
        fig.suptitle(
            f"disease vs Condition Attention - {disease_name}\n"
            f"(Expected conditions: {expected_str})",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        save_path = os.path.join(args.out_dir, f"disease_vs_condition_{disease_name}.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"{save_path}")

    print(f"\nall outputs: {args.out_dir}")


if __name__ == "__main__":
    main()