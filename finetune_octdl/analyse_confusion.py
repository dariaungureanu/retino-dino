"""
Generate confusion matrices from a saved fine-tuning checkpoint.

Usage:
    # Run C (best model)
    python finetune_octdl/analyse_confusion.py \
        --data_path /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --checkpoint_dir saved_models/run_C_unfreeze2 \
        --out_dir results/confusion_matrices/run_C

    # Any other run
    python finetune_octdl/analyse_confusion.py \
        --data_path /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --model_path saved_models/run_A_frozen/best_model.pth \
        --out_dir results/confusion_matrices/run_A
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    IGNORE_INDEX, OCTDLMultiTaskDataset, get_data_splits, get_eval_transform,
)
from model import OCTDLMultiTaskModel, load_backbone


def load_model_from_checkpoint(model_path, device):
    print(f"[INFO] Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device)

    config = ckpt["config"]
    disease_map = ckpt["disease_map"]
    condition_map = ckpt["condition_map"]

    print(f"[INFO] arch={config['arch']}  unfreeze={config['unfreeze_last_n']}  "
          f"epoch={ckpt['epoch']}  val_f1={ckpt['val_disease_f1']:.4f}")

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
    return model, ckpt


@torch.no_grad()
def get_predictions(model, loader, device):
    """Run inference and return (preds_disease, labels_disease, preds_condition, labels_condition)."""
    all_preds_d, all_labels_d = [], []
    all_preds_c, all_labels_c = [], []

    for images, labels_d, labels_c in tqdm(loader, desc="Running inference"):
        images = images.to(device, non_blocking=True)
        logits_d, logits_c = model(images)

        all_preds_d.append(torch.argmax(logits_d, dim=1).cpu().numpy())
        all_labels_d.append(labels_d.numpy())
        all_preds_c.append(torch.argmax(logits_c, dim=1).cpu().numpy())
        all_labels_c.append(labels_c.numpy())

    return (
        np.concatenate(all_preds_d),
        np.concatenate(all_labels_d),
        np.concatenate(all_preds_c),
        np.concatenate(all_labels_c),
    )


def plot_confusion_matrix(
    y_true, y_pred, class_names, title, save_path,
    figsize=None, normalize=False,
):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm / row_sums * 100, 0)
        fmt = ".1f"
        cbar_label = "Recall (%)"
    else:
        cm_plot = cm
        fmt = "d"
        cbar_label = "Count"

    n_classes = len(class_names)
    if figsize is None:
        figsize = (max(8, n_classes * 1.2), max(6, n_classes * 1.0))

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": cbar_label},
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Rotate tick labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate confusion matrices")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to best_model.pth")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory containing best_model.pth")
    parser.add_argument("--out_dir", type=str, default="results/confusion_matrices")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # Find checkpoint
    if args.model_path:
        model_path = args.model_path
    elif args.checkpoint_dir:
        model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    else:
        raise ValueError("Provide --model_path or --checkpoint_dir")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    model, ckpt = load_model_from_checkpoint(model_path, device)
    config = ckpt["config"]
    disease_map = ckpt["disease_map"]
    condition_map = ckpt["condition_map"]

    # Build test set (same split as training)
    csv_path = os.path.join(args.data_path, "OCTDL_clean_metadata.csv")
    _, _, test_df, _, _ = get_data_splits(csv_path)

    eval_transform = get_eval_transform(config["img_size"])
    test_ds = OCTDLMultiTaskDataset(
        test_df, args.data_path, eval_transform, disease_map, condition_map,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"[DATA] Test set: {len(test_ds)} images")

    # Get predictions
    preds_d, labels_d, preds_c, labels_c = get_predictions(model, test_loader, device)

    # Disease confusion matrices
    inv_disease = {v: k for k, v in disease_map.items()}
    disease_names = [inv_disease[i] for i in range(len(disease_map))]

    # Raw counts
    plot_confusion_matrix(
        labels_d, preds_d, disease_names,
        title="Disease Classification - Confusion Matrix (Counts)",
        save_path=os.path.join(args.out_dir, "disease_confusion_counts.png"),
    )

    # Normalized (recall %)
    plot_confusion_matrix(
        labels_d, preds_d, disease_names,
        title="Disease Classification - Confusion Matrix (Recall %)",
        save_path=os.path.join(args.out_dir, "disease_confusion_normalized.png"),
        normalize=True,
    )

    # Condition confusion matrices
    cond_mask = labels_c != IGNORE_INDEX
    if cond_mask.sum() > 0:
        valid_preds_c = preds_c[cond_mask]
        valid_labels_c = labels_c[cond_mask]

        inv_condition = {v: k for k, v in condition_map.items()}
        condition_names = [inv_condition[i] for i in range(len(condition_map))]

        # Raw counts
        plot_confusion_matrix(
            valid_labels_c, valid_preds_c, condition_names,
            title="Condition Classification - Confusion Matrix (Counts)",
            save_path=os.path.join(args.out_dir, "condition_confusion_counts.png"),
        )

        # Normalized (recall %)
        plot_confusion_matrix(
            valid_labels_c, valid_preds_c, condition_names,
            title="Condition Classification - Confusion Matrix (Recall %)",
            save_path=os.path.join(args.out_dir, "condition_confusion_normalized.png"),
            normalize=True,
        )
    else:
        print("[WARN] No valid condition labels in test set - skipping condition matrix")

    # Classification reports
    print(f"\n{'='*60}")
    print(f"  DISEASE Report")
    print(f"{'='*60}")
    print(classification_report(labels_d, preds_d, target_names=disease_names, zero_division=0))

    if cond_mask.sum() > 0:
        print(f"\n{'='*60}")
        print(f"  CONDITION Report")
        print(f"{'='*60}")
        print(classification_report(
            valid_labels_c, valid_preds_c,
            target_names=condition_names, zero_division=0,
        ))

    print(f"\n[DONE] All outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()