"""
Measures how well the domain-adapted backbone learns from limited data.
Val and test sets are FIXED across all fractions.

Usage:
    # Domain-adapted backbone
    python finetune_octdl/data_efficiency.py \
        --data_path /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --checkpoint /home/student/Ungureanu_Daria/antrenare_oct_v2/model_final.rank_0.pth \
        --fractions 0.33 0.66 1.0 \
        --out_dir results/data_efficiency/domain_adapted

    # ImageNet baseline (no --checkpoint)
    python finetune_octdl/data_efficiency.py \
        --data_path /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --fractions 0.33 0.66 1.0 \
        --out_dir results/data_efficiency/imagenet_baseline

    # Plot comparison (after running both)
    python finetune_octdl/data_efficiency.py --plot_only \
        --results_a results/data_efficiency/domain_adapted/results.json \
        --results_b results/data_efficiency/imagenet_baseline/results.json \
        --label_a "Domain-adapted" --label_b "ImageNet baseline" \
        --out_dir results/data_efficiency
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    IGNORE_INDEX, OCTDLMultiTaskDataset, compute_class_weights,
    get_data_splits, get_eval_transform, get_train_transform,
)
from model import OCTDLMultiTaskModel, load_backbone


ARCH            = "dinov2_vits14"
IMG_SIZE        = 224
BATCH_SIZE      = 32
EPOCHS          = 30
LR_BACKBONE     = 1e-5
LR_HEADS        = 5e-4
WEIGHT_DECAY    = 0.05
WARMUP_EPOCHS   = 3
GRAD_CLIP       = 1.0
LAMBDA_COND     = 1.0
UNFREEZE_LAST_N = 2
HEAD_HIDDEN     = 256
HEAD_DROPOUT    = 0.3
PATIENCE        = 8
NUM_WORKERS     = 4


def compute_metrics(logits, labels, ignore_index=IGNORE_INDEX):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    mask = labels != ignore_index
    if mask.sum() == 0:
        return {"acc": 0.0, "bal_acc": 0.0, "macro_f1": 0.0}
    valid_preds = preds[mask]
    valid_labels = labels[mask]
    return {
        "acc": accuracy_score(valid_labels, valid_preds) * 100,
        "bal_acc": balanced_accuracy_score(valid_labels, valid_preds) * 100,
        "macro_f1": f1_score(valid_labels, valid_preds, average="macro", zero_division=0),
    }


def run_epoch(model, loader, criterion_d, criterion_c, device,
              optimizer=None, scheduler=None, grad_clip=None, lambda_cond=1.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_logits_d, all_labels_d = [], []
    all_logits_c, all_labels_c = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels_d, labels_c in loader:
            images = images.to(device, non_blocking=True)
            labels_d = labels_d.to(device, non_blocking=True)
            labels_c = labels_c.to(device, non_blocking=True)

            logits_d, logits_c = model(images)
            loss_d = criterion_d(logits_d, labels_d)
            loss_c = criterion_c(logits_c, labels_c)
            loss = loss_d + lambda_cond * loss_c

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            all_logits_d.append(logits_d.detach())
            all_labels_d.append(labels_d.detach())
            all_logits_c.append(logits_c.detach())
            all_labels_c.append(labels_c.detach())

    if is_train and scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(loader)
    metrics_d = compute_metrics(torch.cat(all_logits_d), torch.cat(all_labels_d))
    metrics_c = compute_metrics(torch.cat(all_logits_c), torch.cat(all_labels_c),
                                ignore_index=IGNORE_INDEX)
    return avg_loss, metrics_d, metrics_c


def subsample_train_df(train_df, fraction, random_state=42):
    """
    Subsample training data at patient level to preserve patient-based integrity.
    If fraction=1.0, returns the full training set unchanged.
    Stratified on disease to maintain class proportions.
    """
    if fraction >= 1.0:
        return train_df

    # Get unique patients with their majority disease label
    patients = train_df[["patient_id", "label_disease"]].drop_duplicates()
    patient_labels = patients.set_index("patient_id")["label_disease"]

    # Stratified subsample of patients
    selected, _ = train_test_split(
        patient_labels.index.to_numpy(),
        train_size=fraction,
        random_state=random_state,
        stratify=patient_labels.values,
    )

    subset_df = train_df[train_df["patient_id"].isin(selected)]
    print(f"Subsampled: {len(subset_df)} images "
          f"({len(selected)} patients) = {fraction:.0%} of training set")
    return subset_df


def train_and_evaluate(
    train_df, val_df, test_df, disease_map, condition_map,
    data_path, checkpoint, device, fraction_label,
):
    """Train a model on the given training subset and evaluate on fixed test set."""

    # Class weights from THIS subset
    weights_d = compute_class_weights(train_df, "label_disease", disease_map).to(device)
    weights_c = compute_class_weights(train_df, "label_condition_raw", condition_map).to(device)

    # Datasets
    train_transform = get_train_transform(IMG_SIZE)
    eval_transform = get_eval_transform(IMG_SIZE)

    train_ds = OCTDLMultiTaskDataset(train_df, data_path, train_transform, disease_map, condition_map)
    val_ds = OCTDLMultiTaskDataset(val_df, data_path, eval_transform, disease_map, condition_map)
    test_ds = OCTDLMultiTaskDataset(test_df, data_path, eval_transform, disease_map, condition_map)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # fresh backbone per fraction
    backbone = load_backbone(ARCH, checkpoint, device)
    model = OCTDLMultiTaskModel(
        backbone=backbone,
        num_diseases=len(disease_map),
        num_conditions=len(condition_map),
        freeze_backbone=(UNFREEZE_LAST_N < 12),
        unfreeze_last_n=UNFREEZE_LAST_N,
        head_hidden=HEAD_HIDDEN,
        head_dropout=HEAD_DROPOUT,
    ).to(device)

    # Optimizer
    param_groups = model.get_param_groups(LR_BACKBONE, LR_HEADS, WEIGHT_DECAY)
    optimizer = optim.AdamW(param_groups)

    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])

    criterion_d = nn.CrossEntropyLoss(weight=weights_d)
    criterion_c = nn.CrossEntropyLoss(weight=weights_c, ignore_index=IGNORE_INDEX)

    # Training loop
    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0

    print(f"\n--- Training with {fraction_label} ({len(train_ds)} images) ---")

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_d, t_c = run_epoch(
            model, train_loader, criterion_d, criterion_c, device,
            optimizer=optimizer, scheduler=scheduler, grad_clip=GRAD_CLIP,
            lambda_cond=LAMBDA_COND,
        )
        v_loss, v_d, v_c = run_epoch(
            model, val_loader, criterion_d, criterion_c, device,
            lambda_cond=LAMBDA_COND,
        )

        if v_d["macro_f1"] > best_val_f1:
            best_val_f1 = v_d["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "ok"
        else:
            patience_counter += 1
            marker = ""

        print(f"Ep {epoch:02d}  train_f1={t_d['macro_f1']:.3f}  "
              f"val_f1={v_d['macro_f1']:.3f}  best={best_val_f1:.3f} {marker}")

        if patience_counter >= PATIENCE:
            print(f"Early stop at epoch {epoch}")
            break

    # Evaluate on test with best checkpoint
    model.load_state_dict(best_state)
    model.eval()

    test_loss, test_d, test_c = run_epoch(
        model, test_loader, criterion_d, criterion_c, device,
        lambda_cond=LAMBDA_COND,
    )

    print(f"\nTEST: disease_acc={test_d['acc']:.2f}%  disease_f1={test_d['macro_f1']:.4f}  "
          f"cond_f1={test_c['macro_f1']:.4f}")

    return {
        "train_images": len(train_ds),
        "disease_acc": test_d["acc"],
        "disease_bal_acc": test_d["bal_acc"],
        "disease_f1": test_d["macro_f1"],
        "condition_acc": test_c["acc"],
        "condition_bal_acc": test_c["bal_acc"],
        "condition_f1": test_c["macro_f1"],
    }


def plot_efficiency_curve(results, fractions, out_dir, label=""):
    """Plot disease macro-F1 and accuracy vs training fraction."""
    f1s = [r["disease_f1"] for r in results]
    accs = [r["disease_acc"] for r in results]
    cond_f1s = [r["condition_f1"] for r in results]
    n_imgs = [r["train_images"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Disease
    ax1.plot(fractions, f1s, "o-", color="#2196F3", linewidth=2, markersize=8, label="Disease Macro-F1")
    ax1.plot(fractions, cond_f1s, "s--", color="#FF9800", linewidth=2, markersize=8, label="Condition Macro-F1")
    ax1.set_xlabel("Training Data Fraction", fontsize=12)
    ax1.set_ylabel("Macro-F1", fontsize=12)
    ax1.set_title(f"Data Efficiency - Macro-F1 {label}", fontsize=13, fontweight="bold")
    ax1.set_xticks(fractions)
    ax1.set_xticklabels([f"{f:.0%}\n({n} imgs)" for f, n in zip(fractions, n_imgs)])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.0])

    # Accuracy
    ax2.plot(fractions, accs, "o-", color="#4CAF50", linewidth=2, markersize=8, label="Disease Accuracy")
    ax2.set_xlabel("Training Data Fraction", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title(f"Data Efficiency - Accuracy {label}", fontsize=13, fontweight="bold")
    ax2.set_xticks(fractions)
    ax2.set_xticklabels([f"{f:.0%}\n({n} imgs)" for f, n in zip(fractions, n_imgs)])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([60, 100])

    plt.tight_layout()
    save_path = os.path.join(out_dir, "data_efficiency_curve.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n{save_path}")


def plot_comparison(results_a_path, results_b_path, label_a, label_b, out_dir):
    """Plot two data efficiency curves overlaid for comparison."""
    with open(results_a_path) as f:
        data_a = json.load(f)
    with open(results_b_path) as f:
        data_b = json.load(f)

    fracs_a = [r["fraction"] for r in data_a["runs"]]
    fracs_b = [r["fraction"] for r in data_b["runs"]]
    f1s_a = [r["disease_f1"] for r in data_a["runs"]]
    f1s_b = [r["disease_f1"] for r in data_b["runs"]]
    accs_a = [r["disease_acc"] for r in data_a["runs"]]
    accs_b = [r["disease_acc"] for r in data_b["runs"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(fracs_a, f1s_a, "o-", linewidth=2, markersize=8, label=label_a, color="#2196F3")
    ax1.plot(fracs_b, f1s_b, "s--", linewidth=2, markersize=8, label=label_b, color="#F44336")
    ax1.set_xlabel("Training Data Fraction", fontsize=12)
    ax1.set_ylabel("Disease Macro-F1", fontsize=12)
    ax1.set_title("Data Efficiency - Domain-Adapted vs ImageNet", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.4, 1.0])

    ax2.plot(fracs_a, accs_a, "o-", linewidth=2, markersize=8, label=label_a, color="#2196F3")
    ax2.plot(fracs_b, accs_b, "s--", linewidth=2, markersize=8, label=label_b, color="#F44336")
    ax2.set_xlabel("Training Data Fraction", fontsize=12)
    ax2.set_ylabel("Disease Accuracy (%)", fontsize=12)
    ax2.set_title("Data Efficiency - Accuracy Comparison", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([50, 100])

    plt.tight_layout()
    save_path = os.path.join(out_dir, "data_efficiency_comparison.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"{save_path}")


def main():
    parser = argparse.ArgumentParser(description="Data Efficiency Curve")

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Domain-adapted checkpoint. Omit for ImageNet baseline")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.33, 0.66, 1.0])
    parser.add_argument("--out_dir", type=str, default="results/data_efficiency")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)

    # Plot-only mode
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--results_a", type=str, default=None)
    parser.add_argument("--results_b", type=str, default=None)
    parser.add_argument("--label_a", type=str, default="Domain-adapted")
    parser.add_argument("--label_b", type=str, default="ImageNet baseline")

    args = parser.parse_args()

    # Plot-only mode
    if args.plot_only:
        if not args.results_a or not args.results_b:
            raise ValueError("--plot_only requires --results_a and --results_b")
        plot_comparison(args.results_a, args.results_b, args.label_a, args.label_b, args.out_dir)
        return

    if not args.data_path:
        raise ValueError("--data_path is required")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # fixed val/test across all fractions
    csv_path = os.path.join(args.data_path, "OCTDL_clean_metadata.csv")
    train_df, val_df, test_df, disease_map, condition_map = get_data_splits(csv_path)

    print(f"Full train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Fractions to test: {args.fractions}")

    checkpoint_label = "domain_adapted" if args.checkpoint else "imagenet_baseline"

    # Run each fraction
    all_results = []
    for frac in args.fractions:
        subset_df = subsample_train_df(train_df, frac)

        result = train_and_evaluate(
            subset_df, val_df, test_df, disease_map, condition_map,
            args.data_path, args.checkpoint, device,
            fraction_label=f"{frac:.0%}",
        )
        result["fraction"] = frac
        all_results.append(result)

    # Save results
    output = {
        "checkpoint": args.checkpoint or "ImageNet baseline",
        "fractions": args.fractions,
        "runs": all_results,
    }
    json_path = os.path.join(args.out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n{json_path}")

    # Plot
    plot_efficiency_curve(all_results, args.fractions, args.out_dir, label=f"({checkpoint_label})")

    # Summary table
    print(f"\n{'='*70}")
    print(f"data efficiency summary - {checkpoint_label}")
    print(f"{'Fraction':<10} {'N_train':<10} {'Disease F1':<12} {'Disease Acc':<12} {'Cond F1':<12}")
    for r in all_results:
        print(f"{r['fraction']:<10.0%} {r['train_images']:<10} "
              f"{r['disease_f1']:<12.4f} {r['disease_acc']:<12.2f} {r['condition_f1']:<12.4f}")


if __name__ == "__main__":
    main()