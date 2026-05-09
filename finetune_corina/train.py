"""
Corina Fine-Tuning - Multi-label DME biomarker detection.

4 sigmoid outputs: DME, HF, ND, Healthy.
Loss: BCEWithLogitsLoss with pos_weight for class imbalance.
Metrics: per-biomarker AUC-ROC, F1, accuracy + macro averages.

Usage:

    # Domain-adapted backbone
    python finetune_corina/train.py \
        --data_path /home/student/Ungureanu_Daria/corina_dataset \
        --csv /home/student/Ungureanu_Daria/corina_dataset/corina_metadata.csv \
        --checkpoint /home/student/Ungureanu_Daria/antrenare_oct_v2/model_final.rank_0.pth \
        --save_dir saved_models/corina_domain_adapted \
        --run_name "corina_domain_adapted"

    # ImageNet baseline
    python finetune_corina/train.py \
        --data_path //home/student/Ungureanu_Daria/corina_dataset \
        --csv /home/student/Ungureanu_Daria/corina_dataset/corina_metadata.csv \
        --save_dir saved_models/corina_imagenet \
        --run_name "corina_imagenet"
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    CorinaDataset, load_corina_splits, compute_pos_weights,
    get_train_transform, get_eval_transform, BIOMARKERS, NUM_LABELS,
)
from model import CorinaModel, load_backbone

ARCH            = "dinov2_vits14"
IMG_SIZE        = 224
BATCH_SIZE      = 32
EPOCHS          = 30
LR_BACKBONE     = 1e-5
LR_HEADS        = 5e-4
WEIGHT_DECAY    = 0.05
WARMUP_EPOCHS   = 3
GRAD_CLIP       = 1.0
UNFREEZE_LAST_N = 2
HEAD_HIDDEN     = 256
HEAD_DROPOUT    = 0.3
PATIENCE        = 8
NUM_WORKERS     = 4
THRESHOLD       = 0.5  # sigmoid threshold for binary predictions


def compute_multilabel_metrics(logits, labels, threshold=THRESHOLD):
    """
    Multi-label metrics: per-biomarker and macro-averaged.
    logits: raw model output (before sigmoid)
    labels: ground truth binary vectors
    """
    if isinstance(logits, torch.Tensor):
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = labels.cpu().numpy()
    else:
        probs = logits

    preds = (probs >= threshold).astype(int)

    metrics = {}

    f1_scores = []
    auc_scores = []
    acc_scores = []

    for i, bm in enumerate(BIOMARKERS):
        f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
        f1_scores.append(f1)
        metrics[f"f1_{bm}"] = f1

        acc = accuracy_score(labels[:, i], preds[:, i])
        acc_scores.append(acc)
        metrics[f"acc_{bm}"] = acc * 100

        # AUC-ROC needs both classes present in the batch.
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            auc = 0.5
        auc_scores.append(auc)
        metrics[f"auc_{bm}"] = auc

    metrics["f1_macro"] = np.mean(f1_scores)
    metrics["auc_macro"] = np.mean(auc_scores)
    metrics["acc_macro"] = np.mean(acc_scores) * 100

    # Exact match: all 4 labels correct simultaneously.
    exact_match = np.all(preds == labels, axis=1).mean() * 100
    metrics["exact_match"] = exact_match

    return metrics


def run_epoch(model, loader, criterion, device,
              optimizer=None, scheduler=None, grad_clip=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_logits, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            all_logits.append(logits.detach())
            all_labels.append(labels.detach())

    if is_train and scheduler is not None:
        scheduler.step()

    cat_logits = torch.cat(all_logits)
    cat_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(loader)
    metrics = compute_multilabel_metrics(cat_logits, cat_labels)

    return avg_loss, metrics


def evaluate_test(model, loader, criterion, device, out_dir):
    """Full test eval with per-biomarker reports and confusion matrices."""
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Test Eval", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            all_logits.append(logits)
            all_labels.append(labels)

    cat_logits = torch.cat(all_logits)
    cat_labels = torch.cat(all_labels)
    metrics = compute_multilabel_metrics(cat_logits, cat_labels)

    probs = torch.sigmoid(cat_logits).cpu().numpy()
    preds = (probs >= THRESHOLD).astype(int)
    labels_np = cat_labels.cpu().numpy()

    print("  BIOMARKER DETECTION REPORT")
    for i, bm in enumerate(BIOMARKERS):
        print(f"\n  {bm}:")
        print(f"    F1={metrics[f'f1_{bm}']:.4f}  "
              f"AUC={metrics[f'auc_{bm}']:.4f}  "
              f"Acc={metrics[f'acc_{bm}']:.1f}%")

    print(f"\n  Macro F1:     {metrics['f1_macro']:.4f}")
    print(f"  Macro AUC:    {metrics['auc_macro']:.4f}")
    print(f"  Exact Match:  {metrics['exact_match']:.1f}%")

    fig, axes = plt.subplots(1, NUM_LABELS, figsize=(5 * NUM_LABELS, 4))
    for i, bm in enumerate(BIOMARKERS):
        cm = confusion_matrix(labels_np[:, i], preds[:, i])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Absent", "Present"],
                    yticklabels=["Absent", "Present"],
                    ax=axes[i])
        axes[i].set_title(f"{bm}\nF1={metrics[f'f1_{bm}']:.3f} AUC={metrics[f'auc_{bm}']:.3f}",
                          fontsize=11)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    fig.suptitle("Corina - Per-Biomarker Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrices.png")
    fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"{cm_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Corina Multi-Label Fine-Tuning")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="saved_models/corina")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr_backbone", type=float, default=LR_BACKBONE)
    parser.add_argument("--lr_heads", type=float, default=LR_HEADS)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--grad_clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--patience", type=int, default=PATIENCE)

    parser.add_argument("--arch", type=str, default=ARCH)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--unfreeze_last_n", type=int, default=UNFREEZE_LAST_N)
    parser.add_argument("--head_hidden", type=int, default=HEAD_HIDDEN)
    parser.add_argument("--head_dropout", type=float, default=HEAD_DROPOUT)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    train_df, val_df, test_df = load_corina_splits(args.csv, args.data_path)

    pos_weights = compute_pos_weights(train_df).to(device)

    train_ds = CorinaDataset(train_df, args.data_path, get_train_transform(args.img_size))
    val_ds   = CorinaDataset(val_df, args.data_path, get_eval_transform(args.img_size))
    test_ds  = CorinaDataset(test_df, args.data_path, get_eval_transform(args.img_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"Batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    backbone = load_backbone(args.arch, args.checkpoint, device)
    model = CorinaModel(
        backbone=backbone,
        num_labels=NUM_LABELS,
        freeze_backbone=(args.unfreeze_last_n < 12),
        unfreeze_last_n=args.unfreeze_last_n,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
    ).to(device)

    param_groups = model.get_param_groups(args.lr_backbone, args.lr_heads, args.weight_decay)
    optimizer = optim.AdamW(param_groups)

    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_epochs])

    # pos_weight handles class imbalance per biomarker.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    run_name = args.run_name or f"{args.arch}_corina_unfreeze{args.unfreeze_last_n}"
    wandb.init(
        project="Corina-FineTune",
        name=run_name,
        config=vars(args),
    )

    best_val_f1 = 0.0
    patience_counter = 0
    best_epoch = 0

    print(f"  TRAINING - {args.epochs} epochs, {NUM_LABELS} biomarkers")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        t_loss, t_met = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scheduler=scheduler, grad_clip=args.grad_clip,
        )
        v_loss, v_met = run_epoch(
            model, val_loader, criterion, device,
        )

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d}/{args.epochs} ({elapsed:.0f}s) lr={lr:.2e}")
        print(f"  Train  loss={t_loss:.4f}  F1_macro={t_met['f1_macro']:.4f}  "
              f"AUC_macro={t_met['auc_macro']:.4f}")
        print(f"  Val    loss={v_loss:.4f}  F1_macro={v_met['f1_macro']:.4f}  "
              f"AUC_macro={v_met['auc_macro']:.4f}  "
              f"(best_f1={best_val_f1:.4f})")

        log_dict = {
            "epoch": epoch, "lr": lr,
            "train/loss": t_loss, "val/loss": v_loss,
            "train/f1_macro": t_met["f1_macro"],
            "train/auc_macro": t_met["auc_macro"],
            "val/f1_macro": v_met["f1_macro"],
            "val/auc_macro": v_met["auc_macro"],
            "val/exact_match": v_met["exact_match"],
        }
        for bm in BIOMARKERS:
            log_dict[f"val/f1_{bm}"] = v_met[f"f1_{bm}"]
            log_dict[f"val/auc_{bm}"] = v_met[f"auc_{bm}"]
        wandb.log(log_dict)

        if v_met["f1_macro"] > best_val_f1:
            best_val_f1 = v_met["f1_macro"]
            best_epoch = epoch
            patience_counter = 0

            save_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1_macro": best_val_f1,
                "val_metrics": v_met,
                "config": vars(args),
                "num_labels": NUM_LABELS,
                "biomarkers": BIOMARKERS,
            }, save_path)
            print(f" New best! Saved -> {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOP] Best: epoch {best_epoch}, F1={best_val_f1:.4f}")
                break

    print(f"  FINAL TEST (best checkpoint: epoch {best_epoch})")
    best_ckpt = torch.load(os.path.join(args.save_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_met = evaluate_test(model, test_loader, criterion, device, args.save_dir)

    print("  FINAL RESULTS")
    for bm in BIOMARKERS:
        print(f"  {bm:>8}: F1={test_met[f'f1_{bm}']:.4f}  "
              f"AUC={test_met[f'auc_{bm}']:.4f}  "
              f"Acc={test_met[f'acc_{bm}']:.1f}%")
    print(f"  {'Macro':>8}: F1={test_met['f1_macro']:.4f}  "
          f"AUC={test_met['auc_macro']:.4f}")
    print(f"  Exact Match: {test_met['exact_match']:.1f}%")

    results = {
        "checkpoint": args.checkpoint or "ImageNet baseline",
        "best_epoch": best_epoch,
        "test_metrics": test_met,
        "config": vars(args),
    }
    json_path = os.path.join(args.save_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{json_path}")

    test_log = {"test/f1_macro": test_met["f1_macro"],
                "test/auc_macro": test_met["auc_macro"],
                "test/exact_match": test_met["exact_match"]}
    for bm in BIOMARKERS:
        test_log[f"test/f1_{bm}"] = test_met[f"f1_{bm}"]
        test_log[f"test/auc_{bm}"] = test_met[f"auc_{bm}"]
    wandb.log(test_log)
    wandb.finish()

    print("Done.")


if __name__ == "__main__":
    main()