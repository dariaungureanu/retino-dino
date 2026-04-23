"""
MMRDR-OCT Fine-Tuning — 3-class DME severity grading.

Single-task version of the OCTDL pipeline.
Metrics: Accuracy, Balanced Acc, Macro-F1, AUC-ROC, Cohen's Kappa
(matches the paper's evaluation for direct comparison).

Usage:
    # Domain-adapted backbone
    python finetune_mmrdr/train.py \
        --data_path /home/student/Ungureanu_Daria/MMRDR-OCT \
        --csv //home/student/Ungureanu_Daria/MMRDR-OCT/OCT.csv \
        --checkpoint /home/student/Ungureanu_Daria/antrenare_oct_v2/model_final.rank_0.pth \
        --save_dir saved_models/mmrdr_domain_adapted \
        --run_name "mmrdr_domain_adapted"

    # ImageNet baseline (no --checkpoint)
    python finetune_mmrdr/train.py \
        --data_path /home/student/Ungureanu_Daria/MMRDR-OCT \
        --csv /home/student/Ungureanu_Daria/MMRDR-OCT/OCT.csv \
        --save_dir saved_models/mmrdr_imagenet \
        --run_name "mmrdr_imagenet"
"""

import argparse
import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, cohen_kappa_score, roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import (
    MMRDRDataset, load_mmrdr_splits, compute_class_weights,
    get_train_transform, get_eval_transform, CLASS_NAMES,
)
from model import MMRDRModel, load_backbone

# ── Defaults (same as OCTDL Run C) ────────────────────────────
ARCH = "dinov2_vits14"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LR_BACKBONE = 1e-5
LR_HEADS = 5e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 3
GRAD_CLIP = 1.0
UNFREEZE_LAST_N = 2
HEAD_HIDDEN = 256
HEAD_DROPOUT = 0.3
PATIENCE = 8
NUM_WORKERS = 4


def compute_all_metrics(logits_or_probs, labels, num_classes):
    """
    Compute all metrics the paper uses:
    accuracy, balanced_accuracy, macro_f1, auc_roc, kappa
    """
    if isinstance(logits_or_probs, torch.Tensor):
        probs = torch.softmax(logits_or_probs, dim=1).cpu().numpy()
        preds = torch.argmax(logits_or_probs, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
    else:
        probs = logits_or_probs
        preds = np.argmax(probs, axis=1)

    acc = accuracy_score(labels, preds) * 100
    bal_acc = balanced_accuracy_score(labels, preds) * 100
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(labels, preds)

    # AUC-ROC (one-vs-rest, macro averaged)
    try:
        labels_bin = label_binarize(labels, classes=list(range(num_classes)))
        auc_roc = roc_auc_score(labels_bin, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc_roc = 0.0  # edge case: only one class present in batch

    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "auc_roc": auc_roc,
        "kappa": kappa,
    }


def run_epoch(model, loader, criterion, device, num_classes,
              optimizer=None, scheduler=None, grad_clip=None):
    """Single-task train/eval epoch."""
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
    metrics = compute_all_metrics(cat_logits, cat_labels, num_classes)

    return avg_loss, metrics


def evaluate_test(model, loader, criterion, device, num_classes, out_dir):
    """Full test evaluation with reports + confusion matrix."""
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
    metrics = compute_all_metrics(cat_logits, cat_labels, num_classes)

    preds = torch.argmax(cat_logits, dim=1).cpu().numpy()
    labels_np = cat_labels.cpu().numpy()
    class_names = [CLASS_NAMES[i] for i in range(num_classes)]

    # Classification report
    print(f"\n{'=' * 60}")
    print(f"  DME Classification Report")
    print(f"{'=' * 60}")
    print(classification_report(labels_np, preds, target_names=class_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(labels_np, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    ax.set_title("MMRDR-OCT DME Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {cm_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="MMRDR-OCT DME Fine-Tuning")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root directory of MMRDR-OCT (contains img/ folder)")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to OCT.csv")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Domain-adapted checkpoint. None = ImageNet baseline.")
    parser.add_argument("--save_dir", type=str, default="saved_models/mmrdr")
    parser.add_argument("--run_name", type=str, default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr_backbone", type=float, default=LR_BACKBONE)
    parser.add_argument("--lr_heads", type=float, default=LR_HEADS)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--grad_clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--patience", type=int, default=PATIENCE)

    # Architecture
    parser.add_argument("--arch", type=str, default=ARCH)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--unfreeze_last_n", type=int, default=UNFREEZE_LAST_N)
    parser.add_argument("--head_hidden", type=int, default=HEAD_HIDDEN)
    parser.add_argument("--head_dropout", type=float, default=HEAD_DROPOUT)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Data ───────────────────────────────────────────────────
    train_df, val_df, test_df, num_classes = load_mmrdr_splits(
        args.csv, args.data_path, val_size=0.1,
    )

    weights = compute_class_weights(train_df, num_classes).to(device)

    train_ds = MMRDRDataset(train_df, args.data_path, get_train_transform(args.img_size))
    val_ds = MMRDRDataset(val_df, args.data_path, get_eval_transform(args.img_size))
    test_ds = MMRDRDataset(test_df, args.data_path, get_eval_transform(args.img_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"[DATA] Batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    # ── Model ──────────────────────────────────────────────────
    backbone = load_backbone(args.arch, args.checkpoint, device)
    model = MMRDRModel(
        backbone=backbone,
        num_classes=num_classes,
        freeze_backbone=(args.unfreeze_last_n < 12),
        unfreeze_last_n=args.unfreeze_last_n,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
    ).to(device)

    # ── Optimizer ──────────────────────────────────────────────
    param_groups = model.get_param_groups(args.lr_backbone, args.lr_heads, args.weight_decay)
    optimizer = optim.AdamW(param_groups)

    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_epochs])

    criterion = nn.CrossEntropyLoss(weight=weights)

    # ── WandB ──────────────────────────────────────────────────
    run_name = args.run_name or f"mmrdr_{args.arch}_unfreeze{args.unfreeze_last_n}"
    wandb.init(
        project="MMRDR-OCT-FineTune",
        name=run_name,
        config=vars(args),
    )

    # ── Training ───────────────────────────────────────────────
    best_val_f1 = 0.0
    patience_counter = 0
    best_epoch = 0

    print(f"\n{'=' * 60}")
    print(f"  TRAINING — {args.epochs} epochs, {num_classes} classes")
    print(f"{'=' * 60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        t_loss, t_met = run_epoch(
            model, train_loader, criterion, device, num_classes,
            optimizer=optimizer, scheduler=scheduler, grad_clip=args.grad_clip,
        )
        v_loss, v_met = run_epoch(
            model, val_loader, criterion, device, num_classes,
        )

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d}/{args.epochs} ({elapsed:.0f}s) lr={lr:.2e}")
        print(f"  Train │ loss={t_loss:.4f}  F1={t_met['macro_f1']:.4f}  "
              f"AUC={t_met['auc_roc']:.4f}  Kappa={t_met['kappa']:.4f}")
        print(f"  Val   │ loss={v_loss:.4f}  F1={v_met['macro_f1']:.4f}  "
              f"AUC={v_met['auc_roc']:.4f}  Kappa={v_met['kappa']:.4f}  "
              f"(best_f1={best_val_f1:.4f})")

        wandb.log({
            "epoch": epoch, "lr": lr,
            "train/loss": t_loss, "train/f1": t_met["macro_f1"],
            "train/acc": t_met["acc"], "train/auc": t_met["auc_roc"],
            "train/kappa": t_met["kappa"],
            "val/loss": v_loss, "val/f1": v_met["macro_f1"],
            "val/acc": v_met["acc"], "val/auc": v_met["auc_roc"],
            "val/kappa": v_met["kappa"],
        })

        if v_met["macro_f1"] > best_val_f1:
            best_val_f1 = v_met["macro_f1"]
            best_epoch = epoch
            patience_counter = 0

            save_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": best_val_f1,
                "val_metrics": v_met,
                "config": vars(args),
                "num_classes": num_classes,
            }, save_path)
            print(f"  ✓ New best! Saved → {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOP] Best: epoch {best_epoch}, F1={best_val_f1:.4f}")
                break

    # ── Test ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  FINAL TEST (best checkpoint: epoch {best_epoch})")
    print(f"{'=' * 60}")

    best_ckpt = torch.load(os.path.join(args.save_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_met = evaluate_test(model, test_loader, criterion, device, num_classes, args.save_dir)

    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"  Accuracy:     {test_met['acc']:.2f}%")
    print(f"  Balanced Acc: {test_met['bal_acc']:.2f}%")
    print(f"  Macro-F1:     {test_met['macro_f1']:.4f}")
    print(f"  AUC-ROC:      {test_met['auc_roc']:.4f}")
    print(f"  Kappa:        {test_met['kappa']:.4f}")

    # Save results JSON
    results = {
        "checkpoint": args.checkpoint or "ImageNet baseline",
        "best_epoch": best_epoch,
        "test_metrics": test_met,
        "config": vars(args),
    }
    json_path = os.path.join(args.save_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {json_path}")

    wandb.log({
        "test/acc": test_met["acc"], "test/bal_acc": test_met["bal_acc"],
        "test/f1": test_met["macro_f1"], "test/auc": test_met["auc_roc"],
        "test/kappa": test_met["kappa"],
    })
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
