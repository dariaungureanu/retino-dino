"""
Multi-task fine-tuning pipeline for OCTDL: DINOv2 ViT-S/14 domain-adapted
backbone with a dual-head classifier (disease + condition).

Usage:
    python finetune_octdl/train.py \
        --data_path /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --checkpoint /home/student/Ungureanu_Daria/antrenare_oct_v2/model_final.rank_0.pth \
        --save_dir saved_models \
        --epochs 30 \
        --img_size 224 \
        --unfreeze_last_n 2
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from dataset import (
    IGNORE_INDEX,
    OCTDLMultiTaskDataset,
    compute_class_weights,
    get_data_splits,
    get_eval_transform,
    get_train_transform,
)
from model import OCTDLMultiTaskModel, load_backbone

ARCH            = "dinov2_vits14"
IMG_SIZE        = 224          # match DINOv2 eval resolution
BATCH_SIZE      = 32
EPOCHS          = 30
LR_BACKBONE     = 1e-5         # Low LR for domain-adapted backbone
LR_HEADS        = 5e-4         # Higher LR for randomly-initialized heads
WEIGHT_DECAY    = 0.05         # Standard for ViT fine-tuning (AdamW)
WARMUP_EPOCHS   = 3            # Linear warmup before cosine decay
GRAD_CLIP       = 1.0          # Max gradient norm
LAMBDA_COND     = 1.0          # Condition loss weight in L_total
UNFREEZE_LAST_N = 2            # Blocks 10-11 + norm
HEAD_HIDDEN     = 256
HEAD_DROPOUT    = 0.3
PATIENCE        = 8            # Early stopping patience (epochs)
NUM_WORKERS     = 4


def compute_metrics(logits, labels, ignore_index=IGNORE_INDEX):
    """
    Compute accuracy, balanced accuracy, and macro-F1.
    Handles ignore_index for condition labels.
    """
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    mask = labels != ignore_index
    if mask.sum() == 0:
        return {"acc": 0.0, "bal_acc": 0.0, "macro_f1": 0.0}

    valid_preds  = preds[mask]
    valid_labels = labels[mask]

    return {
        "acc":      accuracy_score(valid_labels, valid_preds) * 100,
        "bal_acc":  balanced_accuracy_score(valid_labels, valid_preds) * 100,
        "macro_f1": f1_score(valid_labels, valid_preds, average="macro", zero_division=0),
    }


def run_epoch(model, loader, criterion_d, criterion_c, device, optimizer=None,
              scheduler=None, grad_clip=None, epoch=0, phase="train", lambda_cond=1.0):
    """Unified train/eval epoch. optimizer=None means eval mode (no grads)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_logits_d, all_labels_d = [], []
    all_logits_c, all_labels_c = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    desc = f"Ep {epoch:02d} [{phase.capitalize():>5}]"

    with ctx:
        for images, labels_d, labels_c in tqdm(loader, desc=desc, leave=False):
            images   = images.to(device, non_blocking=True)
            labels_d = labels_d.to(device, non_blocking=True)
            labels_c = labels_c.to(device, non_blocking=True)

            logits_d, logits_c = model(images)

            loss_d = criterion_d(logits_d, labels_d)
            loss_c = criterion_c(logits_c, labels_c)
            loss   = loss_d + lambda_cond * loss_c

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

    # Step scheduler per epoch (after all batches)
    if is_train and scheduler is not None:
        scheduler.step()

    # Aggregate metrics
    avg_loss   = total_loss / len(loader)
    metrics_d  = compute_metrics(torch.cat(all_logits_d), torch.cat(all_labels_d))
    metrics_c  = compute_metrics(torch.cat(all_logits_c), torch.cat(all_labels_c),
                                  ignore_index=IGNORE_INDEX)

    return avg_loss, metrics_d, metrics_c


def evaluate_test(model, loader, criterion_d, criterion_c, device,
                  disease_map, condition_map):
    """
    Full test evaluation with per-class classification reports.
    Called once at the end on the best checkpoint.
    """
    model.eval()
    all_logits_d, all_labels_d = [], []
    all_logits_c, all_labels_c = [], []

    with torch.no_grad():
        for images, labels_d, labels_c in tqdm(loader, desc="Test Eval", leave=False):
            images   = images.to(device, non_blocking=True)
            labels_d = labels_d.to(device, non_blocking=True)
            labels_c = labels_c.to(device, non_blocking=True)

            logits_d, logits_c = model(images)
            all_logits_d.append(logits_d)
            all_labels_d.append(labels_d)
            all_logits_c.append(logits_c)
            all_labels_c.append(labels_c)

    cat_logits_d = torch.cat(all_logits_d)
    cat_labels_d = torch.cat(all_labels_d)
    cat_logits_c = torch.cat(all_logits_c)
    cat_labels_c = torch.cat(all_labels_c)

    metrics_d = compute_metrics(cat_logits_d, cat_labels_d)
    metrics_c = compute_metrics(cat_logits_c, cat_labels_c, ignore_index=IGNORE_INDEX)

    # Per-class reports
    inv_disease = {v: k for k, v in disease_map.items()}
    inv_condition = {v: k for k, v in condition_map.items()}

    preds_d  = torch.argmax(cat_logits_d, dim=1).cpu().numpy()
    labels_d = cat_labels_d.cpu().numpy()

    preds_c  = torch.argmax(cat_logits_c, dim=1).cpu().numpy()
    labels_c = cat_labels_c.cpu().numpy()
    cond_mask = labels_c != IGNORE_INDEX

    print("DISEASE Classification Report")
    print(classification_report(
        labels_d, preds_d,
        target_names=[inv_disease[i] for i in range(len(disease_map))],
        zero_division=0,
    ))

    print("CONDITION Classification Report")
    if cond_mask.sum() > 0:
        print(classification_report(
            labels_c[cond_mask], preds_c[cond_mask],
            target_names=[inv_condition[i] for i in range(len(condition_map))],
            zero_division=0,
        ))
    else:
        print("no valid condition labels in test set.")

    return metrics_d, metrics_c


def main():
    parser = argparse.ArgumentParser(description="OCTDL Multi-Task Fine-Tuning")

    # Paths
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to OCTDL_Cleaned directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Domain-adapted checkpoint. None = ImageNet baseline")
    parser.add_argument("--save_dir", type=str, default="saved_models")

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
    parser.add_argument("--lambda_cond", type=float, default=LAMBDA_COND)

    # Infrastructure
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--wandb_project", type=str, default="OCTDL-ViTS14-FineTune")
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(args.save_dir, exist_ok=True)

    # Validate resolution
    if args.img_size % 14 != 0:
        print(f"img_size={args.img_size} not divisible by 14 (patch size)!")
    grid = args.img_size // 14
    print(f"resolution: {args.img_size}x{args.img_size} -> {grid}x{grid} = {grid**2} patches")

    # Data
    csv_path = os.path.join(args.data_path, "OCTDL_clean_metadata.csv")
    train_df, val_df, test_df, disease_map, condition_map = get_data_splits(csv_path)

    # Class weights (computed on training set only)
    weights_d = compute_class_weights(train_df, "label_disease", disease_map).to(device)
    weights_c = compute_class_weights(train_df, "label_condition_raw", condition_map).to(device)

    # Transforms
    train_transform = get_train_transform(args.img_size)
    eval_transform  = get_eval_transform(args.img_size)

    # Datasets & loaders
    train_ds = OCTDLMultiTaskDataset(train_df, args.data_path, train_transform, disease_map, condition_map)
    val_ds   = OCTDLMultiTaskDataset(val_df, args.data_path, eval_transform, disease_map, condition_map)
    test_ds  = OCTDLMultiTaskDataset(test_df, args.data_path, eval_transform, disease_map, condition_map)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"batches per epoch: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    # Model
    backbone = load_backbone(args.arch, args.checkpoint, device)
    model = OCTDLMultiTaskModel(
        backbone=backbone,
        num_diseases=len(disease_map),
        num_conditions=len(condition_map),
        freeze_backbone=(args.unfreeze_last_n < 12),  # 12 means full fine-tune
        unfreeze_last_n=args.unfreeze_last_n,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
    ).to(device)

    # Optimizer with differential LR
    param_groups = model.get_param_groups(
        lr_backbone=args.lr_backbone,
        lr_heads=args.lr_heads,
        weight_decay=args.weight_decay,
    )
    optimizer = optim.AdamW(param_groups)

    # LR scheduler: linear warmup, then cosine decay
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, total_iters=args.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-7,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    # Loss functions
    criterion_d = nn.CrossEntropyLoss(weight=weights_d)
    criterion_c = nn.CrossEntropyLoss(weight=weights_c, ignore_index=IGNORE_INDEX)

    # WandB
    run_name = args.run_name or f"{args.arch}_unfreeze{args.unfreeze_last_n}_{args.img_size}px"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "arch": args.arch,
            "img_size": args.img_size,
            "checkpoint": args.checkpoint,
            "unfreeze_last_n": args.unfreeze_last_n,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr_backbone": args.lr_backbone,
            "lr_heads": args.lr_heads,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "grad_clip": args.grad_clip,
            "lambda_condition": args.lambda_cond,
            "head_hidden": args.head_hidden,
            "head_dropout": args.head_dropout,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "disease_classes": disease_map,
            "condition_classes": condition_map,
        },
    )

    # Training loop
    best_val_f1 = 0.0
    patience_counter = 0
    best_epoch = 0

    print(f"training start - {args.epochs} epochs")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        t_loss, t_met_d, t_met_c = run_epoch(
            model, train_loader, criterion_d, criterion_c, device,
            optimizer=optimizer, scheduler=scheduler,
            grad_clip=args.grad_clip, epoch=epoch, phase="train",
            lambda_cond=args.lambda_cond,
        )

        # Validate
        v_loss, v_met_d, v_met_c = run_epoch(
            model, val_loader, criterion_d, criterion_c, device,
            epoch=epoch, phase="val", lambda_cond=args.lambda_cond,
        )

        elapsed = time.time() - t0

        # Current LR (from first param group = backbone)
        current_lr = optimizer.param_groups[0]["lr"]

        # Print summary
        print(f"\nepoch {epoch:02d}/{args.epochs} ({elapsed:.0f}s)  lr={current_lr:.2e}")
        print(f"train  loss={t_loss:.4f}  disease_F1={t_met_d['macro_f1']:.4f}  "
              f"cond_F1={t_met_c['macro_f1']:.4f}")
        print(f"val    loss={v_loss:.4f}  disease_F1={v_met_d['macro_f1']:.4f}  "
              f"cond_F1={v_met_c['macro_f1']:.4f}  "
              f"(best={best_val_f1:.4f})")

        # WandB logging
        wandb.log({
            "epoch": epoch,
            "lr": current_lr,
            "train/loss": t_loss,
            "train/disease_acc": t_met_d["acc"],
            "train/disease_bal_acc": t_met_d["bal_acc"],
            "train/disease_f1": t_met_d["macro_f1"],
            "train/condition_acc": t_met_c["acc"],
            "train/condition_bal_acc": t_met_c["bal_acc"],
            "train/condition_f1": t_met_c["macro_f1"],
            "val/loss": v_loss,
            "val/disease_acc": v_met_d["acc"],
            "val/disease_bal_acc": v_met_d["bal_acc"],
            "val/disease_f1": v_met_d["macro_f1"],
            "val/condition_acc": v_met_c["acc"],
            "val/condition_bal_acc": v_met_c["bal_acc"],
            "val/condition_f1": v_met_c["macro_f1"],
        })

        # Checkpointing on best val disease F1
        if v_met_d["macro_f1"] > best_val_f1:
            best_val_f1 = v_met_d["macro_f1"]
            best_epoch = epoch
            patience_counter = 0

            save_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_disease_f1": best_val_f1,
                "val_condition_f1": v_met_c["macro_f1"],
                "config": vars(args),
                "disease_map": disease_map,
                "condition_map": condition_map,
            }, save_path)
            print(f"new best! Saved -> {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[EARLY STOP] No improvement for {args.patience} epochs. "
                      f"Best: epoch {best_epoch}, F1={best_val_f1:.4f}")
                break

    # Test evaluation on best checkpoint
    print(f"final test (best checkpoint: epoch {best_epoch})")
    best_ckpt = torch.load(
        os.path.join(args.save_dir, "best_model.pth"),
        map_location=device,
    )
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_d, test_c = evaluate_test(
        model, test_loader, criterion_d, criterion_c, device,
        disease_map, condition_map,
    )

    print("final results")
    print(f"disease:   acc={test_d['acc']:.2f}%  bal_acc={test_d['bal_acc']:.2f}%  "
          f"macro_F1={test_d['macro_f1']:.4f}")
    print(f"condition: acc={test_c['acc']:.2f}%  bal_acc={test_c['bal_acc']:.2f}%  "
          f"macro_F1={test_c['macro_f1']:.4f}")

    wandb.log({
        "test/disease_acc": test_d["acc"],
        "test/disease_bal_acc": test_d["bal_acc"],
        "test/disease_f1": test_d["macro_f1"],
        "test/condition_acc": test_c["acc"],
        "test/condition_bal_acc": test_c["bal_acc"],
        "test/condition_f1": test_c["macro_f1"],
    })

    wandb.finish()
    print("\ndone.")


if __name__ == "__main__":
    main()