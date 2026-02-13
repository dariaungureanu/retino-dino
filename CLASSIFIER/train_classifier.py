import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

# Importurile tale locale
from dataset import get_data_splits, OCTDLMultiTaskDataset
from model import OCTDLMultiTaskModel

# --- CONFIGURARE AUTOMATĂ ---
# Căutăm checkpoint-ul în folderul vecin (1_PRETRAIN)
DEFAULT_SSL_CHECKPOINT = os.path.join("..", "checkpoints_ssl", "checkpoint_latest.pth")

# Hyperparameters pentru Server
BATCH_SIZE = 32  # RTX 3060 duce 32 sau chiar 64 la 224px
LEARNING_RATE = 1e-4
EPOCHS = 30  # Antrenare serioasă
LAMBDA_CONDITION = 1.0
WEIGHT_DECAY = 1e-4


def compute_class_weights(df, col_name, num_classes, ignore_index=-100):
    counts = df[col_name].value_counts()
    total = len(df)
    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        count = counts.get(i, 0)
        if count == 0:
            weights[i] = 0.0
        else:
            weights[i] = total / (num_classes * count)
    return weights


def calculate_metrics_sklearn(logits, labels, ignore_index=-100):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    mask = labels != ignore_index
    valid_preds = preds[mask]
    valid_labels = labels[mask]
    if len(valid_labels) == 0:
        return 0.0, 0.0
    acc = accuracy_score(valid_labels, valid_preds)
    f1 = f1_score(valid_labels, valid_preds, average='macro', zero_division=0)
    return acc * 100, f1


def train_one_epoch(model, loader, optimizer, criterion_d, criterion_c, device, epoch):
    model.train()
    total_loss = 0
    all_preds_d, all_labels_d = [], []
    all_preds_c, all_labels_c = [], []

    progress_bar = tqdm(loader, desc=f"Ep {epoch} [Train]")

    for images, labels_disease, labels_condition in progress_bar:
        images = images.to(device)
        labels_disease = labels_disease.to(device)
        labels_condition = labels_condition.to(device)

        optimizer.zero_grad()
        logits_d, logits_c = model(images)

        loss_d = criterion_d(logits_d, labels_disease)
        loss_c = criterion_c(logits_c, labels_condition)
        loss = loss_d + (LAMBDA_CONDITION * loss_c)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        all_preds_d.append(logits_d.detach())
        all_labels_d.append(labels_disease.detach())
        all_preds_c.append(logits_c.detach())
        all_labels_c.append(labels_condition.detach())

    cat_preds_d = torch.cat(all_preds_d)
    cat_labels_d = torch.cat(all_labels_d)
    cat_preds_c = torch.cat(all_preds_c)
    cat_labels_c = torch.cat(all_labels_c)

    acc_d, f1_d = calculate_metrics_sklearn(cat_preds_d, cat_labels_d)
    acc_c, f1_c = calculate_metrics_sklearn(cat_preds_c, cat_labels_c, ignore_index=-100)

    return total_loss / len(loader), acc_d, f1_d, acc_c, f1_c


def validate(model, loader, criterion_d, criterion_c, device, epoch):
    model.eval()
    total_loss = 0
    all_preds_d, all_labels_d = [], []
    all_preds_c, all_labels_c = [], []

    with torch.no_grad():
        for images, labels_disease, labels_condition in tqdm(loader, desc=f"Ep {epoch} [Val]"):
            images = images.to(device)
            labels_disease = labels_disease.to(device)
            labels_condition = labels_condition.to(device)

            logits_d, logits_c = model(images)

            loss_d = criterion_d(logits_d, labels_disease)
            loss_c = criterion_c(logits_c, labels_condition)
            loss = loss_d + (LAMBDA_CONDITION * loss_c)

            total_loss += loss.item()

            all_preds_d.append(logits_d)
            all_labels_d.append(labels_disease)
            all_preds_c.append(logits_c)
            all_labels_c.append(labels_condition)

    cat_preds_d = torch.cat(all_preds_d)
    cat_labels_d = torch.cat(all_labels_d)
    cat_preds_c = torch.cat(all_preds_c)
    cat_labels_c = torch.cat(all_labels_c)

    acc_d, f1_d = calculate_metrics_sklearn(cat_preds_d, cat_labels_d)
    acc_c, f1_c = calculate_metrics_sklearn(cat_preds_c, cat_labels_c, ignore_index=-100)

    return total_loss / len(loader), acc_d, f1_d, acc_c, f1_c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r"C:\Datasets\OCTDL_Cleaned")
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_SSL_CHECKPOINT)
    parser.add_argument('--save_dir', type=str, default="saved_models")

    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    # CSV-ul e in interiorul folderului de date
    csv_path = os.path.join(args.data_path, "OCTDL_clean_metadata.csv")

    wandb.init(project="Licenta-Classifier-Final", config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load Splits
    print(f"📂 Loading Data from: {args.data_path}")
    train_df, val_df, test_df, disease_map, condition_map = get_data_splits(csv_path)

    # 2. Add Integer Columns
    train_df['label_disease_int'] = train_df['label_disease'].map(disease_map)
    train_df['label_condition_int'] = train_df['label_condition_raw'].map(lambda x: condition_map.get(x, -100))

    # 3. Compute Class Weights
    print("⚖️ Computing Class Weights...")
    weights_d = compute_class_weights(train_df, 'label_disease_int', len(disease_map))
    weights_c = compute_class_weights(train_df, 'label_condition_int', len(condition_map), ignore_index=-100)
    weights_d = weights_d.to(device)
    weights_c = weights_c.to(device)

    # 4. Datasets & Transforms (224px - Esential!)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize on-the-fly
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = OCTDLMultiTaskDataset(train_df, args.data_path, train_transform, disease_map, condition_map)
    val_ds = OCTDLMultiTaskDataset(val_df, args.data_path, val_transform, disease_map, condition_map)

    # Workers=4 pentru viteză
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 5. Model
    print(f"⬇️ Loading SSL Checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"⚠️ WARNING: Checkpoint file NOT FOUND at {args.checkpoint}")
        print("⚠️ Training from scratch (random weights)!")

    model = OCTDLMultiTaskModel(
        checkpoint_path=args.checkpoint,
        num_diseases=len(disease_map),
        num_conditions=len(condition_map),
        freeze_backbone=True  # Înghețăm backbone-ul inițial
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    criterion_disease = nn.CrossEntropyLoss(weight=weights_d)
    criterion_condition = nn.CrossEntropyLoss(weight=weights_c, ignore_index=-100)

    # 6. Training Loop
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc_d, t_f1_d, t_acc_c, t_f1_c = train_one_epoch(
            model, train_loader, optimizer, criterion_disease, criterion_condition, device, epoch
        )

        v_loss, v_acc_d, v_f1_d, v_acc_c, v_f1_c = validate(
            model, val_loader, criterion_disease, criterion_condition, device, epoch
        )

        print(f"\n✨ Ep {epoch}/{args.epochs} Results:")
        print(f"   [Train] Loss: {t_loss:.3f} | Dis F1: {t_f1_d:.3f}")
        print(f"   [Val]   Loss: {v_loss:.3f} | Dis F1: {v_f1_d:.3f} (Best: {best_f1:.3f})")

        wandb.log({
            "epoch": epoch,
            "train_loss": t_loss, "val_loss": v_loss,
            "train_disease_f1": t_f1_d, "val_disease_f1": v_f1_d,
            "train_condition_f1": t_f1_c, "val_condition_f1": v_f1_c
        })

        # Save Best Model
        if v_f1_d > best_f1:
            best_f1 = v_f1_d
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_classifier.pth"))
            print("🏆 Model Saved!")

    wandb.finish()


if __name__ == "__main__":
    main()