import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

# Import our custom modules
from dataset import get_data_splits, OCTDLMultiTaskDataset
from model import OCTDLMultiTaskModel

# --- CONFIGURATION ---
DEFAULT_CHECKPOINT = r"C:\Users\daria\PycharmProjects\Licenta_Final\model_final.rank_0.pth"
DEFAULT_DATA_ROOT = r"C:\Datasets\OCTDL_Cleaned"
DEFAULT_DATA_CSV = f"{DEFAULT_DATA_ROOT}/OCTDL_clean_metadata.csv"
DEFAULT_SAVE_DIR = r"C:\Users\daria\PycharmProjects\Licenta_Final\saved_models"

# Hyperparameters
BATCH_SIZE = 4  # Keep 4 for laptop, increase to 64 on Server
LEARNING_RATE = 1e-4
EPOCHS = 1  # Increase on Server (e.g., 20)
LAMBDA_CONDITION = 1.0
WEIGHT_DECAY = 1e-4


def compute_class_weights(df, col_name, num_classes, ignore_index=-100):
    """
    Calculates class weights: N_total / (n_classes * count_per_class)
    Replicates the logic from your Notebook's 'compute_class_weights_from_df'.
    """
    # Get counts
    counts = df[col_name].value_counts()
    total = len(df)

    weights = torch.zeros(num_classes)

    # Iterate 0 to Num_Classes-1
    for i in range(num_classes):
        # In our dataset.py, we mapped classes to integers 0..N
        # We need to find the count for this integer.
        # Note: The dataframe col usually holds STRINGS or MAPPED INTS depending on when we call this.
        # Our get_data_splits returns a DF with raw strings, but let's look at mapped values if possible.
        # For safety, we rely on the fact that get_data_splits keeps 'label_disease' as original strings usually,
        # but let's assume we pass the mapped column if available, or we do logic here.

        # ACTUALLY: The safest way is to count the occurrences in the mapped dataset logic.
        # But to keep it simple and fast using pandas:

        count = counts.get(i, 0)  # Assumes df column is already integers

        if count == 0:
            weights[i] = 0.0
        else:
            weights[i] = total / (num_classes * count)

    return weights


def calculate_metrics_sklearn(logits, labels, ignore_index=-100):
    """
    Computes Accuracy and Macro-F1 using CPU/Numpy (Standard for Analysis).
    """
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    # Filter out ignore_index (for conditions)
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
        images, labels_disease, labels_condition = images.to(device), labels_disease.to(device), labels_condition.to(
            device)

        optimizer.zero_grad()
        logits_d, logits_c = model(images)

        # Loss Calculation
        loss_d = criterion_d(logits_d, labels_disease)
        loss_c = criterion_c(logits_c, labels_condition)

        loss = loss_d + (LAMBDA_CONDITION * loss_c)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Store for metrics
        all_preds_d.append(logits_d.detach())
        all_labels_d.append(labels_disease.detach())
        all_preds_c.append(logits_c.detach())
        all_labels_c.append(labels_condition.detach())

    # Calc metrics for whole epoch
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
            images, labels_disease, labels_condition = images.to(device), labels_disease.to(
                device), labels_condition.to(device)

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
    wandb.init(project="Licenta-MultiTask-Weighted", config={
        "lr": LEARNING_RATE,
        "batch": BATCH_SIZE,
        "lambda": LAMBDA_CONDITION,
        "weighted": True
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)

    # 1. Load Splits
    train_df, val_df, test_df, disease_map, condition_map = get_data_splits(DEFAULT_DATA_CSV)

    # 2. Add Integer Columns for Weight Calculation
    # We need to map the string columns to integers in the DataFrame to calculate weights easily
    train_df['label_disease_int'] = train_df['label_disease'].map(disease_map)
    # For condition, handle IGNORE
    train_df['label_condition_int'] = train_df['label_condition_raw'].map(lambda x: condition_map.get(x, -100))

    # 3. Compute Class Weights
    print("⚖️ Computing Class Weights...")
    weights_d = compute_class_weights(train_df, 'label_disease_int', len(disease_map))
    weights_c = compute_class_weights(train_df, 'label_condition_int', len(condition_map), ignore_index=-100)

    print(f"   Disease Weights: {weights_d}")
    print(f"   Condition Weights: {weights_c}")

    # Move weights to device
    weights_d = weights_d.to(device)
    weights_c = weights_c.to(device)

    # 4. Datasets
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = OCTDLMultiTaskDataset(train_df, DEFAULT_DATA_ROOT, transform, disease_map, condition_map)
    val_ds = OCTDLMultiTaskDataset(val_df, DEFAULT_DATA_ROOT, transform, disease_map, condition_map)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Model
    model = OCTDLMultiTaskModel(
        checkpoint_path=DEFAULT_CHECKPOINT,
        num_diseases=len(disease_map),
        num_conditions=len(condition_map),
        freeze_backbone=True
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 6. Weighted Loss Functions
    criterion_disease = nn.CrossEntropyLoss(weight=weights_d)
    criterion_condition = nn.CrossEntropyLoss(weight=weights_c, ignore_index=-100)

    # 7. Training
    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc_d, t_f1_d, t_acc_c, t_f1_c = train_one_epoch(
            model, train_loader, optimizer, criterion_disease, criterion_condition, device, epoch
        )

        v_loss, v_acc_d, v_f1_d, v_acc_c, v_f1_c = validate(
            model, val_loader, criterion_disease, criterion_condition, device, epoch
        )

        print(f"\n✨ Ep {epoch} Results:")
        print(f"   [Train] Loss: {t_loss:.3f} | Dis F1: {t_f1_d:.3f} | Cond F1: {t_f1_c:.3f}")
        print(f"   [Val]   Loss: {v_loss:.3f} | Dis F1: {v_f1_d:.3f} | Cond F1: {v_f1_c:.3f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": t_loss, "val_loss": v_loss,
            "train_disease_f1": t_f1_d, "val_disease_f1": v_f1_d,
            "train_condition_f1": t_f1_c, "val_condition_f1": v_f1_c
        })

        torch.save(model.state_dict(), os.path.join(DEFAULT_SAVE_DIR, f"epoch_{epoch}.pth"))

    wandb.finish()


if __name__ == "__main__":
    main()