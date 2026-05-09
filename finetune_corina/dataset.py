"""
Corina Dataset - Multi-label DME biomarker detection.

4 binary outputs: DME, HF, ND, Healthy (each present/absent).
Uses BCEWithLogitsLoss (sigmoid per output), NOT CrossEntropyLoss.
"""

import os
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BIOMARKERS = ["DME", "HF", "ND", "Healthy"]
NUM_LABELS = len(BIOMARKERS)


def get_train_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_eval_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CorinaDataset(Dataset):
    """
    Multi-label dataset: returns (image, label_vector).
    label_vector is a float tensor of shape [4]: [DME, HF, ND, Healthy]
    Each element is 0.0 or 1.0.
    """
    def __init__(self, dataframe, root_dir, transform):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Multi-label: float tensor [DME, HF, ND, Healthy]
        labels = torch.tensor(
            [float(row[bm]) for bm in BIOMARKERS],
            dtype=torch.float32,
        )
        return image, labels


def load_corina_splits(csv_path, root_dir, val_size=0.1, random_state=42):
    """
    Load Corina CSV and split into train/val/test.
    Train/test is predefined. Val is carved from train (patient-stratified).
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    train_full = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"Predefined: {len(train_full)} train, {len(test_df)} test")

    for split_name, split_df in [("Train", train_full), ("Test", test_df)]:
        dist = {bm: int(split_df[bm].sum()) for bm in BIOMARKERS}
        print(f"{split_name} biomarkers: {dist}")

    # Carve val from train at the patient level.
    train_patients = train_full[["patient_id"]].drop_duplicates()

    # Stratification key: most common biomarker per patient.
    patient_majority = train_full.groupby("patient_id")[BIOMARKERS].mean().idxmax(axis=1)
    train_patients = train_patients.set_index("patient_id")
    train_patients["strat_key"] = patient_majority

    try:
        train_pat, val_pat = train_test_split(
            train_patients.index.to_numpy(),
            test_size=val_size,
            random_state=random_state,
            stratify=train_patients["strat_key"].values,
        )
    except ValueError:
        # Fall back to random split if stratification fails (too few patients per class).
        print("Stratified val split failed, using random split")
        train_pat, val_pat = train_test_split(
            train_patients.index.to_numpy(),
            test_size=val_size,
            random_state=random_state,
        )

    train_df = train_full[train_full["patient_id"].isin(train_pat)]
    val_df = train_full[train_full["patient_id"].isin(val_pat)]

    print(f"After val split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    print(f"Patients: train={len(train_pat)}, val={len(val_pat)}, "
          f"test={test_df['patient_id'].nunique()}")

    return train_df, val_df, test_df


def compute_pos_weights(train_df):
    """
    Compute positive class weights for BCEWithLogitsLoss.
    pos_weight = num_negatives / num_positives per label.
    This upweights rare positive labels.
    """
    weights = []
    for bm in BIOMARKERS:
        n_pos = train_df[bm].sum()
        n_neg = len(train_df) - n_pos
        w = n_neg / n_pos if n_pos > 0 else 1.0
        weights.append(w)
        print(f"{bm}: pos={int(n_pos)}, neg={int(n_neg)}, weight={w:.2f}")
    return torch.tensor(weights, dtype=torch.float32)