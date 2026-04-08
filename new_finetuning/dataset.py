"""
OCTDL Multi-Task Dataset & Splitting

Patient-based stratified splits, class-weight computation,
ignore_index=-100 for missing condition labels.

Compatible with the OCTDL_Cleaned layout:
    root / <disease_folder> / <filename>.jpg
"""

import os
from collections import Counter

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IGNORE_INDEX  = -100

def get_train_transform(img_size: int = 518):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(img_size: int = 518):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class OCTDLMultiTaskDataset(Dataset):
    """
    Returns (image_tensor, disease_label, condition_label) per sample.
    """

    def __init__(self, dataframe, root_dir, transform, disease_map, condition_map):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.disease_map = disease_map
        self.condition_map = condition_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Path: root / disease_folder / filename
        img_path = os.path.join(self.root_dir, row["disease"], row["file_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Disease label — always present
        label_disease = self.disease_map[str(row["label_disease"])]

        # Condition label — may be missing / filtered
        cond_str = str(row["label_condition_raw"])
        label_condition = self.condition_map.get(cond_str, IGNORE_INDEX)

        return image, label_disease, label_condition


def get_data_splits(csv_path, test_size=0.2, val_size=0.1, random_state=42):
    """
    Patient-based stratified split → train / val / test DataFrames.

    Returns:
        train_df, val_df, test_df, disease_map, condition_map
    """

    df = pd.read_csv(csv_path)
    print(f"[DATA] Loaded {len(df)} rows from {csv_path}")

    # Build label maps (sorted for reproducibility)
    unique_diseases = sorted(df["label_disease"].astype(str).unique())
    disease_map = {name: i for i, name in enumerate(unique_diseases)}

    # Only conditions with enough samples (already filtered in OCTDL_Cleaned)
    valid_conds = df.loc[
        df["label_condition_raw"] != "IGNORE", "label_condition_raw"
    ].unique()
    condition_map = {name: i for i, name in enumerate(sorted(valid_conds))}

    print(f"[DATA] Disease classes ({len(disease_map)}): {disease_map}")
    print(f"[DATA] Condition classes ({len(condition_map)}): {condition_map}")

    # Split at patient level, stratify on disease
    patients = df[["patient_id", "label_disease"]].drop_duplicates()
    total_held_out = test_size + val_size

    train_pat, temp_pat = train_test_split(
        patients, test_size=total_held_out,
        random_state=random_state, stratify=patients["label_disease"],
    )
    relative_test = test_size / total_held_out
    val_pat, test_pat = train_test_split(
        temp_pat, test_size=relative_test,
        random_state=random_state, stratify=temp_pat["label_disease"],
    )

    train_df = df[df["patient_id"].isin(train_pat["patient_id"])]
    val_df   = df[df["patient_id"].isin(val_pat["patient_id"])]
    test_df  = df[df["patient_id"].isin(test_pat["patient_id"])]

    print(f"[DATA] Train: {len(train_df)} imgs ({len(train_pat)} patients)")
    print(f"[DATA] Val:   {len(val_df)} imgs ({len(val_pat)} patients)")
    print(f"[DATA] Test:  {len(test_df)} imgs ({len(test_pat)} patients)")

    return train_df, val_df, test_df, disease_map, condition_map


def compute_class_weights(df, label_col, label_map):
    """
    Inverse-frequency weighting: w_i = N / (C * n_i)
    """
    num_classes = len(label_map)
    inv_map = {v: k for k, v in label_map.items()}

    # Count samples per mapped class
    mapped = df[label_col].astype(str).map(label_map)
    counts = mapped.dropna().astype(int).value_counts()
    total = counts.sum()

    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        n_i = counts.get(i, 0)
        weights[i] = (total / (num_classes * n_i)) if n_i > 0 else 0.0

    print(f"[WEIGHTS] {label_col}: {dict(zip([inv_map[i] for i in range(num_classes)], weights.tolist()))}")
    return weights