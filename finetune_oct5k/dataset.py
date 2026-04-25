"""
OCT5k Dataset — Multi-label biomarker detection (9 classes).

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

BIOMARKERS = [
    "Choroidalfolds", "Fluid", "Geographicatrophy", "Harddrusen",
    "Hyperfluorescentspots", "PRlayerdisruption", "Reticulardrusen",
    "Softdrusen", "SoftdrusenPED",
]

SHORT_NAMES = {
    "Choroidalfolds": "CF", "Fluid": "Fluid", "Geographicatrophy": "GA",
    "Harddrusen": "HD", "Hyperfluorescentspots": "HFS",
    "PRlayerdisruption": "PRL", "Reticulardrusen": "RD",
    "Softdrusen": "SD", "SoftdrusenPED": "SDPED",
}


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


class OCT5kDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform, biomarkers=None):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.biomarkers = biomarkers or BIOMARKERS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            [float(row[bm]) for bm in self.biomarkers],
            dtype=torch.float32,
        )
        return image, labels


def load_oct5k_splits(csv_path, root_dir, test_size=0.2, val_size=0.1,
                      random_state=42, drop_rare=None):
    """
    Load OCT5k CSV and create train/val/test split by patient.
    No predefined split exists — we create one.

    Args:
        drop_rare: minimum positive images to keep a biomarker. Set to e.g. 15
                   to drop Fluid (only 14 images). None = keep all.
    """
    df = pd.read_csv(csv_path)
    print(f"[DATA] Loaded {len(df)} rows from {csv_path}")

    # Optionally filter biomarkers
    active_biomarkers = BIOMARKERS.copy()
    if drop_rare:
        for bm in BIOMARKERS:
            if df[bm].sum() < drop_rare:
                print(f"[DATA] Dropping {bm} (only {int(df[bm].sum())} positive images)")
                active_biomarkers.remove(bm)

    # Split by patient
    patients = df["patient_id"].unique()
    print(f"[DATA] {len(patients)} unique patients")

    # First split: train+val vs test
    train_val_pat, test_pat = train_test_split(
        patients, test_size=test_size, random_state=random_state,
    )

    # Second split: train vs val
    relative_val = val_size / (1 - test_size)
    train_pat, val_pat = train_test_split(
        train_val_pat, test_size=relative_val, random_state=random_state,
    )

    train_df = df[df["patient_id"].isin(train_pat)]
    val_df = df[df["patient_id"].isin(val_pat)]
    test_df = df[df["patient_id"].isin(test_pat)]

    print(f"[DATA] Split: {len(train_df)} train ({len(train_pat)} patients), "
          f"{len(val_df)} val ({len(val_pat)} patients), "
          f"{len(test_df)} test ({len(test_pat)} patients)")

    # Distribution per split
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = {SHORT_NAMES[bm]: int(split_df[bm].sum()) for bm in active_biomarkers}
        print(f"[DATA] {name}: {dist}")

    return train_df, val_df, test_df, active_biomarkers


def compute_pos_weights(train_df, biomarkers):
    """pos_weight = n_neg / n_pos per biomarker."""
    weights = []
    for bm in biomarkers:
        n_pos = train_df[bm].sum()
        n_neg = len(train_df) - n_pos
        w = n_neg / n_pos if n_pos > 0 else 1.0
        weights.append(w)
        print(f"[WEIGHTS] {SHORT_NAMES[bm]}: pos={int(n_pos)}, neg={int(n_neg)}, weight={w:.2f}")
    return torch.tensor(weights, dtype=torch.float32)