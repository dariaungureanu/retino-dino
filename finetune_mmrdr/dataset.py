"""
MMRDR-OCT Dataset - 3-class DME severity grading.

Classes: 0=No DME, 1=NCI-DME, 2=CI-DME.
Train/test split is predefined in the CSV (tr* = train, ts* = test).
Validation is carved from the training set for early stopping.
No preprocessing applied: images are used as-is for comparability with
the paper's RETFound/ResNet/ViT benchmarks.
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

CLASS_NAMES = {0: "No_DME", 1: "NCI_DME", 2: "CI_DME"}


def get_train_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        # Brightness/contrast only; no hue/saturation jitter on grayscale OCT.
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class MMRDRDataset(Dataset):
    """Single-task dataset: returns (image, grade_label) where grade in {0,1,2}."""
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

        label = int(row["grade"])
        return image, label


def load_mmrdr_splits(csv_path, root_dir, val_size=0.1, random_state=42):
    """
    Load MMRDR-OCT CSV and split into train/val/test.

    Train/test is predefined by filename prefix (tr* = train, ts* = test).
    Validation is carved from the training set (stratified on grade).
    """
    df = pd.read_csv(csv_path)
    print(f"loaded {len(df)} rows from {csv_path}")

    # Filename prefix encodes the predefined split.
    df["split"] = df["image"].apply(lambda x: "train" if os.path.basename(x).startswith("tr") else "test")

    train_full = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"predefined split: {len(train_full)} train, {len(test_df)} test")

    for split_name, split_df in [("Train", train_full), ("Test", test_df)]:
        dist = split_df["grade"].value_counts().sort_index()
        print(f"{split_name} distribution: " +
              ", ".join([f"Grade {g}({CLASS_NAMES[g]})={c}" for g, c in dist.items()]))

    train_df, val_df = train_test_split(
        train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=train_full["grade"],
    )

    print(f"after val split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = sum(1 for _, r in split_df.iterrows()
                      if not os.path.isfile(os.path.join(root_dir, r["image"])))
        if missing > 0:
            print(f"{name}: {missing} images not found!")

    num_classes = len(df["grade"].unique())
    return train_df, val_df, test_df, num_classes


def compute_class_weights(df, num_classes):
    """Inverse-frequency weighting: w_i = N / (C * n_i)"""
    counts = df["grade"].value_counts().sort_index()
    total = len(df)
    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        n_i = counts.get(i, 0)
        weights[i] = (total / (num_classes * n_i)) if n_i > 0 else 0.0
    print(f"{dict(zip([CLASS_NAMES[i] for i in range(num_classes)], weights.tolist()))}")
    return weights