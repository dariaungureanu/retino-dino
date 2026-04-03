"""
Method — Frozen Feature Evaluation (kNN + Linear Probe)
==========================================================
Purpose:
    Quantitatively measure the quality of features learned by the DINOv2
    backbone, WITHOUT any fine-tuning. The backbone is completely frozen;
    we extract one CLS-token embedding per image, then train simple
    classifiers (kNN and Logistic Regression) on those embeddings.

What this tells you:
    - Are the learned features linearly separable by disease class?
    - How much does domain adaptation improve over ImageNet features?
    - kNN measures local neighborhood structure in feature space
    - Linear probe measures global linear separability

    If linear probe >> kNN, the features have good global structure but
    noisy local neighborhoods. If both are high, features are excellent.

Metrics reported:
    - Accuracy, Balanced Accuracy, Macro-F1
    - Per-class classification report (linear probe)

Usage:
    # Domain-adapted checkpoint
    python analyse_pretrain/method_frozen_eval.py \
        --arch dinov2_vits14 \
        --checkpoint /home/student/Ungureanu_Daria/antrenare_oct_v2/model_final.rank_0.pth \
        --csv /home/student/Ungureanu_Daria/OCTDL_Cleaned/OCTDL_clean_metadata.csv \
        --image_root /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --out_dir results/frozen_eval/domain_adapted

    # Baseline (original ImageNet pretrained — no --checkpoint)
    python analyse_pretrain/method_frozen_eval.py \
        --arch dinov2_vits14 \
        --csv /home/student/Ungureanu_Daria/OCTDL_Cleaned/OCTDL_clean_metadata.csv \
        --image_root /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --out_dir results/frozen_eval/imagenet_baseline
"""

import argparse
import json
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from analyse_shared import (
    load_model, get_device, IMAGENET_MEAN, IMAGENET_STD, DEFAULT_ARCH,
)


# ──────────────────────────────────────────────────────────────
# Dataset (simplified for Method 3 — no raw image needed)
# ──────────────────────────────────────────────────────────────

class FeatureExtractionDataset(Dataset):
    """
    Lightweight dataset for frozen feature extraction.
    Returns only (normalized_tensor, label_string) — no raw image
    since we don't need visualizations here.
    """

    def __init__(self, image_paths: List[str], labels: List[str], img_size: int = 224):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


# ──────────────────────────────────────────────────────────────
# Data loading with patient-aware splitting
# ──────────────────────────────────────────────────────────────

def load_and_split_data(
    csv_path: str,
    image_root: str,
    label_col: str,
    path_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Load CSV metadata and split into train/test sets.
    Uses patient-wise stratified split if patient_id column exists,
    otherwise falls back to image-level stratified split.

    Returns:
        (train_paths, train_labels, test_paths, test_labels)
    """
    df = pd.read_csv(csv_path)
    print(f"[DATA] CSV loaded: {len(df)} rows")

    # Validate columns
    if label_col not in df.columns:
        available = [c for c in df.columns if "label" in c.lower() or "disease" in c.lower()]
        raise ValueError(
            f"Label column '{label_col}' not in CSV. "
            f"Available candidates: {available}"
        )

    # Resolve image paths
    paths, labels = [], []
    missing = 0
    for _, row in df.iterrows():
        fname = str(row[path_col]).strip()
        # OCTDL_Cleaned layout: bare filename → prepend disease folder
        if "/" not in fname and "\\" not in fname and "disease" in df.columns:
            fname = os.path.join(str(row["disease"]), fname)
        full_path = os.path.join(image_root, fname)

        if not os.path.isfile(full_path):
            missing += 1
            continue

        paths.append(full_path)
        labels.append(str(row[label_col]))

    if missing > 0:
        print(f"[DATA] WARNING: {missing} images not found, skipped")
    print(f"[DATA] Resolved {len(paths)} images across {len(set(labels))} classes")

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"[DATA]   {cls}: {cnt} ({cnt/len(labels):.1%})")

    # Split: patient-wise if possible
    if "patient_id" in df.columns:
        print(f"[DATA] Using patient-wise stratified split (no data leakage)")
        # Build patient → label mapping (majority vote)
        valid_df = df.iloc[:len(paths)].copy()  # only rows with valid images
        valid_df["_path"] = paths
        valid_df["_label"] = labels

        patient_labels = (
            valid_df.groupby("patient_id")[label_col]
            .agg(lambda s: s.astype(str).mode().iloc[0])
        )

        try:
            train_patients, test_patients = train_test_split(
                patient_labels.index.to_numpy(),
                test_size=test_size,
                random_state=random_state,
                stratify=patient_labels.values,
            )
            train_mask = valid_df["patient_id"].isin(train_patients)
            test_mask = valid_df["patient_id"].isin(test_patients)

            train_paths = valid_df.loc[train_mask, "_path"].tolist()
            train_labels = valid_df.loc[train_mask, "_label"].tolist()
            test_paths = valid_df.loc[test_mask, "_path"].tolist()
            test_labels = valid_df.loc[test_mask, "_label"].tolist()

            print(f"[DATA] Split: {len(train_paths)} train, {len(test_paths)} test "
                  f"({len(train_patients)} / {len(test_patients)} patients)")
            return train_paths, train_labels, test_paths, test_labels

        except ValueError as e:
            print(f"[DATA] Patient-wise split failed ({e}), falling back to image-level")

    # Fallback: image-level stratified split
    print(f"[DATA] Using image-level stratified split")
    indices = np.arange(len(paths))
    try:
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size,
            random_state=random_state, stratify=labels,
        )
    except ValueError:
        print(f"[WARN] Stratified split failed (likely too few samples in some class), "
              f"using random split")
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state,
        )

    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_paths = [paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"[DATA] Split: {len(train_paths)} train, {len(test_paths)} test")
    return train_paths, train_labels, test_paths, test_labels


# ──────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract CLS-token embeddings from the frozen backbone.
    Returns (features [N, D], labels [N,]) as numpy arrays.
    """
    all_feats = []
    all_labels = []

    for x, y in tqdm(loader, desc="Extracting features"):
        x = x.to(device, non_blocking=True)

        # Explicitly use forward_features to get the CLS token
        feats = model.forward_features(x)

        if isinstance(feats, dict) and "x_norm_clstoken" in feats:
            cls_token = feats["x_norm_clstoken"]  # [B, D]
        elif isinstance(feats, dict) and "x_prenorm" in feats:
            cls_token = feats["x_prenorm"][:, 0, :]  # CLS is token 0
        elif torch.is_tensor(feats):
            cls_token = feats[:, 0, :]
        else:
            raise RuntimeError(f"Unexpected forward_features output: {type(feats)}")

        all_feats.append(cls_token.cpu().numpy())
        all_labels.append(np.array(y))

    features = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"[INFO] Extracted features: shape={features.shape}, "
          f"dtype={features.dtype}")
    return features, labels


# ──────────────────────────────────────────────────────────────
# Classifiers
# ──────────────────────────────────────────────────────────────

def eval_knn(
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    k: int = 20,
) -> Dict:
    """k-Nearest Neighbors on frozen features."""
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    clf = KNeighborsClassifier(
        n_neighbors=k, weights="distance", metric="cosine",
    )
    clf.fit(x_train_s, y_train)
    pred = clf.predict(x_test_s)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "k": k,
    }

    print(f"\n[RESULT] kNN (k={k}):")
    print(f"[RESULT]   Accuracy:          {metrics['accuracy']:.4f}")
    print(f"[RESULT]   Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"[RESULT]   Macro-F1:           {metrics['macro_f1']:.4f}")

    return metrics


def eval_linear_probe(
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    max_iter: int = 2000,
) -> Dict:
    """Logistic Regression linear probe on frozen features."""
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    clf = LogisticRegression(
        max_iter=max_iter,
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(x_train_s, y_train)
    pred = clf.predict(x_test_s)

    report = classification_report(
        y_test, pred, output_dict=True, zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "max_iter": max_iter,
        "per_class_report": report,
    }

    print(f"\n[RESULT] Linear Probe:")
    print(f"[RESULT]   Accuracy:          {metrics['accuracy']:.4f}")
    print(f"[RESULT]   Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"[RESULT]   Macro-F1:           {metrics['macro_f1']:.4f}")
    print(f"\n[RESULT] Per-class report:")
    print(classification_report(y_test, pred, zero_division=0))

    return metrics


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Frozen feature evaluation (kNN + Linear Probe) for DINOv2"
    )

    # Model
    ap.add_argument("--arch", default=DEFAULT_ARCH,
                    help="DINOv2 hub architecture")
    ap.add_argument("--checkpoint", default=None,
                    help="Domain-adapted checkpoint. Omit for ImageNet baseline.")

    # Data
    ap.add_argument("--csv", required=True, help="Path to metadata CSV")
    ap.add_argument("--image_root", required=True, help="Root image directory")
    ap.add_argument("--label_col", default="label_disease",
                    help="CSV column with class labels")
    ap.add_argument("--path_col", default="file_name",
                    help="CSV column with image filenames")

    # Split
    ap.add_argument("--test_size", type=float, default=0.2,
                    help="Fraction for test set (default: 0.2)")
    ap.add_argument("--random_state", type=int, default=42)

    # Processing
    ap.add_argument("--img_size", type=int, default=224,
                    help="Input resolution (224 is standard for feature extraction)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--knn_k", type=int, default=20)

    # Output
    ap.add_argument("--out_dir", default="results/frozen_eval")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = get_device()

    # ── Data ──
    train_paths, train_labels, test_paths, test_labels = load_and_split_data(
        csv_path=args.csv,
        image_root=args.image_root,
        label_col=args.label_col,
        path_col=args.path_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    train_ds = FeatureExtractionDataset(train_paths, train_labels, img_size=args.img_size)
    test_ds = FeatureExtractionDataset(test_paths, test_labels, img_size=args.img_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ──
    model = load_model(args.arch, args.checkpoint, device)

    # ── Extract ──
    x_train, y_train = extract_features(model, train_loader, device)
    x_test, y_test = extract_features(model, test_loader, device)

    # ── Evaluate ──
    knn_metrics = eval_knn(x_train, y_train, x_test, y_test, k=args.knn_k)
    lp_metrics = eval_linear_probe(x_train, y_train, x_test, y_test)

    # ── Save ──
    checkpoint_label = args.checkpoint or "ImageNet baseline (no checkpoint)"
    result = {
        "method": "frozen_feature_evaluation",
        "checkpoint": checkpoint_label,
        "arch": args.arch,
        "img_size": args.img_size,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_classes": len(set(y_test)),
        "classes": sorted(set(y_test)),
        "knn": knn_metrics,
        "linear_probe": {
            "accuracy": lp_metrics["accuracy"],
            "balanced_accuracy": lp_metrics["balanced_accuracy"],
            "macro_f1": lp_metrics["macro_f1"],
        },
        "linear_probe_per_class": lp_metrics["per_class_report"],
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = args.out_json or os.path.join(args.out_dir, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n[INFO] Results saved: {out_json}")



if __name__ == "__main__":
    main()