"""
UMAP of CLS tokens, coloured by class.

Extracts one CLS-token embedding per image, projects all embeddings to 2D
via UMAP, and colours each point by its disease label. Tight, separable
clusters indicate that the backbone has learned class-relevant features
without supervision; overlapping blobs predict downstream confusion.

Usage:
    # Domain-adapted checkpoint
    python analyse_pretrain/method_umap.py \
        --arch dinov2_vits14 \
        --checkpoint /home/student/Ungureanu_Daria/antrenare_oct_v2/model_final.rank_0.pth \
        --csv /home/student/Ungureanu_Daria/OCTDL_Cleaned/OCTDL_clean_metadata.csv \
        --image_root /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --out_dir results/umap/domain_adapted

    # Baseline (no --checkpoint -> ImageNet weights)
    python analyse_pretrain/method_umap.py \
        --arch dinov2_vits14 \
        --csv /home/student/Ungureanu_Daria/OCTDL_Cleaned/OCTDL_clean_metadata.csv \
        --image_root /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --out_dir results/umap/imagenet_baseline
"""

import argparse
import json
import os
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from analyse_shared import (
    load_model, get_device, IMAGENET_MEAN, IMAGENET_STD, DEFAULT_ARCH,
)


# Dataset (same lightweight version as Method 3)

class FeatureExtractionDataset(Dataset):
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


# Data loading

def load_all_samples(
    csv_path: str,
    image_root: str,
    label_col: str,
    path_col: str,
) -> Tuple[List[str], List[str]]:
    """Load every image listed in the CSV (UMAP uses all samples)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    print(f"CSV loaded: {len(df)} rows")

    if label_col not in df.columns:
        available = [c for c in df.columns if "label" in c.lower() or "disease" in c.lower()]
        raise ValueError(f"Label column '{label_col}' not found. Candidates: {available}")

    paths, labels = [], []
    missing = 0
    for _, row in df.iterrows():
        fname = str(row[path_col]).strip()
        if "/" not in fname and "\\" not in fname and "disease" in df.columns:
            fname = os.path.join(str(row["disease"]), fname)
        full_path = os.path.join(image_root, fname)

        if not os.path.isfile(full_path):
            missing += 1
            continue

        paths.append(full_path)
        labels.append(str(row[label_col]))

    if missing > 0:
        print(f"WARNING: {missing} images not found, skipped")

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"{len(paths)} images, {len(unique)} classes:")
    for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"{cls}: {cnt} ({cnt / len(labels):.1%})")

    return paths, labels


# Feature extraction

@torch.no_grad()
def extract_cls_tokens(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract CLS-token embeddings. Returns (features [N, D], labels [N,])."""
    all_feats, all_labels = [], []

    for x, y in tqdm(loader, desc="Extracting CLS tokens"):
        x = x.to(device, non_blocking=True)
        feats = model.forward_features(x)

        if isinstance(feats, dict) and "x_norm_clstoken" in feats:
            cls = feats["x_norm_clstoken"]
        elif isinstance(feats, dict) and "x_prenorm" in feats:
            cls = feats["x_prenorm"][:, 0, :]
        elif torch.is_tensor(feats):
            cls = feats[:, 0, :]
        else:
            raise RuntimeError(f"Unexpected forward_features output: {type(feats)}")

        all_feats.append(cls.cpu().numpy())
        all_labels.append(np.array(y))

    features = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"Extracted: {features.shape[0]} vectors, dim={features.shape[1]}")
    return features, labels


# UMAP projection + visualization

def run_umap(
    features: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Project features to 2D via UMAP."""
    import umap

    print(f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")

    # standardize before UMAP for stable projection
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
        verbose=True,
    )
    embedding = reducer.fit_transform(features_scaled)
    print(f"Done. Output shape: {embedding.shape}")
    return embedding


def save_umap_plot(
    embedding: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    title: str,
    n_neighbors: int,
    min_dist: float,
):
    """
    Create a publication-quality UMAP scatter plot with one color per class.
    Plots minority classes on top so they're not hidden under majority classes.
    """
    unique_classes = sorted(set(labels))
    n_classes = len(unique_classes)

    # tab10 covers up to 10 classes cleanly; tab20 for more.
    if n_classes <= 10:
        cmap = plt.get_cmap("tab10")
    else:
        cmap = plt.get_cmap("tab20")
    class_to_color = {cls: cmap(i / max(n_classes - 1, 1)) for i, cls in enumerate(unique_classes)}

    # Count per class for legend and plotting order
    class_counts = {}
    for cls in unique_classes:
        class_counts[cls] = np.sum(labels == cls)

    # Sort classes: plot majority first (background), minority last (foreground)
    # This ensures rare classes are visible on top
    plot_order = sorted(unique_classes, key=lambda c: -class_counts[c])

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for cls in plot_order:
        mask = labels == cls
        count = class_counts[cls]
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[class_to_color[cls]],
            label=f"{cls} (n={count})",
            s=15,
            alpha=0.6,
            edgecolors="none",
        )

    ax.legend(
        loc="best",
        fontsize=9,
        markerscale=2.0,
        framealpha=0.9,
    )
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_title(title, fontsize=13)

    # Add hyperparameters as subtitle
    ax.text(
        0.02, 0.02,
        f"n_neighbors={n_neighbors}, min_dist={min_dist}, metric=cosine",
        transform=ax.transAxes,
        fontsize=8,
        color="gray",
        verticalalignment="bottom",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# Main

def main():
    ap = argparse.ArgumentParser(
        description="UMAP visualization of DINOv2 CLS tokens by disease class"
    )

    # Model
    ap.add_argument("--arch", default=DEFAULT_ARCH)
    ap.add_argument("--checkpoint", default=None,
                    help="Domain-adapted checkpoint. Omit for ImageNet baseline")

    # Data
    ap.add_argument("--csv", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--label_col", default="label_disease")
    ap.add_argument("--path_col", default="file_name")

    # Processing
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)

    # UMAP hyperparameters
    ap.add_argument("--n_neighbors", type=int, default=15,
                    help="UMAP n_neighbors (15-30 recommended). "
                         "Higher = more global structure preserved.")
    ap.add_argument("--min_dist", type=float, default=0.1,
                    help="UMAP min_dist (0.1-0.2 recommended). "
                         "Lower = tighter clusters.")
    ap.add_argument("--random_state", type=int, default=42)

    # Output
    ap.add_argument("--out_dir", default="results/umap")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = get_device()

    # Data
    paths, labels = load_all_samples(
        args.csv, args.image_root, args.label_col, args.path_col,
    )

    ds = FeatureExtractionDataset(paths, labels, img_size=args.img_size)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    model = load_model(args.arch, args.checkpoint, device)

    # Extract
    features, label_array = extract_cls_tokens(model, dl, device)

    # UMAP
    embedding = run_umap(
        features,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
    )

    # Plot
    checkpoint_name = "Domain-Adapted" if args.checkpoint else "ImageNet Baseline"
    plot_title = f"UMAP - DINOv2 ViT-S/14 CLS Tokens ({checkpoint_name})"

    plot_path = os.path.join(args.out_dir, "umap_by_class.png")
    save_umap_plot(
        embedding=embedding,
        labels=label_array,
        out_path=plot_path,
        title=plot_title,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )

    # Save data for reproducibility
    checkpoint_label = args.checkpoint or "ImageNet baseline (no checkpoint)"
    result = {
        "method": "umap_cls_tokens",
        "checkpoint": checkpoint_label,
        "arch": args.arch,
        "img_size": args.img_size,
        "n_samples": len(label_array),
        "n_classes": len(set(label_array)),
        "classes": sorted(set(label_array.tolist())),
        "class_counts": {
            cls: int(np.sum(label_array == cls))
            for cls in sorted(set(label_array.tolist()))
        },
        "umap_params": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": "cosine",
            "random_state": args.random_state,
        },
        "plot_path": plot_path,
    }

    out_json = args.out_json or os.path.join(args.out_dir, "results.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"JSON log saved: {out_json}")

    # Save raw embeddings (for replotting without re-extracting)
    npz_path = os.path.join(args.out_dir, "umap_data.npz")
    np.savez(
        npz_path,
        embedding=embedding,
        features=features,
        labels=label_array,
    )
    print(f"Raw data saved: {npz_path}")
    print(f"\n[TIP] To replot without re-extracting features, load {npz_path} directly.")


if __name__ == "__main__":
    main()