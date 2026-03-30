# ANALYSIS/eval_frozen_features.py
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd


@dataclass
class Sample:
    image_path: str
    label: int


def build_label_mapping(df: pd.DataFrame, label_col: str) -> Dict[str, int]:
    classes = sorted(df[label_col].astype(str).unique().tolist())
    return {c: i for i, c in enumerate(classes)}


def apply_label_mapping(df: pd.DataFrame, label_col: str, label_map: Dict[str, int]) -> pd.DataFrame:
    out = df.copy()
    out[label_col] = out[label_col].astype(str).map(label_map)
    out = out.dropna(subset=[label_col])
    out[label_col] = out[label_col].astype(int)
    return out


def build_samples_from_df(df: pd.DataFrame, image_root: str, path_col: str, label_col: str) -> List[Sample]:
    samples: List[Sample] = []
    missing = 0
    for _, row in df.iterrows():
        rel_path = str(row[path_col])
        img_path = rel_path if os.path.isabs(rel_path) else os.path.join(image_root, rel_path)
        if not os.path.exists(img_path):
            missing += 1
            continue
        samples.append(Sample(image_path=img_path, label=int(row[label_col])))
    if missing > 0:
        print(f"[WARN] Skipped {missing} rows with missing image files.")
    return samples


def build_samples_from_csv(
    csv_path: str,
    image_root: str,
    split_col: str,
    split_name: Optional[str],
    path_col: str,
    label_col: str,
    label_map: Optional[Dict[str, int]] = None,
) -> Tuple[List[Sample], Dict[str, int]]:
    df = pd.read_csv(csv_path)

    required = [path_col, label_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in CSV. Available: {list(df.columns)}")

    if split_name is not None:
        if split_col not in df.columns:
            raise ValueError(f"split_col '{split_col}' not found in CSV. Available: {list(df.columns)}")
        df = df[df[split_col].astype(str) == str(split_name)].copy()

    if label_map is None:
        if np.issubdtype(df[label_col].dtype, np.number):
            label_map = {str(int(v)): int(v) for v in sorted(df[label_col].dropna().unique().tolist())}
            df = df.copy()
            df[label_col] = df[label_col].astype(int)
        else:
            label_map = build_label_mapping(df, label_col)
            df = apply_label_mapping(df, label_col, label_map)
    else:
        if not np.issubdtype(df[label_col].dtype, np.number):
            df = apply_label_mapping(df, label_col, label_map)
        else:
            df = df.copy()
            df[label_col] = df[label_col].astype(int)

    samples = build_samples_from_df(df, image_root, path_col, label_col)
    return samples, label_map


class OCTLabeledDataset(Dataset):
    def __init__(self, samples: List[Sample], img_size: int = 224):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # ImageNet normalization for DINOv2 backbones
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        x = self.transform(img)
        y = s.label
        return x, y


# -----------------------------------------
# 2) LOAD DINOv2 BACKBONE + CHECKPOINT
# -----------------------------------------
def load_dinov2_backbone(arch: str, checkpoint_path: str, device: torch.device):
    """
    arch examples:
      - dinov2_vits14
      - dinov2_vitb14
      - dinov2_vitl14
    """
    # This uses torch.hub official DINOv2 loading.
    model = torch.hub.load("facebookresearch/dinov2", arch)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Try common keys used in official training checkpoints
    if isinstance(ckpt, dict):
        if "teacher" in ckpt:
            state = ckpt["teacher"]
        elif "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    # Remove potential prefixes from DDP wrappers
    clean_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        clean_state[nk] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    print(f"[INFO] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    model.eval().to(device)
    return model


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    feats = []
    labels = []
    for x, y in tqdm(loader, desc="Extracting features"):
        x = x.to(device, non_blocking=True)
        # DINOv2 forward returns global embedding [B, D]
        f = model(x)
        if isinstance(f, (tuple, list)):
            f = f[0]
        feats.append(f.detach().cpu().numpy())
        labels.append(y.numpy())
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    return feats, labels


def eval_knn(x_train, y_train, x_test, y_test, k: int = 20) -> Dict[str, float]:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="cosine")
    clf.fit(x_train_s, y_train)
    pred = clf.predict(x_test_s)
    return {
        "acc": accuracy_score(y_test, pred),
        "bal_acc": balanced_accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
    }


def eval_linear_probe(x_train, y_train, x_test, y_test, max_iter: int = 2000) -> Dict[str, float]:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    clf = LogisticRegression(
        max_iter=max_iter,
        multi_class="auto",
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(x_train_s, y_train)
    pred = clf.predict(x_test_s)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    return {
        "acc": accuracy_score(y_test, pred),
        "bal_acc": balanced_accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "report": report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="dinov2_vits14")
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--split_col", type=str, default="split")
    parser.add_argument("--path_col", type=str, default="image_path")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--test_split", type=str, default="test")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--knn_k", type=int, default=20)

    parser.add_argument("--out_json", type=str, default="analyse_pretrain/frozen_eval_result.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    train_samples, label_map = build_samples_from_csv(
        args.csv,
        args.image_root,
        args.split_col,
        args.train_split,
        args.path_col,
        args.label_col,
        label_map=None,
    )
    test_samples, _ = build_samples_from_csv(
        args.csv,
        args.image_root,
        args.split_col,
        args.test_split,
        args.path_col,
        args.label_col,
        label_map=label_map,
    )
    print(f"[INFO] #train={len(train_samples)} #test={len(test_samples)}")

    if len(train_samples) == 0 or len(test_samples) == 0:
        raise RuntimeError("Train/test sample list is empty. Check split/path/label columns and values.")

    train_ds = OCTLabeledDataset(train_samples, img_size=args.img_size)
    test_ds = OCTLabeledDataset(test_samples, img_size=args.img_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = load_dinov2_backbone(args.arch, args.checkpoint, device)

    x_train, y_train = extract_features(model, train_loader, device)
    x_test, y_test = extract_features(model, test_loader, device)

    knn_metrics = eval_knn(x_train, y_train, x_test, y_test, k=args.knn_k)
    lp_metrics = eval_linear_probe(x_train, y_train, x_test, y_test)

    result = {
        "checkpoint": args.checkpoint,
        "arch": args.arch,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "label_map": label_map,
        "knn": knn_metrics,
        "linear_probe": {
            "acc": lp_metrics["acc"],
            "bal_acc": lp_metrics["bal_acc"],
            "macro_f1": lp_metrics["macro_f1"],
        },
        "linear_probe_report": lp_metrics["report"],
    }

    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("[RESULT] kNN:", knn_metrics)
    print("[RESULT] Linear Probe:", result["linear_probe"])
    print(f"[INFO] Saved: {args.out_json}")


if __name__ == "__main__":
    main()
