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
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

@dataclass
class Sample:
    image_path: str
    label: str


def choose_label_column(df: pd.DataFrame, requested: str) -> str:
    if requested and requested in df.columns:
        return requested
    for c in ["label_disease", "disease", "label_condition_raw", "condition", "label"]:
        if c in df.columns:
            return c
    raise ValueError("No valid label column found. Tried: label_disease, disease, label_condition_raw, condition, label")


def _resolve_rel_path(row: pd.Series, path_col: str) -> str:
    # Prefer explicit path column; fallback to disease/file_name layout.
    if path_col and path_col in row.index and pd.notna(row[path_col]):
        rel = str(row[path_col]).strip()
        if rel:
            return rel

    if "file_name" in row.index and pd.notna(row["file_name"]):
        fname = str(row["file_name"]).strip()
        if "disease" in row.index and pd.notna(row["disease"]):
            return os.path.join(str(row["disease"]).strip(), fname)
        return fname

    raise ValueError("Could not resolve image path. Need path column or file_name column.")


def _stratified_split(df: pd.DataFrame, label_col: str, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Patient-wise split if possible to avoid leakage.
    if "patient_id" in df.columns:
        p = df[["patient_id", label_col]].dropna().copy()
        if not p.empty:
            patient_labels = p.groupby("patient_id")[label_col].agg(lambda s: s.astype(str).mode().iloc[0])
            patient_ids = patient_labels.index.to_numpy()
            patient_y = patient_labels.astype(str).to_numpy()
            try:
                tr_p, te_p = train_test_split(
                    patient_ids,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=patient_y,
                )
                train_df = df[df["patient_id"].isin(tr_p)].copy()
                test_df = df[df["patient_id"].isin(te_p)].copy()
                if len(train_df) > 0 and len(test_df) > 0:
                    return train_df, test_df
            except Exception:
                pass

    y = df[label_col].astype(str)
    try:
        tr_idx, te_idx = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except Exception:
        tr_idx, te_idx = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
    return df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()


def build_samples(csv_path, image_root, split_col, split_name, path_col, label_col, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    label_col = choose_label_column(df, label_col)

    if split_col and split_col in df.columns and split_name:
        df = df[df[split_col].astype(str) == str(split_name)].copy()
    elif split_name in {"train", "test"}:
        train_df, test_df = _stratified_split(df, label_col, test_size=test_size, random_state=random_state)
        df = train_df if split_name == "train" else test_df

    out = []
    missing = 0
    for _, r in df.iterrows():
        try:
            rel = _resolve_rel_path(r, path_col)
            abs_path = os.path.join(image_root, rel)
            if not os.path.isfile(abs_path):
                missing += 1
                continue
            out.append(Sample(
                image_path=abs_path,
                label=str(r[label_col]),
            ))
        except Exception:
            missing += 1
            continue

    if missing:
        print(f"[WARN] Skipped {missing} rows due to unresolved/missing image paths.")
    return out


class OCTDataset(Dataset):
    def __init__(self, samples: List[Sample], img_size=224):
        self.samples = samples
        self.t = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.raw = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(s.image_path).convert("RGB")
        return self.t(img), s.label


def load_model(arch, checkpoint, device):
    model = torch.hub.load("facebookresearch/dinov2", arch)
    ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, dict):
        if "teacher" in ckpt:
            st = ckpt["teacher"]
        elif "model" in ckpt:
            st = ckpt["model"]
        elif "state_dict" in ckpt:
            st = ckpt["state_dict"]
        else:
            st = ckpt
    else:
        st = ckpt

    clean = {}
    for k, v in st.items():
        clean[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(clean, strict=False)
    model.eval().to(device)
    return model

@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    feats = []
    labels = []
    for batch in tqdm(loader, desc="Extracting features"):
        x, y = batch
        x = x.to(device, non_blocking=True)
        f = model(x)
        if isinstance(f, (tuple, list)):
            f = f[0]
        feats.append(f.detach().cpu().numpy())
        labels.append(np.array(y))
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
    parser.add_argument("--label_col", type=str, default="label_disease")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--knn_k", type=int, default=20)

    parser.add_argument("--out_json", type=str, default="analyse_pretrain/frozen_eval_result.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    train_samples = build_samples(
        args.csv, args.image_root, args.split_col, args.train_split, args.path_col, args.label_col,
        test_size=args.test_size, random_state=args.random_state
    )
    test_samples = build_samples(
        args.csv, args.image_root, args.split_col, args.test_split, args.path_col, args.label_col,
        test_size=args.test_size, random_state=args.random_state
    )
    print(f"[INFO] #train={len(train_samples)} #test={len(test_samples)}")

    if len(train_samples) == 0 or len(test_samples) == 0:
        raise RuntimeError("Train/test sample list is empty. Check split/path/label columns and values.")

    train_ds = OCTDataset(train_samples, img_size=args.img_size)
    test_ds = OCTDataset(test_samples, img_size=args.img_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = load_model(args.arch, args.checkpoint, device)

    x_train, y_train = extract_features(model, train_loader, device)
    x_test, y_test = extract_features(model, test_loader, device)

    knn_metrics = eval_knn(x_train, y_train, x_test, y_test, k=args.knn_k)
    lp_metrics = eval_linear_probe(x_train, y_train, x_test, y_test)

    result = {
        "checkpoint": args.checkpoint,
        "arch": args.arch,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
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
