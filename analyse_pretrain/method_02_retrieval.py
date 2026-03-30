import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


@dataclass
class Sample:
    image_path: str
    label: int
    label_name: str


def _label_mapping(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, Dict[int, str]]:
    if np.issubdtype(df[label_col].dtype, np.number):
        df = df.copy()
        df["label_id"] = df[label_col].astype(int)
        names = {int(i): str(i) for i in sorted(df["label_id"].unique().tolist())}
        return df, names
    classes = sorted(df[label_col].astype(str).unique().tolist())
    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {i: c for c, i in c2i.items()}
    df = df.copy()
    df["label_id"] = df[label_col].astype(str).map(c2i)
    return df, i2c


def build_samples(csv_path: str, image_root: str, split_col: str, split_name: str, path_col: str, label_col: str):
    df = pd.read_csv(csv_path)
    df = df[df[split_col] == split_name].copy()
    df, id_to_name = _label_mapping(df, label_col)

    samples = []
    for _, row in df.iterrows():
        rel = str(row[path_col])
        p = os.path.join(image_root, rel)
        lid = int(row["label_id"])
        samples.append(Sample(image_path=p, label=lid, label_name=id_to_name[lid]))
    return samples, id_to_name


class OCTDataset(Dataset):
    def __init__(self, samples: List[Sample], img_size: int = 224):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = Image.open(s.image_path).convert("RGB")
        return self.transform(x), s.label, s.image_path


def load_model(arch: str, checkpoint: str, device: torch.device):
    model = torch.hub.load("facebookresearch/dinov2", arch)
    ckpt = torch.load(checkpoint, map_location="cpu")
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

    clean = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        clean[nk] = v

    model.load_state_dict(clean, strict=False)
    model.eval().to(device)
    return model


@torch.no_grad()
def extract(model, loader, device):
    feats, labels, paths = [], [], []
    for x, y, p in tqdm(loader, desc="Extracting embeddings"):
        x = x.to(device, non_blocking=True)
        f = model(x)
        if isinstance(f, (tuple, list)):
            f = f[0]
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
        paths.extend(p)
    return np.concatenate(feats), np.concatenate(labels), paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="dinov2_vitb14")
    ap.add_argument("--checkpoint", required=True)

    ap.add_argument("--csv", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--split_col", default="split")
    ap.add_argument("--path_col", default="image_path")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--split", default="test")

    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_queries", type=int, default=200)
    ap.add_argument("--out_json", default="analyse_pretrain/retrieval_result.json")

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples, _ = build_samples(
        args.csv, args.image_root, args.split_col, args.split, args.path_col, args.label_col
    )
    ds = OCTDataset(samples, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = load_model(args.arch, args.checkpoint, device)
    feats, labels, paths = extract(model, dl, device)

    sims = cosine_similarity(feats, feats)
    np.fill_diagonal(sims, -1.0)

    n = len(labels)
    q_idx = np.arange(n)
    if args.max_queries > 0 and args.max_queries < n:
        rng = np.random.default_rng(42)
        q_idx = rng.choice(q_idx, size=args.max_queries, replace=False)

    correct_at_k = 0
    per_query = []
    for i in q_idx:
        nn_idx = np.argsort(-sims[i])[:args.top_k]
        nn_labels = labels[nn_idx]
        hit = int(np.any(nn_labels == labels[i]))
        correct_at_k += hit
        per_query.append({
            "query_path": paths[i],
            "query_label": int(labels[i]),
            "neighbors": [
                {"path": paths[j], "label": int(labels[j]), "sim": float(sims[i, j])}
                for j in nn_idx
            ],
            "hit_at_k": hit,
        })

    recall_at_k = correct_at_k / len(q_idx) if len(q_idx) > 0 else 0.0

    result = {
        "checkpoint": args.checkpoint,
        "arch": args.arch,
        "split": args.split,
        "n_samples": int(n),
        "top_k": int(args.top_k),
        "n_queries": int(len(q_idx)),
        "recall_at_k": float(recall_at_k),
        "queries": per_query,
    }

    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[RESULT] recall@{args.top_k} = {recall_at_k:.4f}")
    print(f"[INFO] Saved: {args.out_json}")


if __name__ == "__main__":
    main()
