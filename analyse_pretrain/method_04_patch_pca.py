import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


@dataclass
class Sample:
    image_path: str
    label: str


def build_samples(csv_path, image_root, split_col, split_name, path_col, label_col):
    df = pd.read_csv(csv_path)
    df = df[df[split_col] == split_name].copy()
    out = []
    for _, r in df.iterrows():
        out.append(Sample(
            image_path=os.path.join(image_root, str(r[path_col])),
            label=str(r[label_col]),
        ))
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
        return self.t(img), self.raw(img), s.image_path, s.label


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
def get_patch_tokens(model, x):
    if hasattr(model, "forward_features"):
        feats = model.forward_features(x)
        # DINOv2 dict style often contains x_norm_patchtokens
        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                return feats["x_norm_patchtokens"]  # [B, N, D]
            if "x_prenorm" in feats:
                # [B, 1+N, D] -> remove cls
                return feats["x_prenorm"][:, 1:, :]
        if torch.is_tensor(feats):
            return feats[:, 1:, :]
    raise RuntimeError("Could not extract patch tokens; check architecture compatibility.")


def save_pca_map(raw_img_t, tokens, out_png, overlay_alpha=0.5):
    # tokens: [N, D], N = h*w
    n, d = tokens.shape
    hw = int(np.sqrt(n))
    pca = PCA(n_components=3)
    x3 = pca.fit_transform(tokens)  # [N, 3]
    x3 = x3.reshape(hw, hw, 3)

    x3 = (x3 - x3.min()) / (x3.max() - x3.min() + 1e-8)

    raw = raw_img_t.permute(1, 2, 0).cpu().numpy()

    # Upsample patch-level PCA map to image resolution for overlay.
    pca_img = Image.fromarray((x3 * 255).astype(np.uint8)).resize(
        (raw.shape[1], raw.shape[0]), resample=Image.BILINEAR
    )
    pca_resized = np.asarray(pca_img).astype(np.float32) / 255.0

    alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
    overlay = (1.0 - alpha) * raw + alpha * pca_resized
    overlay = np.clip(overlay, 0.0, 1.0)

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(raw)
    plt.axis("off")
    plt.title("Input")

    plt.subplot(1, 3, 2)
    plt.imshow(pca_resized)
    plt.axis("off")
    plt.title("Patch PCA (RGB)")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"Overlay (alpha={alpha:.2f})")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


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
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_images", type=int, default=50)
    ap.add_argument("--out_dir", default="analyse_pretrain/patch_pca")
    ap.add_argument("--overlay_alpha", type=float, default=0.5)
    ap.add_argument("--out_json", default="analyse_pretrain/patch_pca_result.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = build_samples(
        args.csv, args.image_root, args.split_col, args.split, args.path_col, args.label_col
    )
    if args.max_images > 0:
        samples = samples[:args.max_images]

    ds = OCTDataset(samples, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = load_model(args.arch, args.checkpoint, device)

    records: List[Dict] = []
    for i, (x, raw, p, y) in enumerate(tqdm(dl, desc="Patch PCA")):
        x = x.to(device)
        toks = get_patch_tokens(model, x)[0].cpu().numpy()
        out_png = os.path.join(args.out_dir, f"patch_pca_{i:04d}.png")
        save_pca_map(raw[0], toks, out_png, overlay_alpha=args.overlay_alpha)

        records.append({
            "image_path": p[0],
            "label": y[0],
            "pca_map_path": out_png,
            "num_tokens": int(toks.shape[0]),
            "dim": int(toks.shape[1]),
            "overlay_alpha": float(np.clip(args.overlay_alpha, 0.0, 1.0)),
        })

    result = {
        "checkpoint": args.checkpoint,
        "arch": args.arch,
        "split": args.split,
        "num_images": len(records),
        "records": records,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[INFO] Saved: {args.out_json}")


if __name__ == "__main__":
    main()
