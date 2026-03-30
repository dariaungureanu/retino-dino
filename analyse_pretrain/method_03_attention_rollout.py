import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
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
            label=str(r[label_col])
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


def attention_rollout(model, x):
    hooks = []
    attn_maps = []

    def hook_fn(_m, _i, o):
        if isinstance(o, tuple):
            o = o[1] if len(o) > 1 else o[0]
        attn_maps.append(o.detach())

    for blk in model.blocks:
        if hasattr(blk.attn, "attn_drop"):
            hooks.append(blk.attn.attn_drop.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    if len(attn_maps) == 0:
        return None

    # Fallback for models where hook output is not the attention tensor:
    if attn_maps[0].dim() != 4:
        return None

    # attn: [B, H, T, T]
    attn = [a.mean(dim=1) for a in attn_maps]  # [B, T, T]
    B, T, _ = attn[0].shape
    eye = torch.eye(T, device=attn[0].device).unsqueeze(0).expand(B, T, T)
    joint = eye
    for a in attn:
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True)
        joint = torch.bmm(a, joint)

    # cls token -> patch tokens
    mask = joint[:, 0, 1:]
    return mask


def save_overlay(raw_img_t, mask_1d, out_png, patch_hw):
    img = raw_img_t.permute(1, 2, 0).cpu().numpy()
    m = mask_1d.reshape(patch_hw, patch_hw)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    m = np.uint8(255 * m)
    m = Image.fromarray(m).resize((img.shape[1], img.shape[0]), Image.BILINEAR)
    m = np.asarray(m) / 255.0

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.imshow(m, alpha=0.45, cmap="jet")
    plt.axis("off")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0)
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
    ap.add_argument("--out_dir", default="analyse_pretrain/attention_rollout")
    ap.add_argument("--out_json", default="analyse_pretrain/attention_rollout_result.json")
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
    for i, (x, raw, p, y) in enumerate(tqdm(dl, desc="Attention rollout")):
        x = x.to(device)
        mask = attention_rollout(model, x)
        if mask is None:
            print("[WARN] Could not capture attention maps for this architecture/hook path.")
            break

        # infer patch grid
        tokens = mask.shape[-1]
        patch_hw = int(np.sqrt(tokens))
        out_png = os.path.join(args.out_dir, f"rollout_{i:04d}.png")
        save_overlay(raw[0], mask[0].detach().cpu().numpy(), out_png, patch_hw)

        records.append({
            "image_path": p[0],
            "label": y[0],
            "overlay_path": out_png,
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
