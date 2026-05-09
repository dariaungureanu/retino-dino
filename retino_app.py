"""
Retino-DINO - OCT analysis demo app.

DINOv2 ViT-S/14 (domain-adapted on 40k OCT images via SSL) + per-task heads.
Four fine-tuned checkpoints:
    OCTDL    multitask   (Disease 7-class + Condition 8-class)
    MMRDR    single-task  (DME severity, 3-class)
    Corina   multi-label  (4 biomarkers)
    OCT5k    multi-label  (8 biomarkers)

Run:  streamlit run retino_app.py

  1. Upload (or pick) an OCT eye-scan image.
  2. Choose one of 4 fine-tuned models.
  3. See the model's prediction + a GradCAM heatmap (which pixels mattered).
  4. Optionally compare the domain-adapted (DA) model vs an ImageNet baseline (IN)
  model side-by-side.
  5. Browse pre-generated training/evaluation figures in a "Reports" tab.

"""
import os
import io
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(ROOT, "checkpoints")
SAMPLES_DIR = os.path.join(ROOT, "sample_images")
SSL_BACKBONE = os.path.join(CKPT_DIR, "model_final.rank_0.pth")
DEVICE = torch.device("cpu")
ARCH = "dinov2_vits14"

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224

EVAL_TRANSFORM = transforms.Compose([ # standard image preprocessing
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])

# Loading the backbone
def _build_dinov2():
    return torch.hub.load("facebookresearch/dinov2", ARCH, trust_repo=True)


def _interpolate_pos_embed(state, model):
    if "pos_embed" not in state:
        return
    ckpt_pos = state["pos_embed"]
    model_pos = model.state_dict()["pos_embed"]
    if ckpt_pos.shape == model_pos.shape:
        return
    cls_pos = ckpt_pos[:, :1, :]
    patch_pos = ckpt_pos[:, 1:, :]
    g_ckpt = int(patch_pos.shape[1] ** 0.5)
    g_model = int((model_pos.shape[1] - 1) ** 0.5)
    d = patch_pos.shape[-1]
    patch_pos = patch_pos.reshape(1, g_ckpt, g_ckpt, d).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos.float(), size=(g_model, g_model),
        mode="bicubic", align_corners=False,
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, d)
    state["pos_embed"] = torch.cat([cls_pos, patch_pos], dim=1)


def load_ssl_backbone(weights_path):
    """Load DINOv2 from hub; if weights_path is given, overwrite with the
    FSDP teacher state-dict (keys prefixed with `teacher.backbone.`).

    weights_path=None -> keep hub-default ImageNet weights (used for the IN
    baseline).
    """
    model = _build_dinov2()

    if weights_path is None:
        return model

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"SSL backbone weights not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "teacher" in ckpt:
        st_dict = ckpt["teacher"]
    elif "model" in ckpt:
        st_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        st_dict = ckpt["state_dict"]
    else:
        st_dict = ckpt

    PREFIX = "teacher.backbone."
    clean = {k[len(PREFIX):]: v for k, v in st_dict.items() if k.startswith(PREFIX)}
    if not clean:
        raise RuntimeError(
            f"No keys with prefix '{PREFIX}' in {weights_path}; "
            "this loader expects an FSDP teacher checkpoint."
        )

    _interpolate_pos_embed(clean, model)
    model.load_state_dict(clean, strict=False)
    return model


# Models - heads on top of the backbone
def _make_head(in_dim, hidden, out_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class OCTDLModel(nn.Module): #multi-task learning
    def __init__(self, backbone, num_diseases, num_conditions,
                 head_hidden=256, head_dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.head_disease = _make_head(384, head_hidden, num_diseases, head_dropout)
        self.head_condition = _make_head(384, head_hidden, num_conditions, head_dropout)

    def _features(self, x):
        f = self.backbone(x)
        if isinstance(f, dict):
            f = f["x_norm_clstoken"]
        elif isinstance(f, tuple):
            f = f[0]
        return f

    def forward(self, x):
        f = self._features(x)
        return self.head_disease(f), self.head_condition(f)


class SingleHeadModel(nn.Module): #For MMRDR (3 classes), Corina (4 biomarkers), OCT5k (8 biomarkers)
    def __init__(self, backbone, num_outputs, head_hidden=256, head_dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.head = _make_head(384, head_hidden, num_outputs, head_dropout)

    def forward(self, x):
        f = self.backbone(x)
        if isinstance(f, dict):
            f = f["x_norm_clstoken"]
        elif isinstance(f, tuple):
            f = f[0]
        return self.head(f)


# GradCAM helpers
def vit_reshape_transform(tensor):
    """Drop CLS token and reshape (B, 1+N, D) -> (B, D, H, W) for ViT."""
    patches = tensor[:, 1:, :]
    g = int(round(patches.size(1) ** 0.5))
    out = patches.reshape(tensor.size(0), g, g, tensor.size(2))
    return out.permute(0, 3, 1, 2)


class _HeadSelector(nn.Module):
    """Pick one tensor from a multi-output forward (used for OCTDL)."""
    def __init__(self, model, head_index):
        super().__init__()
        self.model = model
        self.head_index = head_index

    def forward(self, x):
        return self.model(x)[self.head_index]


# The TASKS registry
OCT5K_LABELS = ["Choroidalfolds", "Geographicatrophy", "Harddrusen",
                "Hyperfluorescentspots", "PRlayerdisruption",
                "Reticulardrusen", "Softdrusen", "SoftdrusenPED"]

CORINA_LABELS = ["DME", "HF", "ND", "Healthy"]
MMRDR_CLASSES = ["No_DME", "NCI_DME", "CI_DME"]
OCTDL_DISEASE = ["AMD", "DME", "ERM", "NO", "RAO", "RVO", "VID"]
OCTDL_CONDITION = ["DRIL", "ERM", "ME", "MH", "MNV", "MNV_suspected", "NO", "drusen"]


def _path(*parts):
    return os.path.join(ROOT, *parts)


TASKS = {
    "OCTDL": {
        "label": "OCTDL  ·  Disease + Condition (multi-task)",
        "type": "octdl",
        "ckpt_da": "octdl_da.pth",
        "ckpt_in": "octdl_in.pth",
        "samples_subdir": "octdl",
        "disease": OCTDL_DISEASE,
        "condition": OCTDL_CONDITION,
        "gallery_dirs": [
            _path("finetune_octdl", "results", "confusion_matrices", "run_C_unfreeze2"),
            _path("finetune_octdl", "results", "explainability", "run_finetuning_C"),
            _path("finetune_octdl", "results", "disease_vs_condition"),
            _path("finetune_octdl", "results", "patient_level"),
            _path("finetune_octdl", "results", "data_efficiency"),
        ],
    },
    "MMRDR": {
        "label": "MMRDR  ·  DME Severity (3-class)",
        "type": "single",
        "ckpt_da": "mmrdr_da.pth",
        "ckpt_in": "mmrd_in.pth",
        "samples_subdir": "mmrdr",
        "classes": MMRDR_CLASSES,
        "gallery_dirs": [
            _path("finetune_mmrdr", "results", "confusion_matrices"),
            _path("finetune_mmrdr", "results", "mmrdr"),
        ],
    },
    "Corina": {
        "label": "Corina  ·  DME Biomarkers (multi-label, 4)",
        "type": "multilabel",
        "ckpt_da": "corina_da.pth",
        "ckpt_in": "corina_in.pth",
        "samples_subdir": "corina",
        "biomarkers": CORINA_LABELS,
        "gallery_dirs": [
            _path("finetune_corina", "results"),
            _path("finetune_corina", "results", "corina"),
        ],
    },
    "OCT5k": {
        "label": "OCT5k  ·  Biomarkers (multi-label, 8)",
        "type": "multilabel",
        "ckpt_da": "oct5k_da.pth",
        "ckpt_in": "oct5k_in.pth",
        "samples_subdir": "oct5k",
        "biomarkers": OCT5K_LABELS,
        "gallery_dirs": [
            _path("finetune_oct5k", "results"),
        ],
    },
}


# Cached model loading
@st.cache_resource(show_spinner=False)
def get_model(task_key: str, variant: str):
    """Return (model, labels). `labels` is derived from the saved maps in
    the checkpoint and falls back to the spec lists when absent."""
    spec = TASKS[task_key]
    ckpt_name = spec["ckpt_da"] if variant == "da" else spec["ckpt_in"]
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    backbone_weights = SSL_BACKBONE if variant == "da" else None
    backbone = load_ssl_backbone(backbone_weights)

    label_sources = {}
    if spec["type"] == "octdl":
        d_src = "ckpt.disease_map" if ckpt.get("disease_map") else "spec.disease"
        c_src = "ckpt.condition_map" if ckpt.get("condition_map") else "spec.condition"
        d_map = ckpt.get("disease_map") or {n: i for i, n in enumerate(spec["disease"])}
        c_map = ckpt.get("condition_map") or {n: i for i, n in enumerate(spec["condition"])}
        diseases   = [n for n, _ in sorted(d_map.items(), key=lambda kv: kv[1])]
        conditions = [n for n, _ in sorted(c_map.items(), key=lambda kv: kv[1])]
        labels = {"disease": diseases, "condition": conditions}
        label_sources = {"disease": d_src, "condition": c_src}
        model = OCTDLModel(backbone, len(diseases), len(conditions))
    elif spec["type"] == "single":
        n_src = "ckpt.num_classes" if "num_classes" in ckpt else "spec.classes (count)"
        n = ckpt.get("num_classes", len(spec["classes"]))
        if "classes" in ckpt and ckpt["classes"]:
            names = list(ckpt["classes"])[:n]
            names_src = "ckpt.classes"
        else:
            names = list(spec["classes"][:n])
            names_src = "spec.classes"
        labels = {"classes": names}
        label_sources = {"count": n_src, "names": names_src}
        model = SingleHeadModel(backbone, n)
    else:
        n_src = "ckpt.num_labels" if "num_labels" in ckpt else "spec.biomarkers (count)"
        n = ckpt.get("num_labels", len(spec["biomarkers"]))
        if "biomarkers" in ckpt and ckpt["biomarkers"]:
            names = list(ckpt["biomarkers"])[:n]
            names_src = "ckpt.biomarkers"
        else:
            names = list(spec["biomarkers"][:n])
            names_src = "spec.biomarkers"
        labels = {"biomarkers": names}
        label_sources = {"count": n_src, "names": names_src}
        model = SingleHeadModel(backbone, n)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)

    head_missing = [k for k in missing if "head" in k]
    backbone_missing = [k for k in missing if "head" not in k]
    head_unexpected = [k for k in unexpected if "head" in k]

    variant_long = "domain-adapted (DA)" if variant == "da" else "ImageNet baseline (IN)"
    print(f"\n=== {task_key}/{variant} [{variant_long}] checkpoint diagnostic ===",
          file=sys.stderr)
    print(f"  ckpt file:             {ckpt_name}", file=sys.stderr)
    print(f"  backbone init:         "
          f"{'SSL teacher (model_final.rank_0.pth)' if variant == 'da' else 'hub default (ImageNet)'}",
          file=sys.stderr)
    print(f"  label sources:         {label_sources}", file=sys.stderr)
    print(f"  ckpt top-level keys:   {list(ckpt.keys())}", file=sys.stderr)
    print(f"  ckpt epoch / best:     "
          f"{ckpt.get('epoch', '?')} / "
          f"{ckpt.get('best_val_f1', ckpt.get('best_metric', ckpt.get('val_f1', ckpt.get('val_disease_f1', ckpt.get('val_f1_macro', '?')))))}",
          file=sys.stderr)
    print(f"  state-dict size:       {len(ckpt['model_state_dict'])}", file=sys.stderr)
    print(f"  head_missing:          {head_missing}", file=sys.stderr)
    print(f"  head_unexpected:       {head_unexpected}", file=sys.stderr)
    print(f"  backbone_missing(N):   {len(backbone_missing)} "
          f"(e.g. {backbone_missing[:3]})", file=sys.stderr)

    if hasattr(model, "head_disease"):
        w = model.head_disease[0].weight
        print(f"  head_disease[0].weight: "
              f"mean={w.mean().item():+.4f} std={w.std().item():.4f}",
              file=sys.stderr)
    elif hasattr(model, "head"):
        w = model.head[0].weight
        print(f"  head[0].weight:         "
              f"mean={w.mean().item():+.4f} std={w.std().item():.4f}",
              file=sys.stderr)

    if head_missing or head_unexpected:
        st.error(
            f"Head weights did not load cleanly for **{task_key} / {variant}**.\n\n"
            f"- missing head keys: `{head_missing}`\n"
            f"- unexpected head keys: `{head_unexpected}`\n\n"
            f"This usually means the saved state-dict has a different prefix or "
            f"layer naming. The model below will be partly randomly initialised - "
            f"predictions are not trustworthy."
        )

    model.to(DEVICE).eval()

    for p in model.parameters():
        p.requires_grad_(True)
    return model, labels


# Inference helpers
def to_tensor(image: Image.Image) -> torch.Tensor:
    return EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)


def denormalize(t: torch.Tensor) -> np.ndarray:
    img = t.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = img * np.array(IMG_STD) + np.array(IMG_MEAN)
    return np.clip(img, 0.0, 1.0)


def predict(model, x, task_type):
    """Forward pass + softmax (octdl/single) or sigmoid (multilabel)."""
    with torch.no_grad():
        out = model(x)
    if task_type == "octdl":
        prob_d = F.softmax(out[0], dim=1)[0].cpu().numpy()
        prob_c = F.softmax(out[1], dim=1)[0].cpu().numpy()
        return {"disease": prob_d, "condition": prob_c}
    if task_type == "single":
        return {"probs": F.softmax(out, dim=1)[0].cpu().numpy()}
    return {"probs": torch.sigmoid(out)[0].cpu().numpy()}


def gradcam(model, x, task_type, target_kind, target_idx):
    """Run GradCAM on the last ViT block. For OCTDL the multi-output forward
    is wrapped in _HeadSelector so GradCAM sees a single tensor."""
    target_layers = [model.backbone.blocks[-1].norm1]

    if task_type == "octdl":
        head_idx = 0 if target_kind == "disease" else 1
        wrapped = _HeadSelector(model, head_idx)
        cam = GradCAM(model=wrapped, target_layers=target_layers,
                      reshape_transform=vit_reshape_transform)
    else:
        cam = GradCAM(model=model, target_layers=target_layers,
                      reshape_transform=vit_reshape_transform)

    heatmap = cam(input_tensor=x,
                  targets=[ClassifierOutputTarget(int(target_idx))])[0]
    rgb = denormalize(x)
    return show_cam_on_image(rgb, heatmap, use_rgb=True)


# UI helpers
_BAR_LEFT   = 0.34
_BAR_RIGHT  = 0.96
_BAR_BOTTOM = 0.18
_BAR_TOP    = 0.86
_BAR_FIG_W  = 4.8
_BAR_ROW_H  = 0.38
_BAR_PAD_H  = 0.95

GRADCAM_HERO_W = 360
GRADCAM_GRID_W = 280


def _truncate(label, n=14):
    return label if len(label) <= n else label[: n - 1] + "..."


def _bar_chart(probs, classes, predicted_idx, title=""):
    """Horizontal softmax bar chart; the predicted bar is highlighted red."""
    fig, ax = plt.subplots(
        figsize=(_BAR_FIG_W, _BAR_ROW_H * len(classes) + _BAR_PAD_H)
    )
    colors = ["#1f77b4"] * len(classes)
    colors[predicted_idx] = "#d62728"
    y = np.arange(len(classes))
    ax.barh(y, probs * 100.0, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([_truncate(c) for c in classes], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("probability (%)", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    for i, p in enumerate(probs * 100.0):
        ax.text(p + 1.5, i, f"{p:.1f}", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=_BAR_LEFT, right=_BAR_RIGHT,
                        bottom=_BAR_BOTTOM, top=_BAR_TOP)
    return fig


def _multilabel_panel(probs, labels, threshold=0.5, title=""):
    """Horizontal sigmoid bar chart; bars >= threshold are green, others grey,
    with a dashed vertical line at the threshold."""
    fig, ax = plt.subplots(
        figsize=(_BAR_FIG_W, _BAR_ROW_H * len(labels) + _BAR_PAD_H)
    )
    colors = ["#2ca02c" if p >= threshold else "#9aa0a6" for p in probs]
    y = np.arange(len(labels))
    ax.barh(y, probs * 100.0, color=colors, edgecolor="none")
    ax.axvline(threshold * 100, color="#d62728", linestyle="--",
               linewidth=1, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([_truncate(l) for l in labels], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("sigmoid probability (%)", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    for i, p in enumerate(probs * 100.0):
        ax.text(p + 1.5, i, f"{p:.1f}", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=_BAR_LEFT, right=_BAR_RIGHT,
                        bottom=_BAR_BOTTOM, top=_BAR_TOP)
    return fig


def _confidence_caption(container, max_conf, n_classes):
    # For an in-distribution image, a well-trained K-class model should not sit
    # near 1/K. Flag suspiciously diffuse softmax outputs.
    container.caption(f"Top confidence: {max_conf*100:.1f}%")
    if max_conf < 0.40:
        container.caption(
            f"Top confidence only {max_conf*100:.0f}% on a {n_classes}-class "
            "head - head may be undertrained or input out-of-distribution."
        )


def render_predictions(spec, labels, preds, container, header,
                       side_by_side=True, threshold=0.5):
    """Dispatch on spec["type"] and render the appropriate predictions panel.

    `side_by_side` is OCTDL-only: False stacks disease and condition charts
    so the panel fits inside an outer column in compare mode (Streamlit caps
    column nesting).
    """
    if header:
        container.markdown(f"#### {header}")
    if spec["type"] == "octdl":
        diseases = labels["disease"]
        conditions = labels["condition"]
        d_idx = int(np.argmax(preds["disease"]))
        c_idx = int(np.argmax(preds["condition"]))
        d_conf = float(preds["disease"][d_idx])
        c_conf = float(preds["condition"][c_idx])
        container.markdown(
            f"**Disease:** `{diseases[d_idx]}` ({d_conf*100:.1f}%)"
        )
        container.markdown(
            f"**Condition:** `{conditions[c_idx]}` ({c_conf*100:.1f}%)"
        )
        if side_by_side:
            col_d, col_c = container.columns(2)
            col_d.pyplot(_bar_chart(preds["disease"], diseases, d_idx,
                                    "Disease head"), clear_figure=True)
            _confidence_caption(col_d, d_conf, len(diseases))
            col_c.pyplot(_bar_chart(preds["condition"], conditions, c_idx,
                                    "Condition head"), clear_figure=True)
            _confidence_caption(col_c, c_conf, len(conditions))
        else:
            container.pyplot(_bar_chart(preds["disease"], diseases, d_idx,
                                        "Disease head"), clear_figure=True)
            _confidence_caption(container, d_conf, len(diseases))
            container.pyplot(_bar_chart(preds["condition"], conditions, c_idx,
                                        "Condition head"), clear_figure=True)
            _confidence_caption(container, c_conf, len(conditions))
    elif spec["type"] == "single":
        classes = labels["classes"]
        idx = int(np.argmax(preds["probs"]))
        conf = float(preds["probs"][idx])
        container.markdown(
            f"**Prediction:** `{classes[idx]}` ({conf*100:.1f}%)"
        )
        container.pyplot(_bar_chart(preds["probs"], classes, idx),
                         clear_figure=True)
        _confidence_caption(container, conf, len(classes))
    else:
        biomarkers = labels["biomarkers"]
        active = [biomarkers[i] for i, p in enumerate(preds["probs"]) if p >= threshold]
        container.markdown(
            f"**Active biomarkers (>={threshold:.2f}):** "
            + (", ".join(f"`{a}`" for a in active) if active else "_none_")
        )
        container.pyplot(_multilabel_panel(preds["probs"], biomarkers,
                                           threshold=threshold), clear_figure=True)


def _select_grid_indices(probs, threshold, fallback_top_k=3):
    """Pick which biomarkers to GradCAM. Active set if any cross threshold,
    else top-K by probability so the grid is never empty."""
    active = [i for i, p in enumerate(probs) if p >= threshold]
    if active:
        return active, False
    k = min(fallback_top_k, len(probs))
    top = list(np.argsort(probs)[::-1][:k])
    return [int(i) for i in top], True


def render_multilabel_gradcam_grid(model, x, biomarkers, probs, threshold,
                                   container, n_cols=None):
    """One GradCAM panel per active biomarker (or top-3 fallback). Layout is
    2 columns for Corina (4 labels) and 4 for OCT5k (8 labels)."""
    if n_cols is None:
        n_cols = 4 if len(biomarkers) >= 8 else 2

    idxs, fell_back = _select_grid_indices(probs, threshold)
    if fell_back:
        container.caption(
            f"No biomarkers above threshold {threshold:.2f}; "
            f"showing top {len(idxs)} by probability instead."
        )

    rows = [idxs[i:i + n_cols] for i in range(0, len(idxs), n_cols)]
    for row in rows:
        cols = container.columns(n_cols)
        for j, bio_idx in enumerate(row):
            cam_img = gradcam(model, x, "multilabel", None, bio_idx)
            cols[j].image(
                cam_img,
                caption=f"{biomarkers[bio_idx]} · {probs[bio_idx]*100:.1f}%",
                width=GRADCAM_GRID_W,
            )


def _resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
    """Resize a PIL image to a fixed display height, preserving aspect.
    Used so the original and GradCAM panels line up vertically even when the
    user uploads a landscape OCT scan."""
    if img.height == 0:
        return img
    w = max(1, int(round(img.width * target_h / img.height)))
    return img.resize((w, target_h), Image.LANCZOS)


# Sample images
OCT5K_ABBREV = {
    "CF":    "Choroidalfolds",
    "GA":    "Geographicatrophy",
    "HD":    "Harddrusen",
    "HFS":   "Hyperfluorescentspots",
    "PRL":   "PRlayerdisruption",
    "RD":    "Reticulardrusen",
    "SD":    "Softdrusen",
    "SDPED": "SoftdrusenPED",
}


@st.cache_data(show_spinner=False)
def list_samples(task_key: str):
    """Return [(display_name, full_path), ...] for the task's sample folder."""
    spec = TASKS[task_key]
    sub = spec.get("samples_subdir")
    if not sub:
        return []
    folder = os.path.join(SAMPLES_DIR, sub)
    if not os.path.isdir(folder):
        return []
    out = []
    for n in sorted(os.listdir(folder)):
        if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            out.append((n, os.path.join(folder, n)))
    return out


def parse_sample_label(task_key: str, filename: str) -> str:
    """Extract the expected label encoded in a sample filename."""
    base = os.path.splitext(filename)[0]
    parts = base.split("_")

    if task_key == "OCTDL":
        if parts[0] == "COND" and len(parts) >= 2:
            cond = parts[1]
            if cond == "MNV" and len(parts) >= 3 and parts[2] == "suspected":
                cond = "MNV_suspected"
            return f"condition · {cond}"
        return f"disease · {parts[0]}"

    if task_key == "MMRDR":
        for cls in ("CI_DME", "NCI_DME", "NoDME"):
            if base.startswith(cls):
                return cls.replace("NoDME", "No_DME")
        return "?"

    if task_key == "Corina":
        present = []
        for p in parts:
            if p in CORINA_LABELS:
                present.append(p)
            else:
                break
        return ", ".join(present) if present else "?"

    if task_key == "OCT5k":
        prefix = base.split("_", 1)[0]
        names = []
        for token in prefix.split("+"):
            names.append(OCT5K_ABBREV.get(token, token))
        return ", ".join(names)

    return "?"


def _gallery_mtimes(spec):
    """Tuple of dir mtimes used as the cache key for gallery_files"""
    return tuple(os.path.getmtime(d)
                 for d in spec["gallery_dirs"] if os.path.isdir(d))


@st.cache_data(show_spinner=False)
def gallery_files(task_key, mtimes):
    """Collect PNG/JPGs from the task's results dirs (recursive). Cached on
    (task_key, mtimes) so directory walks happen only when files change."""
    spec = TASKS[task_key]
    files = []
    for d in spec["gallery_dirs"]:
        if not os.path.isdir(d):
            continue
        for root, _, names in os.walk(d):
            for n in sorted(names):
                if n.lower().endswith((".png", ".jpg", ".jpeg")):
                    full = os.path.join(root, n)
                    rel = os.path.relpath(full, ROOT).replace(os.sep, "/")
                    files.append((rel, full))
    seen = set()
    out = []
    for rel, full in files:
        if full in seen:
            continue
        seen.add(full)
        out.append((rel, full))
    return out


REPORT_CATEGORIES = [
    ("All",                  lambda r: True),
    ("Confusion matrices",   lambda r: "confusion" in r.lower()),
    ("GradCAM",              lambda r: "explainability" in r.lower()
                                       or "gradcam" in r.lower()
                                       or "grad_cam" in r.lower()),
    ("Latent space (t-SNE)", lambda r: "tsne" in r.lower()
                                       or "t-sne" in r.lower()
                                       or "latent" in r.lower()),
    ("Disease vs Condition", lambda r: "disease_vs_condition" in r.lower()),
    ("Patient-level",        lambda r: "patient_level" in r.lower()
                                       or "patient-level" in r.lower()),
    ("Data efficiency",      lambda r: "data_efficiency" in r.lower()
                                       or "efficiency" in r.lower()),
]

def _md_table(header, rows):
    head = "| " + " | ".join(header) + " |"
    align = "|" + "|".join(["---:" if i > 0 else "---" for i in range(len(header))]) + "|"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join([head, align, body])


PRETRAIN_FROZEN = {
    "header": ["Checkpoint", "kNN Acc", "kNN Bal-Acc", "kNN F1",
               "LP Acc", "LP Bal-Acc", "LP F1"],
    "rows": [
        ("ImageNet ViT-S/14 (baseline)",      "0.739", "0.382", "0.419",
                                              "0.804", "0.663", "0.667"),
        ("ViT-S/14 Epoch 5 (6,250 iter)",     "0.911", "0.755", "0.774",
                                              "0.896", "0.764", "0.738"),
        ("ViT-S/14 Epoch 15 (18,750 iter)",   "0.931", "0.796", "0.812",
                                              "0.936", "0.810", "0.820"),
        ("ViT-S/14 Final (37,500 iter)",      "0.938", "0.804", "0.824",
                                              "0.938", "0.835", "0.841"),
        ("Old manual ViT-B/14 (before using dino repo)",     "0.794", "0.498", "0.558",
                                              "0.888", "0.748", "0.742"),
    ],
}

OCTDL_RUNS = {
    "header": ["Run", "Strategy", "Disease F1", "Cond F1", "Best epoch"],
    "rows": [
        ("A", "Frozen (0 blocks), lr_heads 1e-3",   "0.808", "0.792", "5"),
        ("B", "Unfreeze 1 block (block 11 + norm)", "0.833", "0.804", "4"),
        ("C", "Unfreeze 2 blocks (10-11 + norm)",   "**0.845**", "**0.820**", "4"),
        ("D", "Run C + RandomHorizontalFlip",       "0.841", "0.801", "4"),
        ("E", "Run C config, ImageNet ViT-S/14 (no DA)", "0.800", "0.741", "18"),
        ("Before using dino repo", "ViT-Large partial unfreeze, old SSL", "0.749", "0.704", "15-23"),
    ],
}

OCTDL_PATIENT = {
    "header": ["Metric", "Image-level", "Patient (majority vote)", "Patient (avg-probs)"],
    "rows": [
        ("Disease Acc",        "93.5 %",  "90.9 %",  "91.5 %"),
        ("Disease Macro-F1",   "0.8446",  "0.8285",  "0.8344"),
        ("Condition Acc",      "84.2 %",  "83.4 %",  "—"),
        ("Condition Macro-F1", "0.8202",  "0.8142",  "—"),
    ],
}

OCTDL_DATAEFF = {
    "header": ["Fraction", "N_train", "DA Disease F1", "IN Disease F1", "Δ",
               "DA Cond F1", "IN Cond F1", "Δ"],
    "rows": [
        ("33 %",  "459",   "0.820", "0.674", "+0.146", "0.734", "0.634", "+0.100"),
        ("66 %",  "947",   "0.850", "0.784", "+0.066", "0.802", "0.652", "+0.150"),
        ("100 %", "1,410", "0.837", "0.797", "+0.040", "0.807", "0.696", "+0.111"),
    ],
}

MMRDR_COMPARE = {
    "header": ["Method", "Pretraining", "Aug", "Acc", "κ", "Macro F1", "F1 NCI-DME"],
    "rows": [
        ("ViT-Base (paper)",      "ImageNet", "Yes", "0.883", "0.773", "0.700", "0.280"),
        ("ResNet-50 (paper)",     "ImageNet", "Yes", "0.890", "0.791", "0.701", "0.259"),
        ("RETFound (paper)",      "SSL 1.6 M", "Yes", "0.897", "0.803", "0.759", "0.436"),
        ("My ImageNet (Run E)",   "ImageNet", "No",  "0.845", "0.722", "0.689", "0.28"),
        ("My ImageNet + aug",     "ImageNet", "Yes", "0.769", "0.625", "0.662", "0.29"),
        ("**My DA (40 k SSL)**",  "SSL 40 k", "No",  "0.875", "0.770", "0.760", "0.47"),
        ("**My DA + aug**",       "SSL 40 k", "Yes", "**0.890**", "**0.798**", "**0.775**", "**0.49**"),
    ],
}

CORINA_PER_BIO = {
    "header": ["Biomarker", "F1", "AUC", "Acc"],
    "rows": [
        ("DME",     "0.909", "0.980", "88.7 %"),
        ("HF",      "0.922", "0.942", "89.0 %"),
        ("ND",      "0.813", "0.964", "92.6 %"),
        ("Healthy", "0.917", "0.993", "95.7 %"),
        ("**Macro**", "**0.890**", "**0.970**", "—"),
    ],
}

CORINA_VS = {
    "header": ["Variant", "Macro F1", "Macro AUC", "Exact Match"],
    "rows": [
        ("My DA (22 M, SSL 40 k)",        "0.890", "0.970", "74.8 %"),
        ("My ImageNet (22 M)",            "0.707", "0.930", "49.1 %"),
        ("ConvNeXt-base reference (87 M)", "≈0.86", "—",     "—"),
    ],
}

OCT5K_PER_BIO = {
    "header": ["Biomarker (n_pos)", "DA F1", "IN F1", "Δ F1",
               "DA AUC", "IN AUC", "Δ AUC"],
    "rows": [
        ("CF (11)",   "0.50", "0.35", "+0.15", "0.83", "0.76", "+0.07"),
        ("GA (19)",   "0.84", "0.73", "+0.11", "0.98", "0.90", "+0.08"),
        ("HD (43)",   "0.79", "0.61", "+0.18", "0.85", "0.77", "+0.08"),
        ("HFS (1)",   "0.08", "0.00", "—",     "0.87", "0.10", "—"),
        ("PRL (42)",  "0.84", "0.78", "+0.06", "0.95", "0.93", "+0.02"),
        ("RD (16)",   "0.39", "0.20", "+0.19", "0.69", "0.53", "+0.16"),
        ("SD (69)",   "0.85", "0.86", "−0.01", "0.88", "0.87", "+0.01"),
        ("SDPED (21)", "0.24", "0.23", "+0.01", "0.70", "0.57", "+0.13"),
        ("**Macro**", "**0.567**", "**0.469**", "**+0.098**",
                      "**0.843**", "**0.678**", "**+0.165**"),
        ("Exact Match", "30.8 %", "13.1 %", "+17.7 %", "—", "—", "—"),
    ],
}


# The main Streamlit UI
def main():
    #Page setup
    st.set_page_config(
        page_title="Retino-DINO",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        h1 { letter-spacing: -0.5px; }
        .small-caption { color: #6b7280; font-size: 0.85rem; }
        .stRadio label p { font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Retino-DINO")
    st.markdown(
        "<p class='small-caption'>"
        "DINOv2 ViT-S/14 backbone, domain-adapted on 40k OCT images via SSL, "
        "fine-tuned for four downstream OCT analysis tasks."
        "</p>",
        unsafe_allow_html=True,
    )

    #Sidebar
    with st.sidebar:
        st.header("Configuration")

        #Task radio-picks one of the 4 tasks
        task_key = st.radio(
            "Task",
            list(TASKS.keys()),
            format_func=lambda k: TASKS[k]["label"],
        )
        spec = TASKS[task_key]

        #Compare checkbox,turns on the DA-vs-IN dual view
        compare = st.checkbox(
            "Compare against ImageNet baseline",
            value=False,
            help="Run the same image through the ImageNet-only fine-tuned model "
                 "and show the predictions side by side.",
        )

        threshold = 0.5
        if spec["type"] == "multilabel":
            threshold = st.slider(
                "Decision threshold", 0.1, 0.9, 0.5, 0.05,
                help="Biomarkers with sigmoid probability >= this value are "
                     "called 'present'. Drives both the bar panel and the "
                     "active set in the GradCAM grid.",
            )

        #File uploader, accepts standard image formats
        st.divider()
        st.markdown("**Image source**")
        uploaded = st.file_uploader(
            "Upload an OCT image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        )

        # "(none)" - keeps the dropdown valid when the user wants to
        # upload an image instead of picking a sample
        samples = list_samples(task_key)
        sample_choice = None
        if samples:
            options = ["(none)"] + [n for n, _ in samples]
            sample_choice = st.selectbox(
                "Or pick a sample",
                options,
                help="Curated examples per task. Filenames encode the expected "
                     "label so you can sanity-check predictions.",
            )
            if sample_choice == "(none)":
                sample_choice = None

        #footer caption summarizes the architecture.
        st.divider()
        st.caption(
            f"Backbone: `{ARCH}` (384-dim, 12 blocks)  \n"
            f"Input: {IMG_SIZE}×{IMG_SIZE}, ImageNet-norm  \n"
            f"Device: `{DEVICE.type}`"
        )

        with st.expander("About / Model card"):
            st.markdown(
                "**Backbone:** ViT-S/14 (~22 M params)  \n"
                "**SSL data:** 40 k OCT images, official DINOv2 repo with "
                "iBOT + KoLeo, 30 epochs  \n"
                "**Limitation:** A subset of OCT5k images was present in the "
                "SSL pretraining pool - documented in the thesis."
            )

    tab_predict, tab_reports, tab_perf = st.tabs(["Predict", "Reports", "Performance"])

    #Predict tab
    with tab_predict:
        image, source_name, expected_label = None, None, None
        if uploaded is not None:
            try:
                image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
                source_name = uploaded.name
            except Exception as e:
                st.error(f"Could not read uploaded image: {e}")
                return
        elif sample_choice is not None:
            full_path = dict(samples)[sample_choice]
            try:
                image = Image.open(full_path).convert("RGB")
                source_name = sample_choice
                expected_label = parse_sample_label(task_key, sample_choice)
            except Exception as e:
                st.error(f"Could not read sample image: {e}")
                return

        if image is None:
            st.info("Upload an OCT image, or pick a sample from the sidebar.")
            _show_task_summary(spec)
            return

        x = to_tensor(image)

        with st.spinner("Loading domain-adapted model..."):
            try:
                model_da, labels_da = get_model(task_key, "da")
            except Exception as e:
                st.error(f"Failed to load DA checkpoint: {e}")
                return

        model_in, labels_in = None, None
        if compare:
            with st.spinner("Loading ImageNet baseline..."):
                try:
                    model_in, labels_in = get_model(task_key, "in")
                except Exception as e:
                    st.warning(f"ImageNet checkpoint not available: {e}")

        # GradCAM target picker
        st.markdown("### Controls")
        target_kind, target_idx, target_label, focus = _target_picker(
            spec, labels_da, force_focus=(model_in is not None),
        )
        st.caption(
            "GradCAM highlights the regions that most influenced the "
            f"selected output: **{target_label}**."
        )

        #Run predictions + GradCAM
        preds_da = predict(model_da, x, spec["type"])
        cam_da = None
        if spec["type"] != "multilabel" or focus:
            with st.spinner("Computing GradCAM (DA)..."):
                cam_da = gradcam(model_da, x, spec["type"], target_kind, target_idx)

        preds_in = cam_in = None
        if model_in is not None:
            preds_in = predict(model_in, x, spec["type"])
            if spec["type"] != "multilabel" or focus:
                with st.spinner("Computing GradCAM (ImageNet)..."):
                    cam_in = gradcam(model_in, x, spec["type"], target_kind, target_idx)

        DISPLAY_HEIGHT = GRADCAM_HERO_W
        cam_h = Image.fromarray(cam_da) if cam_da is not None else None

        st.markdown("### Results")
        if model_in is None:
            if spec["type"] == "multilabel" and not focus:
                col_orig, col_label = st.columns([2, 3])
                with col_orig:
                    st.image(_resize_to_height(image, DISPLAY_HEIGHT),
                             caption=source_name or "Uploaded image")
                    if expected_label:
                        st.caption(f"Expected (from filename): **{expected_label}**")
                with col_label:
                    st.markdown("**Original scan (left).**  GradCAMs for each "
                                "active biomarker are shown below the predictions.")
                render_predictions(spec, labels_da, preds_da, st,
                                   header="Predictions  ·  domain-adapted",
                                   threshold=threshold)
                st.markdown("##### Active biomarker GradCAMs")
                render_multilabel_gradcam_grid(
                    model_da, x, labels_da["biomarkers"],
                    preds_da["probs"], threshold, st,
                )
            else:
                col_orig, col_cam = st.columns([2, 3])
                with col_orig:
                    st.image(_resize_to_height(image, DISPLAY_HEIGHT),
                             caption=source_name or "Uploaded image")
                    if expected_label:
                        st.caption(f"Expected (from filename): **{expected_label}**")
                with col_cam:
                    st.image(_resize_to_height(cam_h, DISPLAY_HEIGHT),
                             caption=f"GradCAM · {target_label}")
                render_predictions(spec, labels_da, preds_da, st,
                                   header="Predictions  ·  domain-adapted",
                                   threshold=threshold,
                                   side_by_side=True)
        else:
            st.image(_resize_to_height(image, DISPLAY_HEIGHT),
                     caption=source_name or "Uploaded image")
            if expected_label:
                st.caption(f"Expected (from filename): **{expected_label}**")

            cam_in_h = Image.fromarray(cam_in) if cam_in is not None else None

            if spec["type"] == "octdl":
                st.markdown("##### Domain-adapted")
                render_predictions(spec, labels_da, preds_da, st, header="",
                                   side_by_side=True, threshold=threshold)
                st.image(cam_h, caption=f"DA · GradCAM ({target_label})",
                         width=GRADCAM_HERO_W)
                st.divider()
                st.markdown("##### ImageNet baseline")
                render_predictions(spec, labels_in, preds_in, st, header="",
                                   side_by_side=True, threshold=threshold)
                st.image(cam_in_h, caption=f"IN · GradCAM ({target_label})",
                         width=GRADCAM_HERO_W)
            else:
                row1_l, row1_r = st.columns(2)
                with row1_l:
                    st.markdown("##### Domain-adapted")
                    render_predictions(spec, labels_da, preds_da, row1_l,
                                       header="", side_by_side=False,
                                       threshold=threshold)
                with row1_r:
                    st.markdown("##### ImageNet baseline")
                    render_predictions(spec, labels_in, preds_in, row1_r,
                                       header="", side_by_side=False,
                                       threshold=threshold)
                row2_l, row2_r = st.columns(2)
                row2_l.image(cam_h, caption=f"DA - GradCAM ({target_label})",
                             width=GRADCAM_HERO_W)
                row2_r.image(cam_in_h, caption=f"IN - GradCAM ({target_label})",
                             width=GRADCAM_HERO_W)

    #Reports tab
    with tab_reports:
        st.markdown(f"### Pre-generated reports - {task_key}")
        files = gallery_files(task_key, _gallery_mtimes(spec))
        if not files:
            st.info("No report images found in the task's results folders yet.")
        else:
            cat_names = [c for c, _ in REPORT_CATEGORIES]
            cat = st.radio("Category", cat_names, horizontal=True)
            predicate = dict(REPORT_CATEGORIES)[cat]
            filtered = [(rel, full) for rel, full in files if predicate(rel)]

            if not filtered:
                st.info(f"No figures match category **{cat}**.")
            else:
                names = [rel for rel, _ in filtered]
                choice = st.selectbox("Pick a figure", names, index=0)
                chosen_full = dict(filtered)[choice]
                st.image(chosen_full, caption=choice, width="stretch")

                with st.expander(f"All figures in {cat} ({len(filtered)})",
                                 expanded=False):
                    cols = st.columns(3)
                    for i, (rel, full) in enumerate(filtered):
                        with cols[i % 3]:
                            st.image(full, caption=rel, width="stretch")

    #Performance tab
    with tab_perf:
        _render_performance_tab()


def _target_picker(spec, labels, force_focus=False):
    """Return (kind, idx, label, focus). For multilabel tasks, focus=False
    triggers the GradCAM grid view; focus=True falls back to a single-target
    dropdown. `force_focus=True` is used in compare mode where two grids do
    not fit side by side."""
    if spec["type"] == "octdl":
        kind = st.radio("GradCAM head", ["disease", "condition"],
                        horizontal=True, format_func=str.capitalize)
        classes = labels[kind]
        idx = st.selectbox(f"{kind.capitalize()} class", range(len(classes)),
                           format_func=lambda i: classes[i])
        return kind, idx, f"{kind} -> {classes[idx]}", False
    if spec["type"] == "single":
        classes = labels["classes"]
        idx = st.selectbox("Target class", range(len(classes)),
                           format_func=lambda i: classes[i])
        return None, idx, classes[idx], False
    biomarkers = labels["biomarkers"]
    focus = force_focus or st.checkbox(
        "Focus on a single biomarker",
        value=False,
        help="Off (default): show one GradCAM per active biomarker as a grid. "
             "On: pick a single biomarker for higher-detail inspection.",
    )
    if focus:
        idx = st.selectbox("Target biomarker", range(len(biomarkers)),
                           format_func=lambda i: biomarkers[i])
        return None, idx, biomarkers[idx], True
    return None, 0, "active biomarkers (grid)", False


def _show_task_summary(spec):
    st.markdown("### Task")
    st.markdown(f"**{spec['label']}**")
    if spec["type"] == "octdl":
        st.markdown(
            f"- Disease head - {len(spec['disease'])} classes: "
            f"{', '.join(spec['disease'])}"
        )
        st.markdown(
            f"- Condition head - {len(spec['condition'])} classes: "
            f"{', '.join(spec['condition'])}"
        )
    elif spec["type"] == "single":
        st.markdown(
            f"- {len(spec['classes'])} classes: "
            f"{', '.join(spec['classes'])}"
        )
    else:
        st.markdown(
            f"- {len(spec['biomarkers'])} independent biomarkers: "
            f"{', '.join(spec['biomarkers'])}"
        )


def _render_performance_tab():
    """Static thesis numbers, hardcoded. Organised as sub-tabs so each
    dataset gets its run table, per-class breakdown, and external comparison
    without overflowing the screen."""
    st.markdown("### Performance")
    st.caption(
        "Numbers from the thesis report (Phases 7-12). Bold rows mark the "
        "domain-adapted thesis result; comparison rows are external papers "
        "or my own ImageNet baseline."
    )

    sub_overview, sub_pretrain, sub_octdl, sub_mmrdr, sub_corina, sub_oct5k = st.tabs(
        ["Overview", "SSL pretraining",
         "OCTDL", "MMRDR", "Corina", "OCT5k"]
    )

    with sub_overview:
        st.markdown("#### Headline numbers per task")
        st.markdown(_md_table(
            ["Task", "Metric", "DA value", "Compared to"],
            [
                ("OCTDL",  "Disease F1 (test)",   "0.845",
                 "ImageNet baseline 0.800; old ViT-Large SSL 0.749"),
                ("OCTDL",  "Condition F1 (test)", "0.820",
                 "ImageNet baseline 0.741"),
                ("OCTDL",  "Patient-level F1",    "0.829",
                 "Image-level 0.845 (avg-probs)"),
                ("MMRDR",  "Macro F1",            "0.775",
                 "RETFound (SSL 1.6 M) 0.759"),
                ("MMRDR",  "Cohen's κ",           "0.798",
                 "RETFound 0.803"),
                ("MMRDR",  "F1 on NCI-DME (rare)", "0.49",
                 "RETFound 0.436, ResNet-50 0.259"),
                ("Corina", "Macro F1",            "0.890",
                 "ImageNet baseline 0.707; ConvNeXt-base 87 M ≈0.86"),
                ("Corina", "Macro AUC",           "0.970",
                 "ImageNet baseline 0.930"),
                ("Corina", "Exact match (4/4)",   "74.8 %",
                 "ImageNet baseline 49.1 %"),
                ("OCT5k",  "Macro F1",            "0.567",
                 "ImageNet baseline 0.469 (+0.098)"),
                ("OCT5k",  "Macro AUC",           "0.843",
                 "ImageNet baseline 0.678 (+0.165)"),
                ("OCT5k",  "Exact match",         "30.8 %",
                 "ImageNet baseline 13.1 %"),
            ],
        ))

    with sub_pretrain:
        st.markdown("#### Frozen-feature evaluation on OCTDL_CLEANED")
        st.caption(
            "kNN (k=20) + linear probe at 224 px, train=1661 / test=403, "
            "patient-stratified. The four DA rows show how pretraining "
            "quality grows over the 30-epoch SSL run; the ImageNet row and "
            "the old ViT-B/14 row are baselines."
        )
        st.markdown(_md_table(PRETRAIN_FROZEN["header"], PRETRAIN_FROZEN["rows"]))
        st.caption(
            "Linear-probe macro-F1 jumped +0.174 absolute (+26 %) from frozen "
            "features alone. Largest per-class gains: RVO +0.46, NO +0.20, "
            "DME +0.19, VID +0.17."
        )

    with sub_octdl:
        st.markdown("#### Run comparison")
        st.markdown(_md_table(OCTDL_RUNS["header"], OCTDL_RUNS["rows"]))
        st.caption(
            "Run C is the final thesis model: Disease acc 93.48 %, "
            "balanced-acc 83.58 %, macro-F1 0.8446. Condition acc 84.20 %, "
            "balanced-acc 84.05 %, macro-F1 0.8202. AMD F1 = 0.98, NO F1 = 0.94. "
            "ImageNet baseline (Run E) lost 0.044 disease F1 / 0.079 condition "
            "F1 - clean evidence of SSL value."
        )

        st.markdown("#### Patient-level evaluation")
        st.caption(
            "Per-image predictions grouped by patient_id, then aggregated "
            "(majority vote or averaged probabilities). 165 test patients."
        )
        st.markdown(_md_table(OCTDL_PATIENT["header"], OCTDL_PATIENT["rows"]))

        st.markdown("#### Data-efficiency experiment")
        st.caption(
            "Patient-level subsampling at 33 / 66 / 100 % of training data; "
            "val and test fixed. DA already exceeds the 100 % ImageNet "
            "baseline at 33 % training data on disease F1."
        )
        st.markdown(_md_table(OCTDL_DATAEFF["header"], OCTDL_DATAEFF["rows"]))

    with sub_mmrdr:
        st.markdown("#### Cross-dataset transfer to MMRDR")
        st.caption(
            "DME severity grading, 3 classes (No DME / NCI-DME / CI-DME). "
            "NCI-DME is severely under-represented (7.5 %) and is the "
            "hardest class. Patient-level split, train 2376 / test 562."
        )
        st.markdown(_md_table(MMRDR_COMPARE["header"], MMRDR_COMPARE["rows"]))
        st.caption(
            "Single strongest result in the thesis: with 40× less SSL "
            "pretraining data than RETFound (the foundation model for "
            "retinal imaging), the DA ViT-S/14 matches/beats it on macro-F1 "
            "and on the hardest class (NCI-DME)."
        )

    with sub_corina:
        st.markdown("#### Per-biomarker results, DA model")
        st.caption(
            "Patient-level split: train 2414 (38 patients) / val 268 (5) / "
            "test 326 (9). No augmentation. BCEWithLogitsLoss with "
            "pos_weight = neg/pos per label, threshold 0.5."
        )
        st.markdown(_md_table(CORINA_PER_BIO["header"], CORINA_PER_BIO["rows"]))

        st.markdown("#### DA vs ImageNet vs ConvNeXt-base reference")
        st.markdown(_md_table(CORINA_VS["header"], CORINA_VS["rows"]))
        st.caption(
            "Largest single-biomarker gap on HF (DA 0.92 vs IN 0.55). "
            "Multi-label exact-match jumped from 49.1 % to 74.8 %."
        )

    with sub_oct5k:
        st.markdown("#### Per-biomarker DA vs ImageNet")
        st.caption(
            "AMD/DRUSEN patients only. Patient-level 80/10/10 split: "
            "train 399 (42 patients) / val 60 (6) / test 107 (12). "
            "Fluid dropped (<15 positives); 8 active biomarkers."
        )
        st.markdown(_md_table(OCT5K_PER_BIO["header"], OCT5K_PER_BIO["rows"]))
        st.caption(
            "Largest gains on the morphologically subtle features "
            "(RD +0.19, HD +0.18). Small dataset limits absolute scores; "
            "the relative DA advantage is the point."
        )


if __name__ == "__main__":
    main()