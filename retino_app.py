"""
Retino-DINO - OCT analysis demo app.

DINOv2 ViT-S/14 (domain-adapted on 40k OCT images via SSL) + per-task heads.
Four fine-tuned checkpoints:
    OCTDL multitask (Disease 7-class + Condition 8-class)
    MMRDR single-task (DME severity, 3-class)
    Corina multi-label (4 biomarkers)
    OCT5k multi-label (8 biomarkers)

Run: streamlit run retino_app.py
(see details of usage in the app interafec)
"""
import os
import io
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import altair as alt
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

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])


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
    """Load DINOv2 from hub; if weights_path is given, overwrite with the FSDP teacher state-dict (keys prefixed with `teacher.backbone.`).
    weights_path=None -> keep hub-default ImageNet weights (used for the IN baseline).
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
            f"No keys with prefix '{PREFIX}' in {weights_path}; this loader expects an FSDP teacher checkpoint."
        )

    _interpolate_pos_embed(clean, model)
    model.load_state_dict(clean, strict=False)
    return model


def _make_head(in_dim, hidden, out_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class OCTDLModel(nn.Module):
    def __init__(self, backbone, num_diseases, num_conditions, head_hidden=256, head_dropout=0.3):
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


class SingleHeadModel(nn.Module):
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


OCT5K_LABELS = [
    "Choroidalfolds", "Geographicatrophy", "Harddrusen", "Hyperfluorescentspots",
    "PRlayerdisruption", "Reticulardrusen", "Softdrusen", "SoftdrusenPED",
]
CORINA_LABELS = ["DME", "HF", "ND", "Healthy"]
MMRDR_CLASSES = ["No_DME", "NCI_DME", "CI_DME"]
OCTDL_DISEASE = ["AMD", "DME", "ERM", "NO", "RAO", "RVO", "VID"]
OCTDL_CONDITION = ["DRIL", "ERM", "ME", "MH", "MNV", "MNV_suspected", "NO", "drusen"]


def _path(*parts):
    return os.path.join(ROOT, *parts)


TASKS = {
    "OCTDL": {
        "label": "Broad disease screen (7) - OCTDL",
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
        "label": "DME severity grading (3) - MMRDR",
        "type": "single",
        "ckpt_da": "mmrdr_da.pth",
        "ckpt_in": "mmrdr_in.pth",
        "samples_subdir": "mmrdr",
        "classes": MMRDR_CLASSES,
        "gallery_dirs": [ _path("finetune_mmrdr", "results", "confusion_matrices"),
            _path("finetune_mmrdr", "results", "mmrdr"),
        ],
    },
    "Corina": {
        "label": "DME biomarkers (4) - Suciu et al.",
        "type": "multilabel",
        "ckpt_da": "corina_da.pth",
        "ckpt_in": "corina_in.pth",
        "samples_subdir": "corina",
        "biomarkers": CORINA_LABELS,
        "gallery_dirs": [ _path("finetune_corina", "results"), _path("finetune_corina", "results", "corina"),],
    },
    "OCT5k": {
        "label": "AMD / drusen biomarkers (8) - OCT5k",
        "type": "multilabel",
        "ckpt_da": "oct5k_da.pth",
        "ckpt_in": "oct5k_in.pth",
        "samples_subdir": "oct5k",
        "biomarkers": OCT5K_LABELS,
        "gallery_dirs": [ _path("finetune_oct5k", "results"),],
    },
}


@st.cache_resource(show_spinner=False)
def get_model(task_key: str, variant: str):
    """Return (model, labels). `labels` is derived from the saved maps in the checkpoint and falls back to the spec lists when absent."""
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
        diseases = [n for n, _ in sorted(d_map.items(), key=lambda kv: kv[1])]
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

    print(f"[{task_key}/{variant}] ckpt={ckpt_name} epoch={ckpt.get('epoch', '?')} state_dict={len(ckpt['model_state_dict'])} head_missing={head_missing} head_unexpected={head_unexpected} backbone_missing={len(backbone_missing)}", file=sys.stderr)

    if head_missing or head_unexpected:
        st.error(
            f"Head weights did not load cleanly for {task_key}/{variant}: "
            f"missing={head_missing} unexpected={head_unexpected}"
        )

    model.to(DEVICE).eval()

    for p in model.parameters():
        p.requires_grad_(True)
    return model, labels


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
    """Run GradCAM on the last ViT block. For OCTDL the multi-output forward is wrapped in _HeadSelector so GradCAM sees a single tensor."""
    target_layers = [model.backbone.blocks[-1].norm1]

    if task_type == "octdl":
        head_idx = 0 if target_kind == "disease" else 1
        wrapped = _HeadSelector(model, head_idx)
        cam = GradCAM(model=wrapped, target_layers=target_layers, reshape_transform=vit_reshape_transform)
    else:
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=vit_reshape_transform)

    heatmap = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(target_idx))])[0]
    rgb = denormalize(x)
    return show_cam_on_image(rgb, heatmap, use_rgb=True)


_BAR_LEFT = 0.34
_BAR_RIGHT = 0.96
_BAR_BOTTOM = 0.18
_BAR_TOP = 0.86
_BAR_FIG_W = 4.8
_BAR_ROW_H = 0.38
_BAR_PAD_H = 0.95

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
    fig.subplots_adjust(left=_BAR_LEFT, right=_BAR_RIGHT, bottom=_BAR_BOTTOM, top=_BAR_TOP)
    return fig


def _multilabel_panel(probs, labels, threshold=0.5, title=""):
    """Horizontal sigmoid bar chart; bars >= threshold are green, others grey, with a dashed vertical line at the threshold."""
    fig, ax = plt.subplots(figsize=(_BAR_FIG_W, _BAR_ROW_H * len(labels) + _BAR_PAD_H))
    colors = ["#2ca02c" if p >= threshold else "#9aa0a6" for p in probs]
    y = np.arange(len(labels))
    ax.barh(y, probs * 100.0, color=colors, edgecolor="none")
    ax.axvline(threshold * 100, color="#d62728", linestyle="--", linewidth=1, alpha=0.7)
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
    fig.subplots_adjust(left=_BAR_LEFT, right=_BAR_RIGHT, bottom=_BAR_BOTTOM, top=_BAR_TOP)
    return fig


def _confidence_caption(container, max_conf, n_classes):
    container.caption(f"Top confidence: {max_conf*100:.1f}%")
    if max_conf < 0.40:
        container.caption(f"Top confidence only {max_conf*100:.0f}% on a {n_classes}-class head - head may be undertrained or input out-of-distribution.")


DA_COLOR = "#1f77b4"
IN_COLOR = "#ff7f0e"


def _compare_bar_chart(probs_da, probs_in, classes, idx_da, idx_in, title=""):
    n = len(classes)
    fig, ax = plt.subplots(figsize=(_BAR_FIG_W + 1.9, _BAR_ROW_H * n * 1.5 + _BAR_PAD_H))
    y = np.arange(n)
    bar_h = 0.38

    ax.barh(y - bar_h / 2, probs_da * 100.0, height=bar_h, color=DA_COLOR, edgecolor="none")
    ax.barh(y + bar_h / 2, probs_in * 100.0, height=bar_h, color=IN_COLOR, edgecolor="none")

    ax.barh(y[idx_da] - bar_h / 2, probs_da[idx_da] * 100.0, height=bar_h, fill=False, edgecolor="#111111", linewidth=1.8)
    ax.barh(y[idx_in] + bar_h / 2, probs_in[idx_in] * 100.0, height=bar_h, fill=False, edgecolor="#111111", linewidth=1.8)

    ax.set_yticks(y)
    ax.set_yticklabels([_truncate(c) for c in classes], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("probability (%)", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left")

    for i, p in enumerate(probs_da * 100.0):
        ax.text(p + 1.0, i - bar_h / 2, f"{p:.0f}", va="center", fontsize=7)
    for i, p in enumerate(probs_in * 100.0):
        ax.text(p + 1.0, i + bar_h / 2, f"{p:.0f}", va="center", fontsize=7)

    handles = [
        Patch(color=DA_COLOR, label="Domain-adapted (DA)"),
        Patch(color=IN_COLOR, label="ImageNet baseline (IN)"),
        Patch(facecolor="none", edgecolor="#111111", label="top prediction"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=_BAR_LEFT, right=0.72, bottom=_BAR_BOTTOM, top=_BAR_TOP)
    return fig


def _compare_multilabel_panel(probs_da, probs_in, labels, threshold=0.5, title=""):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(_BAR_FIG_W + 1.9, _BAR_ROW_H * n * 1.5 + _BAR_PAD_H))
    y = np.arange(n)
    bar_h = 0.38

    da_colors = [to_rgba(DA_COLOR, 1.0 if p >= threshold else 0.3) for p in probs_da]
    in_colors = [to_rgba(IN_COLOR, 1.0 if p >= threshold else 0.3) for p in probs_in]

    ax.barh(y - bar_h / 2, probs_da * 100.0, height=bar_h, color=da_colors, edgecolor="none")
    ax.barh(y + bar_h / 2, probs_in * 100.0, height=bar_h, color=in_colors, edgecolor="none")
    ax.axvline(threshold * 100, color="#d62728", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels([_truncate(l) for l in labels], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("sigmoid probability (%)", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left")

    for i, p in enumerate(probs_da * 100.0):
        ax.text(p + 1.0, i - bar_h / 2, f"{p:.0f}", va="center", fontsize=7)
    for i, p in enumerate(probs_in * 100.0):
        ax.text(p + 1.0, i + bar_h / 2, f"{p:.0f}", va="center", fontsize=7)

    handles = [
        Patch(color=DA_COLOR, label="Domain-adapted (DA)"),
        Patch(color=IN_COLOR, label="ImageNet baseline (IN)"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=_BAR_LEFT, right=0.72, bottom=_BAR_BOTTOM, top=_BAR_TOP)
    return fig


def render_predictions(spec, labels, preds, container, header, side_by_side=True, threshold=0.5):
    """Dispatch on spec["type"] and render the appropriate predictions panel. `side_by_side` is OCTDL-only
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
        container.markdown(f"**Disease:** `{diseases[d_idx]}` ({d_conf*100:.1f}%)")
        container.markdown(f"**Condition:** `{conditions[c_idx]}` ({c_conf*100:.1f}%)")
        container.caption("In each chart, the red bar is the predicted class.")
        if side_by_side:
            col_d, col_c = container.columns(2)
            col_d.pyplot(_bar_chart(preds["disease"], diseases, d_idx, "Disease head"), clear_figure=True)
            _confidence_caption(col_d, d_conf, len(diseases))
            col_c.pyplot(_bar_chart(preds["condition"], conditions, c_idx, "Condition head"), clear_figure=True)
            _confidence_caption(col_c, c_conf, len(conditions))
        else:
            container.pyplot(_bar_chart(preds["disease"], diseases, d_idx, "Disease head"), clear_figure=True)
            _confidence_caption(container, d_conf, len(diseases))
            container.pyplot(_bar_chart(preds["condition"], conditions, c_idx, "Condition head"), clear_figure=True)
            _confidence_caption(container, c_conf, len(conditions))
    elif spec["type"] == "single":
        classes = labels["classes"]
        idx = int(np.argmax(preds["probs"]))
        conf = float(preds["probs"][idx])
        container.markdown(f"**Prediction:** `{classes[idx]}` ({conf*100:.1f}%)")
        container.caption("The red bar is the predicted class.")
        col_chart, _ = container.columns([3, 2])
        col_chart.pyplot(_bar_chart(preds["probs"], classes, idx), clear_figure=True, use_container_width=False)
        _confidence_caption(container, conf, len(classes))
    else:
        biomarkers = labels["biomarkers"]
        active = [biomarkers[i] for i, p in enumerate(preds["probs"]) if p >= threshold]
        container.markdown(
            f"**Active biomarkers (>={threshold:.2f}):** "
            + (", ".join(f"`{a}`" for a in active) if active else "_none_")
        )
        col_chart, _ = container.columns([3, 2])
        col_chart.pyplot(_multilabel_panel(preds["probs"], biomarkers, threshold=threshold), clear_figure=True, use_container_width=False)


def render_predictions_compare(spec, labels_da, labels_in, preds_da, preds_in, container, threshold=0.5):
    if spec["type"] == "octdl":
        diseases = labels_da["disease"]
        conditions = labels_da["condition"]
        d_idx_da = int(np.argmax(preds_da["disease"]))
        c_idx_da = int(np.argmax(preds_da["condition"]))
        d_idx_in = int(np.argmax(preds_in["disease"]))
        c_idx_in = int(np.argmax(preds_in["condition"]))

        container.markdown(
            f"**Disease** - DA: `{diseases[d_idx_da]}` "
            f"({preds_da['disease'][d_idx_da]*100:.1f}%) "
            f"vs IN: `{diseases[d_idx_in]}` "
            f"({preds_in['disease'][d_idx_in]*100:.1f}%)"
        )
        container.markdown(
            f"**Condition** - DA: `{conditions[c_idx_da]}` "
            f"({preds_da['condition'][c_idx_da]*100:.1f}%) "
            f"vs IN: `{conditions[c_idx_in]}` "
            f"({preds_in['condition'][c_idx_in]*100:.1f}%)"
        )

        col_d, col_c = container.columns(2)
        col_d.pyplot(
            _compare_bar_chart(preds_da["disease"], preds_in["disease"], diseases, d_idx_da, d_idx_in, "Disease head"),
            clear_figure=True,
        )
        col_c.pyplot(
            _compare_bar_chart(preds_da["condition"], preds_in["condition"], conditions, c_idx_da, c_idx_in, "Condition head"),
            clear_figure=True,
        )
    elif spec["type"] == "single":
        classes = labels_da["classes"]
        idx_da = int(np.argmax(preds_da["probs"]))
        idx_in = int(np.argmax(preds_in["probs"]))
        container.markdown(
            f"**Prediction** - DA: `{classes[idx_da]}` ({preds_da['probs'][idx_da]*100:.1f}%) "
            f"vs IN: `{classes[idx_in]}` ({preds_in['probs'][idx_in]*100:.1f}%)"
        )
        col_chart, _ = container.columns([3, 2])
        col_chart.pyplot(
            _compare_bar_chart(preds_da["probs"], preds_in["probs"], classes, idx_da, idx_in),
            clear_figure=True, use_container_width=False,
        )
    else:
        biomarkers = labels_da["biomarkers"]
        active_da = [biomarkers[i] for i, p in enumerate(preds_da["probs"]) if p >= threshold]
        active_in = [biomarkers[i] for i, p in enumerate(preds_in["probs"]) if p >= threshold]
        container.markdown(
            f"**Active (DA, >={threshold:.2f}):** "
            + (", ".join(f"`{a}`" for a in active_da) if active_da else "_none_")
        )
        container.markdown(
            f"**Active (IN, >={threshold:.2f}):** "
            + (", ".join(f"`{a}`" for a in active_in) if active_in else "_none_")
        )
        col_chart, _ = container.columns([3, 2])
        col_chart.pyplot(
            _compare_multilabel_panel(preds_da["probs"], preds_in["probs"], biomarkers, threshold=threshold),
            clear_figure=True, use_container_width=False,
        )


def _select_grid_indices(probs, threshold, fallback_top_k=3):
    """Pick which biomarkers to GradCAM. Active set if any cross threshold, else top-K by probability so the grid is never empty."""
    active = [i for i, p in enumerate(probs) if p >= threshold]
    if active:
        return active, False
    k = min(fallback_top_k, len(probs))
    top = list(np.argsort(probs)[::-1][:k])
    return [int(i) for i in top], True


def compute_multilabel_grid_cams(model, x, biomarkers, probs, threshold):
    idxs, fell_back = _select_grid_indices(probs, threshold)
    panels = []
    for bio_idx in idxs:
        cam_img = gradcam(model, x, "multilabel", None, bio_idx)
        panels.append((biomarkers[bio_idx], float(probs[bio_idx]), cam_img))
    return panels, fell_back


def render_multilabel_gradcam_grid(panels, fell_back, threshold, container, n_cols=None):
    """One GradCAM panel per active biomarker (or top-3 fallback). 2 columns for Corina, 4 for OCT5k."""
    if not panels:
        return
    if n_cols is None:
        n_cols = 4 if len(panels) >= 4 else 2
    if fell_back:
        container.caption(f"No biomarkers above threshold {threshold:.2f}; showing top {len(panels)} by probability instead.")
    rows = [panels[i:i + n_cols] for i in range(0, len(panels), n_cols)]
    for row in rows:
        cols = container.columns(n_cols)
        for j, (name, prob, cam_img) in enumerate(row):
            cols[j].image(cam_img, caption=f"{name} - {prob*100:.1f}%", width=GRADCAM_GRID_W)


def _resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
    """Resize a PIL image to a fixed display height, preserving aspect.
    Used so the original and GradCAM panels line up vertically even when the
    user uploads a landscape OCT scan."""
    if img.height == 0:
        return img
    w = max(1, int(round(img.width * target_h / img.height)))
    return img.resize((w, target_h), Image.LANCZOS)


LATENT_DIRS = [
    ("OCTDL backbone - domain-adapted", _path("results", "umap", "domain_adapted", "umap_data.npz")),
    ("OCTDL backbone - ImageNet baseline", _path("results", "umap", "imagenet_baseline", "umap_data.npz")),
]


@st.cache_data(show_spinner=False)
def _load_latent(path):
    data = np.load(path, allow_pickle=True)
    return {
        "x": data["embedding"][:, 0].astype(float),
        "y": data["embedding"][:, 1].astype(float),
        "label": [str(l) for l in data["labels"]],
    }


def _interactive_latent_chart(payload, title=""):
    df = pd.DataFrame(payload)
    df["idx"] = np.arange(len(df))
    selection = alt.selection_point(fields=["label"], bind="legend")
    chart = (
        alt.Chart(df)
        .mark_circle(size=55, opacity=0.75)
        .encode(
            x=alt.X("x:Q", title="component 1"),
            y=alt.Y("y:Q", title="component 2"),
            color=alt.Color("label:N", title="class", scale=alt.Scale(scheme="tableau10")),
            tooltip=[
                alt.Tooltip("label:N", title="class"),
                alt.Tooltip("idx:Q", title="point #"),
                alt.Tooltip("x:Q", format=".2f"),
                alt.Tooltip("y:Q", format=".2f"),
            ],
            opacity=alt.condition(selection, alt.value(0.85), alt.value(0.05)),
        )
        .add_params(selection)
        .properties(height=520, title=title)
        .interactive()
    )
    return chart


def _pil_to_fig(img, title=None):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def _text_fig(lines, title="Retino-DINO report"):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.5, 0.95, title, ha="center", va="top", fontsize=18, weight="bold", transform=ax.transAxes)
    body = "\n".join(lines)
    ax.text(0.06, 0.88, body, ha="left", va="top", fontsize=11, family="monospace", transform=ax.transAxes, wrap=True)
    return fig


def build_pdf_report(
    image, source_name, expected_label, spec, task_key,
    labels_da, preds_da, cam_da, target_label_da,
    labels_in=None, preds_in=None, cam_in=None, target_label_in=None,
    threshold=0.5, grid_panels=None,
):
    buf = io.BytesIO()
    summary_lines = [
        f"Task        : {spec['label']}",
        f"Image       : {source_name or 'uploaded image'}",
    ]
    if expected_label:
        summary_lines.append(f"Expected    : {expected_label}")
    summary_lines.append("")

    if spec["type"] == "octdl":
        d_idx = int(np.argmax(preds_da["disease"]))
        c_idx = int(np.argmax(preds_da["condition"]))
        summary_lines.append(f"DA Disease  : {labels_da['disease'][d_idx]} ({preds_da['disease'][d_idx]*100:.1f}%)")
        summary_lines.append(f"DA Condition: {labels_da['condition'][c_idx]} ({preds_da['condition'][c_idx]*100:.1f}%)")
    elif spec["type"] == "single":
        i = int(np.argmax(preds_da["probs"]))
        summary_lines.append(f"DA Predict  : {labels_da['classes'][i]} ({preds_da['probs'][i]*100:.1f}%)")
    else:
        active = [labels_da["biomarkers"][i] for i, p in enumerate(preds_da["probs"]) if p >= threshold]
        summary_lines.append(f"DA active   : {', '.join(active) if active else 'none above threshold'}")
        summary_lines.append(f"Threshold   : {threshold:.2f}")

    if preds_in is not None:
        summary_lines.append("")
        if spec["type"] == "octdl":
            d_idx = int(np.argmax(preds_in["disease"]))
            c_idx = int(np.argmax(preds_in["condition"]))
            summary_lines.append(f"IN Disease  : {labels_in['disease'][d_idx]} ({preds_in['disease'][d_idx]*100:.1f}%)")
            summary_lines.append(f"IN Condition: {labels_in['condition'][c_idx]} ({preds_in['condition'][c_idx]*100:.1f}%)")
        elif spec["type"] == "single":
            i = int(np.argmax(preds_in["probs"]))
            summary_lines.append(f"IN Predict  : {labels_in['classes'][i]} ({preds_in['probs'][i]*100:.1f}%)")
        else:
            active = [labels_in["biomarkers"][i] for i, p in enumerate(preds_in["probs"]) if p >= threshold]
            summary_lines.append(f"IN active   : {', '.join(active) if active else 'none above threshold'}")

    summary_lines.append("")
    summary_lines.append("DA = domain-adapted (SSL on 40k OCT images)")
    summary_lines.append("IN = ImageNet baseline")

    with PdfPages(buf) as pdf:
        pdf.savefig(_text_fig(summary_lines), bbox_inches="tight")
        pdf.savefig(_pil_to_fig(image, title="Original OCT scan"), bbox_inches="tight")

        if spec["type"] == "octdl":
            d_idx_da = int(np.argmax(preds_da["disease"]))
            c_idx_da = int(np.argmax(preds_da["condition"]))
            if preds_in is not None:
                d_idx_in = int(np.argmax(preds_in["disease"]))
                c_idx_in = int(np.argmax(preds_in["condition"]))
                pdf.savefig(
                    _compare_bar_chart(preds_da["disease"], preds_in["disease"], labels_da["disease"], d_idx_da, d_idx_in, "Disease head"),
                    bbox_inches="tight",
                )
                pdf.savefig(
                    _compare_bar_chart(preds_da["condition"], preds_in["condition"], labels_da["condition"], c_idx_da, c_idx_in, "Condition head"),
                    bbox_inches="tight",
                )
            else:
                pdf.savefig(_bar_chart(preds_da["disease"], labels_da["disease"], d_idx_da, "Disease head (DA)"), bbox_inches="tight")
                pdf.savefig(_bar_chart(preds_da["condition"], labels_da["condition"], c_idx_da, "Condition head (DA)"), bbox_inches="tight")
        elif spec["type"] == "single":
            idx_da = int(np.argmax(preds_da["probs"]))
            if preds_in is not None:
                idx_in = int(np.argmax(preds_in["probs"]))
                pdf.savefig(
                    _compare_bar_chart(preds_da["probs"], preds_in["probs"], labels_da["classes"], idx_da, idx_in, "Predictions"),
                    bbox_inches="tight",
                )
            else:
                pdf.savefig(_bar_chart(preds_da["probs"], labels_da["classes"], idx_da, "Predictions (DA)"), bbox_inches="tight")
        else:
            if preds_in is not None:
                pdf.savefig(
                    _compare_multilabel_panel(preds_da["probs"], preds_in["probs"], labels_da["biomarkers"], threshold=threshold, title="Biomarkers"),
                    bbox_inches="tight",
                )
            else:
                pdf.savefig(
                    _multilabel_panel(preds_da["probs"], labels_da["biomarkers"], threshold=threshold, title="Biomarkers (DA)"),
                    bbox_inches="tight",
                )

        if cam_da is not None:
            pdf.savefig(_pil_to_fig(cam_da, title=f"DA GradCAM - {target_label_da}"), bbox_inches="tight")
        if cam_in is not None:
            pdf.savefig(_pil_to_fig(cam_in, title=f"IN GradCAM - {target_label_in}"), bbox_inches="tight")

        for name, prob, cam_img in (grid_panels or []):
            pdf.savefig(_pil_to_fig(cam_img, title=f"DA GradCAM - {name} ({prob*100:.1f}%)"), bbox_inches="tight")

    buf.seek(0)
    return buf.getvalue()


OCT5K_ABBREV = {
    "CF": "Choroidalfolds",
    "GA": "Geographicatrophy",
    "HD": "Harddrusen",
    "HFS": "Hyperfluorescentspots",
    "PRL": "PRlayerdisruption",
    "RD": "Reticulardrusen",
    "SD": "Softdrusen",
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
            return f"condition - {cond}"
        return f"disease - {parts[0]}"

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
    return tuple(os.path.getmtime(d) for d in spec["gallery_dirs"] if os.path.isdir(d))


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
    ("All", lambda r: True),
    ("Confusion matrices", lambda r: "confusion" in r.lower()),
    ("GradCAM", lambda r: "explainability" in r.lower() or "gradcam" in r.lower() or "grad_cam" in r.lower()),
    ("Latent space (t-SNE)", lambda r: "tsne" in r.lower() or "t-sne" in r.lower() or "latent" in r.lower()),
    ("Disease vs Condition", lambda r: "disease_vs_condition" in r.lower()),
    ("Patient-level", lambda r: "patient_level" in r.lower() or "patient-level" in r.lower()),
    ("Data efficiency", lambda r: "data_efficiency" in r.lower() or "efficiency" in r.lower()),
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
        ("ImageNet ViT-S/14 (baseline)", "0.739", "0.382", "0.419", "0.804", "0.663", "0.667"),
        ("ViT-S/14 Epoch 5 (6,250 iter)", "0.911", "0.755", "0.774", "0.896", "0.764", "0.738"),
        ("ViT-S/14 Epoch 15 (18,750 iter)", "0.931", "0.796", "0.812", "0.936", "0.810", "0.820"),
        ("ViT-S/14 Final (37,500 iter)", "0.938", "0.804", "0.824", "0.938", "0.835", "0.841"),
        ("Old manual ViT-B/14 (before using dino repo)", "0.794", "0.498", "0.558", "0.888", "0.748", "0.742"),
    ],
}

OCTDL_RUNS = {
    "header": ["Run", "Strategy", "Disease F1", "Cond F1", "Best epoch"],
    "rows": [
        ("A", "Frozen (0 blocks), lr_heads 1e-3", "0.808", "0.792", "5"),
        ("B", "Unfreeze 1 block (block 11 + norm)", "0.833", "0.804", "4"),
        ("C", "Unfreeze 2 blocks (10-11 + norm)", "**0.845**", "**0.820**", "4"),
        ("D", "Run C + RandomHorizontalFlip", "0.841", "0.801", "4"),
        ("E", "Run C config, ImageNet ViT-S/14 (no DA)", "0.800", "0.741", "18"),
        ("Before using dino repo", "ViT-Large partial unfreeze, old SSL", "0.749", "0.704", "15-23"),
    ],
}

OCTDL_PATIENT = {
    "header": ["Metric", "Image-level", "Patient (majority vote)", "Patient (avg-probs)"],
    "rows": [
        ("Disease Acc", "93.5 %", "90.9 %", "91.5 %"),
        ("Disease Macro-F1", "0.8446", "0.8285", "0.8344"),
        ("Condition Acc", "84.2 %", "83.4 %", "-"),
        ("Condition Macro-F1", "0.8202", "0.8142", "-"),
    ],
}

OCTDL_DATAEFF = {
    "header": ["Fraction", "N_train", "DA Disease F1", "IN Disease F1", "d",
               "DA Cond F1", "IN Cond F1", "d"],
    "rows": [
        ("33 %", "459", "0.820", "0.674", "+0.146", "0.734", "0.634", "+0.100"),
        ("66 %", "947", "0.850", "0.784", "+0.066", "0.802", "0.652", "+0.150"),
        ("100 %", "1,410", "0.837", "0.797", "+0.040", "0.807", "0.696", "+0.111"),
    ],
}

MMRDR_COMPARE = {
    "header": ["Method", "Pretraining", "Aug", "Acc", "kappa", "Macro F1", "F1 NCI-DME"],
    "rows": [
        ("ViT-Base (paper)", "ImageNet", "Yes", "0.883", "0.773", "0.700", "0.280"),
        ("ResNet-50 (paper)", "ImageNet", "Yes", "0.890", "0.791", "0.701", "0.259"),
        ("RETFound (paper)", "SSL 1.6 M", "Yes", "0.897", "0.803", "0.759", "0.436"),
        ("My ImageNet (Run E)", "ImageNet", "No", "0.845", "0.722", "0.689", "0.28"),
        ("My ImageNet + aug", "ImageNet", "Yes", "0.769", "0.625", "0.662", "0.29"),
        ("**My DA (40 k SSL)**", "SSL 40 k", "No", "0.875", "0.770", "0.760", "0.47"),
        ("**My DA + aug**", "SSL 40 k", "Yes", "**0.890**", "**0.798**", "**0.775**", "**0.49**"),
    ],
}

CORINA_PER_BIO = {
    "header": ["Biomarker", "F1", "AUC", "Acc"],
    "rows": [
        ("DME", "0.909", "0.980", "88.7 %"),
        ("HF", "0.922", "0.942", "89.0 %"),
        ("ND", "0.813", "0.964", "92.6 %"),
        ("Healthy", "0.917", "0.993", "95.7 %"),
        ("**Macro**", "**0.890**", "**0.970**", "-"),
    ],
}

CORINA_VS = {
    "header": ["Variant", "Macro F1", "Macro AUC", "Exact Match"],
    "rows": [
        ("My DA (22 M, SSL 40 k)", "0.890", "0.970", "74.8 %"),
        ("My ImageNet (22 M)", "0.707", "0.930", "49.1 %"),
        ("ConvNeXt-base reference (87 M)", "~0.86", "-", "-"),
    ],
}

OCT5K_PER_BIO = {
    "header": ["Biomarker (n_pos)", "DA F1", "IN F1", "d F1",
               "DA AUC", "IN AUC", "d AUC"],
    "rows": [
        ("CF (11)", "0.50", "0.35", "+0.15", "0.83", "0.76", "+0.07"),
        ("GA (19)", "0.84", "0.73", "+0.11", "0.98", "0.90", "+0.08"),
        ("HD (43)", "0.79", "0.61", "+0.18", "0.85", "0.77", "+0.08"),
        ("HFS (1)", "0.08", "0.00", "-", "0.87", "0.10", "-"),
        ("PRL (42)", "0.84", "0.78", "+0.06", "0.95", "0.93", "+0.02"),
        ("RD (16)", "0.39", "0.20", "+0.19", "0.69", "0.53", "+0.16"),
        ("SD (69)", "0.85", "0.86", "-0.01", "0.88", "0.87", "+0.01"),
        ("SDPED (21)", "0.24", "0.23", "+0.01", "0.70", "0.57", "+0.13"),
        ("**Macro**", "**0.567**", "**0.469**", "**+0.098**", "**0.843**", "**0.678**", "**+0.165**"),
        ("Exact Match", "30.8 %", "13.1 %", "+17.7 %", "-", "-", "-"),
    ],
}


SIDEBAR_BLURBS = {
    "OCTDL": "Broad disease screen: 7 macular OCT disease classes plus a finer pathological-condition label.",
    "MMRDR": "DME staging: grade the OCT scan as No DME / non-center-involved / center-involved.",
    "Corina": "DME biomarkers: check which of intraretinal cysts (DME), hyperreflective foci (HF) and neurosensory detachment (ND) are visible.",
    "OCT5k": "AMD / drusen workup: panel of 8 structural biomarkers (drusen subtypes, GA, PRL, etc.).",
}


PRETRAIN_BLURB = (
    "46,175 unlabelled retinal images, drawn only from the training partitions of three sources: OCTDL (all 7 disease classes), an institutional clinical OCT collection from the same source as Suciu et al. (94 patients, B-scans labelled by clinical finding - CNV, CSR, diabetic retinopathy, macular hole), and a subset of OCT5k. Predominantly macular B-scans; a small number of fundus photographs and scanner artefacts are kept because the self-supervised objective works on raw pixels and is label-agnostic.\n\n"
    "This is what the domain-adapted (DA) backbone was fine-tuned from. The ImageNet baseline (IN) uses the same architecture but starts from generic photos, which is what we compare against."
)


DATASETS_INFO = {
    "OCTDL": {
        "title": "OCTDL - primary disease + pathological condition",
        "use_for": "Use this when you have one macular B-scan and want a broad differential across the most common retinal diseases, together with a finer-grained pathological-finding label.",
        "what": "Publicly released collection of macular OCT B-scans, each annotated with both a primary disease class and a more specific pathological condition. After cleaning (we strip the black vitreous band and burned-in scanner text with an adaptive flood-fill) the working set is 2,064 images from 820 patients.",
        "labels": [
            "Disease head, 7 classes: AMD (age-related macular degeneration), DME (diabetic macular edema), ERM (epiretinal membrane), NO (normal), RAO (retinal artery occlusion), RVO (retinal vein occlusion), VID (vitreomacular interface disease).",
            "Condition head, 8 classes (only labels with at least 30 samples are kept): MNV (macular neovascular membranes), suspected MNV, DRIL (disorganisation of retinal inner layers), drusen, ME (macular edema), MH (macular hole), ERM, normal.",
        ],
        "split": "Patient-level 70 / 10 / 20 split, stratified on disease: 1,410 train / 209 val / 445 test, from 574 / 82 / 165 patients.",
    },
    "MMRDR": {
        "title": "MMRDR-OCT - DME severity grading",
        "use_for": "Use this when you want to stage diabetic macular edema on a single OCT B-scan, e.g. to flag whether the macula is at risk and intervention is warranted.",
        "what": "OCT subset of the multimodal retinal imaging dataset of Tang et al. (the fundus and ultrawide-field portions are not used here). 2,938 B-scans annotated with three DME severity grades.",
        "labels": [
            "Grade 0: No DME.",
            "Grade 1: Non-center-involved DME (NCI-DME). Only 7.5 % of the dataset, so the hardest class to call correctly.",
            "Grade 2: Center-involved DME (CI-DME).",
        ],
        "split": "Released train/test split is at the image level (no patient IDs in the metadata): 2,376 train / 562 test. We carved a 10 % grade-stratified validation slice from the training pool.",
    },
    "Corina": {
        "title": "Suciu et al. (Corina) - DME biomarkers, multi-label",
        "use_for": "Use this when you have an OCT of a suspected DME patient and want to confirm which specific biomarkers are visible on the scan - more than one can be present at the same time.",
        "what": "Macular B-scans collected from 52 patients at the Iuliu Hatieganu UMF and the Emergency County Hospital in Cluj-Napoca. After excluding 337 images flagged in the original release as other diseases or imaging artefacts, the working set is 3,008 scans.",
        "labels": [
            "DME: intraretinal cystoid spaces (present in 64.5 % of scans).",
            "HF: hyperreflective foci (73.6 %).",
            "ND: neurosensory detachment (22.2 %).",
            "Healthy: biomarker-free control flag (17.0 %).",
            "A single scan can carry several biomarkers at once (e.g. DME + HF + ND), so the model output is a four-dimensional binary vector, not a single class.",
        ],
        "split": "Patient-level split released by the authors: 43 patients (2,682 images) for training, 9 patients (326 images) for test. We carved a 10 % patient-stratified validation slice from the training pool (stratification on each patient's most-frequent biomarker).",
    },
    "OCT5k": {
        "title": "OCT5k - AMD / drusen structural biomarkers, multi-label",
        "use_for": "Use this for an AMD or drusen workup, when you want a panel of structural biomarkers rather than a single disease label.",
        "what": "566 macular OCT B-scans from 60 patients with age-related macular degeneration or drusen. Originally annotated with 4,698 bounding boxes; we collapse those to scan-level binary indicators per biomarker. The Fluid class had fewer than 15 positive scans and was dropped, leaving 8 active biomarkers. Each scan carries on average 2.2 annotations.",
        "labels": [
            "CF: choroidal folds (11 % of scans, rare).",
            "GA: geographic atrophy.",
            "HD: hard drusen.",
            "HFS: hyperfluorescent spots.",
            "PRL: photoreceptor-layer disruption.",
            "RD: reticular drusen.",
            "SD: soft drusen (63 % of scans, dominant class).",
            "SDPED: soft drusen with pigment-epithelium detachment.",
        ],
        "split": "Patient-level 80 / 10 / 10 split: 42 / 6 / 12 patients = 399 / 60 / 107 scans.",
    },
}


def main():
    st.set_page_config(
        page_title="Retino-DINO",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1280px; }
        .small-caption { color: #6b7280; font-size: 0.9rem; }
        section[data-testid="stSidebar"] h4 {
            margin-top: 0.4rem;
            margin-bottom: 0.2rem;
            color: #4338ca;
            font-size: 0.78rem;
            text-transform: uppercase;
        }
        @keyframes rdFadeUp {
            from { opacity: 0; transform: translateY(12px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes rdSheen {
            from { background-position: 0% 50%; }
            to   { background-position: 100% 50%; }
        }
        .hero-title {
            font-size: 2.6rem; font-weight: 700; margin: 0.2rem 0 0.1rem;
            background: linear-gradient(90deg, #1f2430 0%, #4338ca 50%, #1f2430 100%);
            background-size: 220% auto;
            -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: rdFadeUp 0.6s ease both, rdSheen 7s linear infinite alternate;
        }
        .hero-sub { font-size: 1.15rem; color: #4338ca; font-weight: 500; margin-bottom: 1.4rem;
            animation: rdFadeUp 0.6s ease 0.08s both; }
        .info-card {
            background: #f4f5fb;
            border: 1px solid #e5e7f2;
            border-left: 3px solid #4338ca;
            border-radius: 10px;
            padding: 1.05rem 1.2rem;
            height: 100%;
            animation: rdFadeUp 0.5s ease both;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-left-width 0.18s ease;
        }
        .info-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 24px rgba(67,56,202,0.13);
            border-left-width: 6px;
        }
        .info-card h4 { margin: 0 0 0.45rem; font-size: 0.78rem; letter-spacing: 0.04em;
            text-transform: uppercase; color: #4338ca; }
        .info-card p { margin: 0; color: #353a45; font-size: 0.95rem; line-height: 1.45; }
        .stat-strip { display: flex; gap: 0.9rem; flex-wrap: wrap; margin: 0.2rem 0 1.7rem; }
        .stat-card {
            flex: 1; min-width: 150px;
            background: #ffffff;
            border: 1px solid #e5e7f2;
            border-top: 3px solid #4338ca;
            border-radius: 10px;
            padding: 0.9rem 1.05rem;
            animation: rdFadeUp 0.5s ease both;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .stat-card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(67,56,202,0.11); }
        .stat-num { font-size: 1.95rem; font-weight: 700; color: #4338ca; line-height: 1.1; }
        .stat-label { font-size: 0.78rem; color: #6b7280; text-transform: uppercase;
            letter-spacing: 0.03em; margin-top: 0.3rem; }
        .step-num { color: #4338ca; font-weight: 700; }
        .stButton > button { transition: transform 0.15s ease, box-shadow 0.15s ease; }
        .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 18px rgba(67,56,202,0.22); }
        button[data-baseweb="tab"] { transition: color 0.15s ease; }
        .disclaimer {
            background: #fff8ec;
            border: 1px solid #f3e2bf;
            border-radius: 8px;
            padding: 0.6rem 0.9rem;
            color: #7a5b16;
            font-size: 0.88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.get("app_started"):
        _render_landing()
        return

    st.title("Retino-DINO")
    st.markdown(
        "<p class='small-caption'>"
        "Decision-support tool for retinal OCT scans. The Analyse tab is the clinical workspace: pick a task, upload a scan, and read the prediction with a heatmap of the regions it relied on. The Task guide, Latent space, Reports and Performance tabs are the research layer, with the evidence behind the model."
        "</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Setup")

        st.markdown("#### 1. Clinical task")
        task_key = st.radio(
            " ",
            list(TASKS.keys()),
            format_func=lambda k: TASKS[k]["label"],
            label_visibility="collapsed",
        )
        spec = TASKS[task_key]
        if task_key in SIDEBAR_BLURBS:
            st.caption(SIDEBAR_BLURBS[task_key])
        st.caption("Need a longer description? See the Task guide tab.")

        st.markdown("#### 2. Scan")
        uploaded = st.file_uploader(
            "Upload an OCT image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            label_visibility="collapsed",
        )

        samples = list_samples(task_key)
        sample_choice = None
        if samples:
            options = ["(none)"] + [n for n, _ in samples]
            sample_choice = st.selectbox("Or pick a curated sample", options)
            if sample_choice == "(none)":
                sample_choice = None

        st.markdown("#### 3. Options")
        compare = st.checkbox(
            "Compare against ImageNet baseline",
            value=False,
            help="Off: show only the domain-adapted model (DA), the one trained on retinal scans. "
                 "On: also run a generic ImageNet model (IN) side by side, to show how much the retinal "
                 "adaptation helps. Results are labelled DA (blue) and IN (orange).",
        )

        threshold = 0.5
        if spec["type"] == "multilabel":
            threshold = st.slider(
                "Decision threshold", 0.1, 0.9, 0.5, 0.05,
                help="Biomarkers with sigmoid probability >= this value are called 'present'. Drives both the bar chart and the set of biomarkers shown in the GradCAM grid.",
            )

        st.divider()
        with st.expander("Model card", expanded=False):
            st.markdown(f"**Backbone:** `{ARCH}` (ViT-S/14, ~22 M params, 384-dim, 12 transformer blocks)")
            st.markdown(f"**Input:** {IMG_SIZE}x{IMG_SIZE}, ImageNet-norm")
            st.markdown(f"**Device:** `{DEVICE.type}`")
            st.markdown("**SSL data:** ~40 k retinal images, official DINOv2 repo with iBOT + KoLeo, 30 epochs.")

    tab_predict, tab_datasets, tab_latent, tab_reports, tab_perf = st.tabs(
        ["Analyse", "Task guide", "Latent space (OCTDL)", "Reports", "Performance"]
    )

    with tab_predict:
        image, source_name, expected_label = None, None, None
        if uploaded is not None:
            try:
                image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
                source_name = uploaded.name
            except Exception as e:
                st.error(f"Could not read uploaded image: {e}")
                image = None
        elif sample_choice is not None:
            full_path = dict(samples)[sample_choice]
            try:
                image = Image.open(full_path).convert("RGB")
                source_name = sample_choice
                expected_label = parse_sample_label(task_key, sample_choice)
            except Exception as e:
                st.error(f"Could not read sample image: {e}")
                image = None

        model_da, labels_da = None, None
        if image is None:
            _render_welcome(task_key, spec)
        else:
            x = to_tensor(image)
            with st.spinner("Loading domain-adapted model..."):
                try:
                    model_da, labels_da = get_model(task_key, "da")
                except Exception as e:
                    st.error(f"Failed to load DA checkpoint: {e}")
                    model_da = None

        if image is not None and model_da is not None:
            _run_analyse_body(image, source_name, expected_label, x, task_key, spec, compare, threshold, model_da, labels_da)

    with tab_datasets:
        _render_datasets_tab(current_task=task_key)

    with tab_latent:
        _render_latent_tab()

    with tab_reports:
        st.caption("Archive of figures from training and evaluation. Filter by category, then pick a figure.")
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
                st.info(f"No static figures match category **{cat}**.")
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

    with tab_perf:
        _render_performance_tab()


def _resolve_targets(spec, labels_da, labels_in, preds_da, preds_in, auto_target, force_focus=False):
    if spec["type"] == "octdl":
        kind = st.radio(
            "Explain which prediction?", ["disease", "condition"], horizontal=True, format_func=str.capitalize,
            help="Disease = the broad diagnosis (7 classes). Condition = the finer pathological finding (8 classes). "
                 "The heatmap below explains the head you pick here.",
        )
        classes_da = labels_da[kind]
        classes_in = labels_in[kind] if labels_in else classes_da
        if auto_target:
            idx_da = int(np.argmax(preds_da[kind]))
            idx_in = int(np.argmax(preds_in[kind])) if preds_in is not None else idx_da
        else:
            idx_da = st.selectbox(f"{kind.capitalize()} class", range(len(classes_da)), format_func=lambda i: classes_da[i])
            idx_in = idx_da
        label_da = f"{kind} -> {classes_da[idx_da]}"
        label_in = f"{kind} -> {classes_in[idx_in]}"
        return kind, idx_da, idx_in, label_da, label_in, False

    if spec["type"] == "single":
        classes_da = labels_da["classes"]
        classes_in = labels_in["classes"] if labels_in else classes_da
        if auto_target:
            idx_da = int(np.argmax(preds_da["probs"]))
            idx_in = int(np.argmax(preds_in["probs"])) if preds_in is not None else idx_da
        else:
            idx_da = st.selectbox("Target class", range(len(classes_da)), format_func=lambda i: classes_da[i])
            idx_in = idx_da
        return None, idx_da, idx_in, classes_da[idx_da], classes_in[idx_in], False

    biomarkers_da = labels_da["biomarkers"]
    biomarkers_in = labels_in["biomarkers"] if labels_in else biomarkers_da
    focus = force_focus or st.checkbox(
        "Focus on a single biomarker",
        value=False,
        help="Off: show one GradCAM per active biomarker as a grid. On: pick a single biomarker for higher-detail inspection.",
    )
    if focus:
        if auto_target:
            idx_da = int(np.argmax(preds_da["probs"]))
            idx_in = int(np.argmax(preds_in["probs"])) if preds_in is not None else idx_da
        else:
            idx_da = st.selectbox("Target biomarker", range(len(biomarkers_da)), format_func=lambda i: biomarkers_da[i])
            idx_in = idx_da
        return None, idx_da, idx_in, biomarkers_da[idx_da], biomarkers_in[idx_in], True
    return None, 0, 0, "active biomarkers (grid)", "active biomarkers (grid)", False


def _show_task_summary(spec):
    st.markdown("**What it returns**")
    if spec["type"] == "octdl":
        st.markdown(f"- A disease label, 1 of {len(spec['disease'])}: {', '.join(spec['disease'])}")
        st.markdown(f"- A condition label, 1 of {len(spec['condition'])}: {', '.join(spec['condition'])}")
    elif spec["type"] == "single":
        st.markdown(f"- A severity grade, 1 of {len(spec['classes'])}: {', '.join(spec['classes'])}")
    else:
        st.markdown(f"- A yes/no for each of {len(spec['biomarkers'])} biomarkers: {', '.join(spec['biomarkers'])}")


def _run_analyse_body(image, source_name, expected_label, x, task_key, spec, compare, threshold, model_da, labels_da):
    model_in, labels_in = None, None
    if compare:
        with st.spinner("Loading ImageNet baseline..."):
            try:
                model_in, labels_in = get_model(task_key, "in")
            except Exception as e:
                st.warning(f"ImageNet checkpoint not available: {e}")

    preds_da = predict(model_da, x, spec["type"])
    preds_in = None
    if model_in is not None:
        preds_in = predict(model_in, x, spec["type"])

    DISPLAY_HEIGHT = GRADCAM_HERO_W

    st.markdown("### Prediction")
    if model_in is not None:
        st.caption(
            "Comparison mode is on: domain-adapted model (DA) vs ImageNet baseline (IN). "
            "Turn it off in the sidebar under Options."
        )
    st.image(_resize_to_height(image, DISPLAY_HEIGHT), caption=source_name or "Uploaded image")
    if expected_label:
        st.caption(f"Expected (from filename): **{expected_label}**")

    if model_in is None:
        render_predictions(spec, labels_da, preds_da, st, header=None, threshold=threshold, side_by_side=True)
    else:
        if spec["type"] == "multilabel":
            st.caption("Blue = domain-adapted (DA), orange = ImageNet baseline (IN). Dashed red line = decision threshold; bars past it are called present. Faded bars are below threshold.")
        else:
            st.caption("Blue = domain-adapted (DA), orange = ImageNet baseline (IN). The black-outlined bar is each model's top prediction.")
        render_predictions_compare(spec, labels_da, labels_in, preds_da, preds_in, st, threshold=threshold)

    st.markdown("### Where the model looked")
    st.caption("GradCAM highlights the regions that most influenced the prediction. Red/warm = strong contribution, blue/cool = weak.")

    auto_target = st.checkbox(
        "Auto-target each model's own predicted class (recommended)",
        value=True,
        help="On: the heatmap explains the class the model actually chose, so DA and IN can highlight different regions if they disagree. Off: you pick one class and both models are explained against it.",
    )
    target_kind, idx_da, idx_in, target_label_da, target_label_in, focus = _resolve_targets(
        spec, labels_da, labels_in, preds_da, preds_in, auto_target, force_focus=(model_in is not None),
    )

    cam_da = None
    if spec["type"] != "multilabel" or focus:
        with st.spinner("Computing GradCAM (DA)..."):
            cam_da = gradcam(model_da, x, spec["type"], target_kind, idx_da)

    cam_in = None
    if model_in is not None and (spec["type"] != "multilabel" or focus):
        with st.spinner("Computing GradCAM (ImageNet)..."):
            cam_in = gradcam(model_in, x, spec["type"], target_kind, idx_in)

    cam_h = Image.fromarray(cam_da) if cam_da is not None else None
    cam_in_h = Image.fromarray(cam_in) if cam_in is not None else None

    grid_panels = []
    grid_fell_back = False
    if model_in is None and spec["type"] == "multilabel" and not focus:
        with st.spinner("Computing biomarker GradCAMs..."):
            grid_panels, grid_fell_back = compute_multilabel_grid_cams(
                model_da, x, labels_da["biomarkers"], preds_da["probs"], threshold,
            )

    if model_in is None:
        if spec["type"] == "multilabel" and not focus:
            st.caption("One heatmap per biomarker the model considers present.")
            render_multilabel_gradcam_grid(grid_panels, grid_fell_back, threshold, st)
        else:
            st.image(_resize_to_height(cam_h, DISPLAY_HEIGHT), caption=f"GradCAM - {target_label_da}")
    else:
        st.caption(
            "Each model is explained on its own prediction, so DA and IN can light up different regions."
            if auto_target
            else "Both models are explained against the same target class."
        )
        row_l, row_r = st.columns(2)
        row_l.image(cam_h, caption=f"DA - {target_label_da}", width=GRADCAM_HERO_W)
        row_r.image(cam_in_h, caption=f"IN - {target_label_in}", width=GRADCAM_HERO_W)

    st.divider()
    try:
        pdf_bytes = build_pdf_report(
            image=image, source_name=source_name, expected_label=expected_label,
            spec=spec, task_key=task_key,
            labels_da=labels_da, preds_da=preds_da, cam_da=cam_da, target_label_da=target_label_da,
            labels_in=labels_in, preds_in=preds_in, cam_in=cam_in, target_label_in=target_label_in,
            threshold=threshold, grid_panels=grid_panels,
        )
        stem = os.path.splitext(source_name or "report")[0]
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name=f"retino_dino_{task_key}_{stem}.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.warning(f"PDF export failed: {e}")


def _render_landing():
    st.markdown("<div class='hero-title'>Retino-DINO</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-sub'>An AI second opinion for retinal OCT scans.</div>",
        unsafe_allow_html=True,
    )

    stats = [
        ("4", "Clinical tasks"),
        ("~40k", "Self-supervised images"),
        ("0.845", "OCTDL disease macro-F1"),
        ("22M", "Backbone parameters"),
    ]
    cards = "".join(
        f"<div class='stat-card' style='animation-delay:{0.05*i:.2f}s'>"
        f"<div class='stat-num'>{num}</div>"
        f"<div class='stat-label'>{label}</div></div>"
        for i, (num, label) in enumerate(stats)
    )
    st.markdown(f"<div class='stat-strip'>{cards}</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        "<div class='info-card' style='animation-delay:0.12s'><h4>What it does</h4>"
        "<p>Upload an eye scan and the model names the likely condition, then "
        "highlights the regions it looked at to decide.</p></div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        "<div class='info-card' style='animation-delay:0.20s'><h4>Who it is for</h4>"
        "<p>Two audiences, one tool. Clinicians use the Analyse tab as decision "
        "support; reviewers use the comparison, performance and latent-space tabs "
        "to weigh the evidence behind it.</p></div>",
        unsafe_allow_html=True,
    )
    c3.markdown(
        "<div class='info-card' style='animation-delay:0.28s'><h4>What is behind it</h4>"
        "<p>A vision model adapted on ~40k retinal images and fine-tuned on four "
        "clinical tasks, from broad disease screening to specific biomarkers.</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### How it works")
    st.markdown(
        "<span class='step-num'>1.</span> Pick a clinical task &nbsp;&nbsp; "
        "<span class='step-num'>2.</span> Upload a scan or choose a sample &nbsp;&nbsp; "
        "<span class='step-num'>3.</span> Read the prediction and the heatmap, then export a PDF.",
        unsafe_allow_html=True,
    )

    st.write("")
    start, _ = st.columns([1, 3])
    if start.button("Start analysing a scan", type="primary", use_container_width=True):
        st.session_state.app_started = True
        st.rerun()

    st.caption(
        "Evaluating the research? The Performance and Latent space tabs hold the "
        "thesis numbers and the domain-adapted vs ImageNet comparison."
    )


def _render_welcome(task_key, spec):
    info = DATASETS_INFO.get(task_key, {})
    st.markdown(f"### {spec['label']}")
    if info.get("use_for"):
        st.markdown(f"**When to use this**")
        st.markdown(info["use_for"])
    else:
        st.markdown(SIDEBAR_BLURBS.get(task_key, ""))
    _show_task_summary(spec)
    st.info("Upload an OCT image or pick a sample from the sidebar to run the model. Full dataset and evaluation detail is in the Task guide tab.")


def _render_datasets_tab(current_task=None):
    st.markdown("### Task guide")
    st.caption("Start here to choose the task that matches your clinical question. The table is the quick version; expand a task for the full dataset and evaluation detail.")

    st.markdown("#### Which task do I pick?")
    st.markdown(_md_table(
        ["Task", "Use it when you want to..."],
        [(f"`{key}`", DATASETS_INFO[key]["use_for"].replace("Use this when ", "").replace("Use this for ", "").rstrip("."))
         for key in DATASETS_INFO],
    ))

    st.markdown("#### Detail per task")
    for key, info in DATASETS_INFO.items():
        spec = TASKS[key]
        expanded = (key == current_task)
        header = f"{info['title']} - `{key}`"
        if expanded:
            header += " - currently selected"
        with st.expander(header, expanded=expanded):
            st.markdown(info["use_for"])
            for line in info["labels"]:
                st.markdown(f"- {line}")
            if st.checkbox("Show dataset & evaluation detail", key=f"detail_{key}"):
                st.markdown(info["what"])
                st.markdown(f"**Split:** {info['split']}")

    with st.expander("Self-supervised pre-training corpus (for all tasks)", expanded=False):
        st.markdown(PRETRAIN_BLURB)


def _render_latent_tab():
    st.markdown("### Latent space (interactive) - OCTDL only")
    st.caption("Each dot is one OCT scan from the OCTDL test set, projected to 2D with UMAP and coloured by disease class. Drag to pan, scroll to zoom, click a class in the legend to isolate it.")
    st.info("OCTDL-only: OCTDL is the only single-label task where colouring a UMAP by class is meaningful. For MMRDR, Corina and OCT5k, static t-SNE plots are in the Reports tab under the Latent space (t-SNE) category.")
    available = [(name, path) for name, path in LATENT_DIRS if os.path.isfile(path)]
    if not available:
        st.warning("No UMAP embeddings found under results/umap/.")
        return
    name_to_path = dict(available)
    pick = st.radio("Backbone", [n for n, _ in available], horizontal=True)
    payload = _load_latent(name_to_path[pick])
    st.altair_chart(_interactive_latent_chart(payload, title=pick),
                    use_container_width=True)
    st.caption(f"{len(payload['x'])} scans, {len(set(payload['label']))} classes. The domain-adapted backbone should pull same-colour scans into tighter clusters than the ImageNet baseline.")


def _render_performance_tab():
    """Static thesis numbers, grouped into per-dataset sub-tabs."""
    st.markdown("### Performance")
    st.caption("Numbers from the thesis report. Bold rows mark the domain-adapted result; the other rows are external papers or my own ImageNet baseline.")

    sub_overview, sub_pretrain, sub_octdl, sub_mmrdr, sub_corina, sub_oct5k = st.tabs(
        ["Overview", "SSL pretraining",
         "OCTDL", "MMRDR", "Corina", "OCT5k"]
    )

    with sub_overview:
        st.markdown("#### Headline numbers per task")
        st.markdown(_md_table(
            ["Task", "Metric", "DA value", "Compared to"],
            [
                ("OCTDL", "Disease F1 (test)", "0.845", "ImageNet baseline 0.800; old ViT-Large SSL 0.749"),
                ("OCTDL", "Condition F1 (test)", "0.820", "ImageNet baseline 0.741"),
                ("OCTDL", "Patient-level F1", "0.829", "Image-level 0.845 (avg-probs)"),
                ("MMRDR", "Macro F1", "0.775", "RETFound (SSL 1.6 M) 0.759"),
                ("MMRDR", "Cohen's kappa", "0.798", "RETFound 0.803"),
                ("MMRDR", "F1 on NCI-DME (rare)", "0.49", "RETFound 0.436, ResNet-50 0.259"),
                ("Corina", "Macro F1", "0.890", "ImageNet baseline 0.707; ConvNeXt-base 87 M ~0.86"),
                ("Corina", "Macro AUC", "0.970", "ImageNet baseline 0.930"),
                ("Corina", "Exact match (4/4)", "74.8 %", "ImageNet baseline 49.1 %"),
                ("OCT5k", "Macro F1", "0.567", "ImageNet baseline 0.469 (+0.098)"),
                ("OCT5k", "Macro AUC", "0.843", "ImageNet baseline 0.678 (+0.165)"),
                ("OCT5k", "Exact match", "30.8 %", "ImageNet baseline 13.1 %"),
            ],
        ))

    with sub_pretrain:
        st.markdown("#### Frozen-feature evaluation on OCTDL_CLEANED")
        st.caption("kNN (k=20) and linear probe at 224 px, train=1661 / test=403, patient-stratified. The four DA rows track pretraining quality over the 30-epoch SSL run; ImageNet and the old ViT-B/14 are baselines.")
        st.markdown(_md_table(PRETRAIN_FROZEN["header"], PRETRAIN_FROZEN["rows"]))
        st.caption("Linear-probe macro-F1 went up by +0.174 (+26 %) from frozen features alone. Largest per-class gains: RVO +0.46, NO +0.20, DME +0.19, VID +0.17.")

    with sub_octdl:
        st.markdown("#### Run comparison")
        st.markdown(_md_table(OCTDL_RUNS["header"], OCTDL_RUNS["rows"]))
        st.caption("Run C is the final thesis model: Disease acc 93.48 %, macro-F1 0.8446. Condition acc 84.20 %, macro-F1 0.8202. The ImageNet baseline (Run E) loses 0.044 disease F1 and 0.079 condition F1.")

        st.markdown("#### Patient-level evaluation")
        st.caption("Per-image predictions grouped by patient_id, then aggregated (majority vote or averaged probabilities). 165 test patients.")
        st.markdown(_md_table(OCTDL_PATIENT["header"], OCTDL_PATIENT["rows"]))

        st.markdown("#### Data-efficiency experiment")
        st.caption("Patient-level subsampling at 33 / 66 / 100 % of training data, with val and test fixed. DA already exceeds the 100 % ImageNet baseline at 33 % training data on disease F1.")
        st.markdown(_md_table(OCTDL_DATAEFF["header"], OCTDL_DATAEFF["rows"]))

    with sub_mmrdr:
        st.markdown("#### Cross-dataset transfer to MMRDR")
        st.caption("DME severity grading, 3 classes (No DME / NCI-DME / CI-DME). NCI-DME is severely under-represented (7.5 %) and is the hardest class. Patient-level split, train 2376 / test 562.")
        st.markdown(_md_table(MMRDR_COMPARE["header"], MMRDR_COMPARE["rows"]))
        st.caption("With roughly 40x less SSL pretraining data than RETFound, the DA ViT-S/14 matches or beats it on macro-F1 and on the hardest class (NCI-DME).")

    with sub_corina:
        st.markdown("#### Per-biomarker results, DA model")
        st.caption("Patient-level split: train 2414 (38 patients) / val 268 (5) / test 326 (9). No augmentation. BCEWithLogitsLoss with pos_weight = neg/pos per label, threshold 0.5.")
        st.markdown(_md_table(CORINA_PER_BIO["header"], CORINA_PER_BIO["rows"]))

        st.markdown("#### DA vs ImageNet vs ConvNeXt-base reference")
        st.markdown(_md_table(CORINA_VS["header"], CORINA_VS["rows"]))
        st.caption("Largest single-biomarker gap on HF (DA 0.92 vs IN 0.55). Multi-label exact-match jumped from 49.1 % to 74.8 %.")

    with sub_oct5k:
        st.markdown("#### Per-biomarker DA vs ImageNet")
        st.caption("AMD/DRUSEN patients only. Patient-level 80/10/10 split: train 399 (42 patients) / val 60 (6) / test 107 (12). Fluid dropped (<15 positives); 8 active biomarkers.")
        st.markdown(_md_table(OCT5K_PER_BIO["header"], OCT5K_PER_BIO["rows"]))
        st.caption("Largest gains on the morphologically subtle features (RD +0.19, HD +0.18). Small dataset limits absolute scores, so the DA vs IN delta is what matters here.")


if __name__ == "__main__":
    main()