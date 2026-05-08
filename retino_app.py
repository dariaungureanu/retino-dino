"""
Retino-DINO — OCT analysis demo app.

DINOv2 ViT-S/14 (domain-adapted on 40k OCT images via SSL) + per-task heads.
Four fine-tuned checkpoints:
    OCTDL    multitask   (Disease 7-class + Condition 8-class)
    MMRDR    single-task  (DME severity, 3-class)
    Corina   multi-label  (4 biomarkers)
    OCT5k    multi-label  (8 biomarkers)

Run:  streamlit run retino_app.py
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


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(ROOT, "checkpoints")
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


# ─────────────────────────────────────────────────────────────────
# Backbone loader (mirrors finetune_*/model.py::load_backbone but
# always points at the local SSL checkpoint and without train prints)
# ─────────────────────────────────────────────────────────────────
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
    """Load DINOv2 from hub, then overwrite with SSL teacher weights."""
    model = _build_dinov2()

    if weights_path is None:
        return model

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"SSL backbone weights not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        st_dict = ckpt["model"]
    elif "teacher" in ckpt:
        st_dict = ckpt["teacher"]
    elif "state_dict" in ckpt:
        st_dict = ckpt["state_dict"]
    else:
        st_dict = ckpt

    model_keys = set(model.state_dict().keys())
    PREFIXES = ["teacher.backbone.", "backbone.", "module.backbone.", "module.", ""]
    clean = {}
    for prefix in PREFIXES:
        cand = {}
        for k, v in st_dict.items():
            if prefix and k.startswith(prefix):
                cand[k[len(prefix):]] = v
            elif prefix == "" and not any(k.startswith(p) for p in
                ["dino_loss", "ibot_patch_loss", "dino_head", "ibot_head",
                 "student.", "teacher."]):
                cand[k] = v
        if cand:
            overlap = set(cand.keys()) & model_keys
            if len(overlap) > len(model_keys) * 0.5:
                clean = cand
                break
    if not clean:
        for k, v in st_dict.items():
            for prefix in ("teacher.backbone.", "student.backbone.", "backbone."):
                if k.startswith(prefix):
                    clean[k[len(prefix):]] = v
                    break

    _interpolate_pos_embed(clean, model)
    model.load_state_dict(clean, strict=False)
    return model


# ─────────────────────────────────────────────────────────────────
# Heads & models — mirror finetune_*/model.py (state_dict keys must
# match the saved checkpoints exactly).
# ─────────────────────────────────────────────────────────────────
def _make_head(in_dim, hidden, out_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class OCTDLModel(nn.Module):
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


# ─────────────────────────────────────────────────────────────────
# GradCAM wrappers
# ─────────────────────────────────────────────────────────────────
def vit_reshape_transform(tensor):
    """Drop CLS token and reshape (B, 1+N, D) → (B, D, H, W) for ViT."""
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


# ─────────────────────────────────────────────────────────────────
# Task registry
# ─────────────────────────────────────────────────────────────────
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
        "disease": OCTDL_DISEASE,
        "condition": OCTDL_CONDITION,
        "gallery_dirs": [
            _path("finetune_octdl", "results", "confusion_matrices", "run_C_unfreeze2"),
            _path("finetune_octdl", "results", "explainability", "run_finetuning_C"),
        ],
    },
    "MMRDR": {
        "label": "MMRDR  ·  DME Severity (3-class)",
        "type": "single",
        "ckpt_da": "mmrdr_da.pth",
        "ckpt_in": "mmrd_in.pth",
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
        "biomarkers": OCT5K_LABELS,
        "gallery_dirs": [
            _path("finetune_oct5k", "results"),
        ],
    },
}


# ─────────────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(task_key: str, variant: str):
    """variant ∈ {'da', 'in'}. Returns model in eval mode, on CPU."""
    spec = TASKS[task_key]
    ckpt_name = spec["ckpt_da"] if variant == "da" else spec["ckpt_in"]
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Always init backbone with our local SSL weights for DA, hub-default for IN.
    backbone_weights = SSL_BACKBONE if variant == "da" else None
    backbone = load_ssl_backbone(backbone_weights)

    if spec["type"] == "octdl":
        d_map = ckpt.get("disease_map", {n: i for i, n in enumerate(spec["disease"])})
        c_map = ckpt.get("condition_map", {n: i for i, n in enumerate(spec["condition"])})
        model = OCTDLModel(backbone, len(d_map), len(c_map))
    elif spec["type"] == "single":
        n = ckpt.get("num_classes", len(spec["classes"]))
        model = SingleHeadModel(backbone, n)
    else:  # multilabel
        n = ckpt.get("num_labels", len(spec["biomarkers"]))
        model = SingleHeadModel(backbone, n)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"[{task_key}/{variant}] missing keys (first 3): {list(missing)[:3]}")
    if unexpected:
        print(f"[{task_key}/{variant}] unexpected keys (first 3): {list(unexpected)[:3]}")

    model.to(DEVICE).eval()
    # GradCAM needs gradients to flow through the target layer.
    for p in model.parameters():
        p.requires_grad_(True)
    return model


# ─────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────
def to_tensor(image: Image.Image) -> torch.Tensor:
    return EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)


def denormalize(t: torch.Tensor) -> np.ndarray:
    img = t.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = img * np.array(IMG_STD) + np.array(IMG_MEAN)
    return np.clip(img, 0.0, 1.0)


def predict(model, x, task_type):
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
    """target_kind: 'disease' | 'condition' (OCTDL) or None."""
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


# ─────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────
def _bar_chart(probs, classes, predicted_idx, title=""):
    fig, ax = plt.subplots(figsize=(4.5, 0.4 * len(classes) + 0.6))
    colors = ["#1f77b4"] * len(classes)
    colors[predicted_idx] = "#d62728"
    y = np.arange(len(classes))
    ax.barh(y, probs * 100.0, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("probability (%)", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    for i, p in enumerate(probs * 100.0):
        ax.text(p + 1.5, i, f"{p:.1f}", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _multilabel_panel(probs, labels, threshold=0.5, title=""):
    fig, ax = plt.subplots(figsize=(4.5, 0.4 * len(labels) + 0.6))
    colors = ["#2ca02c" if p >= threshold else "#9aa0a6" for p in probs]
    y = np.arange(len(labels))
    ax.barh(y, probs * 100.0, color=colors, edgecolor="none")
    ax.axvline(threshold * 100, color="#d62728", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("sigmoid probability (%)", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    for i, p in enumerate(probs * 100.0):
        ax.text(p + 1.5, i, f"{p:.1f}", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def render_predictions(spec, preds, container, header):
    container.markdown(f"#### {header}")
    if spec["type"] == "octdl":
        d_idx = int(np.argmax(preds["disease"]))
        c_idx = int(np.argmax(preds["condition"]))
        container.markdown(
            f"**Disease:** `{spec['disease'][d_idx]}` "
            f"({preds['disease'][d_idx]*100:.1f}%)"
        )
        container.markdown(
            f"**Condition:** `{spec['condition'][c_idx]}` "
            f"({preds['condition'][c_idx]*100:.1f}%)"
        )
        container.pyplot(_bar_chart(preds["disease"], spec["disease"], d_idx,
                                    "Disease head"), clear_figure=True)
        container.pyplot(_bar_chart(preds["condition"], spec["condition"], c_idx,
                                    "Condition head"), clear_figure=True)
    elif spec["type"] == "single":
        idx = int(np.argmax(preds["probs"]))
        container.markdown(
            f"**Prediction:** `{spec['classes'][idx]}` "
            f"({preds['probs'][idx]*100:.1f}%)"
        )
        container.pyplot(_bar_chart(preds["probs"], spec["classes"], idx),
                         clear_figure=True)
    else:
        labels = spec["biomarkers"]
        active = [labels[i] for i, p in enumerate(preds["probs"]) if p >= 0.5]
        container.markdown(
            f"**Active biomarkers (≥0.5):** "
            + (", ".join(f"`{a}`" for a in active) if active else "_none_")
        )
        container.pyplot(_multilabel_panel(preds["probs"], labels), clear_figure=True)


def gallery_files(spec):
    """Collect PNG/JPGs from the task's results dirs (recursive)."""
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
    # de-dup, preserve order
    seen = set()
    out = []
    for rel, full in files:
        if full in seen:
            continue
        seen.add(full)
        out.append((rel, full))
    return out


# ─────────────────────────────────────────────────────────────────
# Streamlit app
# ─────────────────────────────────────────────────────────────────
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

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")

        task_key = st.radio(
            "Task",
            list(TASKS.keys()),
            format_func=lambda k: TASKS[k]["label"],
        )
        spec = TASKS[task_key]

        compare = st.checkbox(
            "Compare against ImageNet baseline",
            value=False,
            help="Run the same image through the ImageNet-only fine-tuned model "
                 "and show the predictions side by side.",
        )

        st.divider()
        uploaded = st.file_uploader(
            "Upload an OCT image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        )

        st.divider()
        st.caption(
            f"Backbone: `{ARCH}` (384-dim, 12 blocks)  \n"
            f"Input: {IMG_SIZE}×{IMG_SIZE}, ImageNet-norm  \n"
            f"Device: `{DEVICE.type}`"
        )

    tab_predict, tab_reports = st.tabs(["Predict", "Reports"])

    # ── Predict tab ────────────────────────────────────────────
    with tab_predict:
        if uploaded is None:
            st.info("Upload an OCT image from the sidebar to run inference.")
            _show_task_summary(spec)
            return

        try:
            image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        except Exception as e:
            st.error(f"Could not read image: {e}")
            return

        x = to_tensor(image)

        with st.spinner("Loading domain-adapted model…"):
            try:
                model_da = get_model(task_key, "da")
            except Exception as e:
                st.error(f"Failed to load DA checkpoint: {e}")
                return

        model_in = None
        if compare:
            with st.spinner("Loading ImageNet baseline…"):
                try:
                    model_in = get_model(task_key, "in")
                except Exception as e:
                    st.warning(f"ImageNet checkpoint not available: {e}")

        # GradCAM target picker
        st.markdown("### Inputs")
        c_orig, c_ctrl = st.columns([2, 3])
        with c_orig:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with c_ctrl:
            target_kind, target_idx, target_label = _target_picker(spec)
            st.caption(
                "GradCAM highlights the regions that most influenced the "
                f"selected output: **{target_label}**."
            )

        # Run predictions + GradCAM
        preds_da = predict(model_da, x, spec["type"])
        with st.spinner("Computing GradCAM (DA)…"):
            cam_da = gradcam(model_da, x, spec["type"], target_kind, target_idx)

        preds_in = cam_in = None
        if model_in is not None:
            preds_in = predict(model_in, x, spec["type"])
            with st.spinner("Computing GradCAM (ImageNet)…"):
                cam_in = gradcam(model_in, x, spec["type"], target_kind, target_idx)

        st.markdown("### Results")
        if model_in is None:
            col_cam, col_pred = st.columns([1, 1])
            with col_cam:
                st.image(cam_da, caption=f"GradCAM · {target_label}",
                         use_container_width=True)
            render_predictions(spec, preds_da, col_pred,
                               header="Predictions  ·  domain-adapted")
        else:
            cda, cin = st.columns(2)
            with cda:
                st.markdown("##### Domain-adapted")
                st.image(cam_da, use_container_width=True)
                render_predictions(spec, preds_da, st, header="")
            with cin:
                st.markdown("##### ImageNet baseline")
                st.image(cam_in, use_container_width=True)
                render_predictions(spec, preds_in, st, header="")

    # ── Reports tab ────────────────────────────────────────────
    with tab_reports:
        st.markdown(f"### Pre-generated reports — {task_key}")
        files = gallery_files(spec)
        if not files:
            st.info("No report images found in the task's results folders yet.")
            return

        names = [rel for rel, _ in files]
        choice = st.selectbox("Pick a figure", names, index=0)
        chosen_full = dict(files)[choice]
        st.image(chosen_full, caption=choice, use_container_width=True)

        with st.expander(f"All figures ({len(files)})", expanded=False):
            cols = st.columns(3)
            for i, (rel, full) in enumerate(files):
                with cols[i % 3]:
                    st.image(full, caption=rel, use_container_width=True)


def _target_picker(spec):
    """Sidebar-less target selector. Returns (kind, idx, label)."""
    if spec["type"] == "octdl":
        kind = st.radio("GradCAM head", ["disease", "condition"],
                        horizontal=True, format_func=str.capitalize)
        classes = spec["disease"] if kind == "disease" else spec["condition"]
        idx = st.selectbox(f"{kind.capitalize()} class", range(len(classes)),
                           format_func=lambda i: classes[i])
        return kind, idx, f"{kind} → {classes[idx]}"
    if spec["type"] == "single":
        classes = spec["classes"]
        idx = st.selectbox("Target class", range(len(classes)),
                           format_func=lambda i: classes[i])
        return None, idx, classes[idx]
    labels = spec["biomarkers"]
    idx = st.selectbox("Target biomarker", range(len(labels)),
                       format_func=lambda i: labels[i])
    return None, idx, labels[idx]


def _show_task_summary(spec):
    st.markdown("### Task")
    st.markdown(f"**{spec['label']}**")
    if spec["type"] == "octdl":
        st.markdown(
            f"- Disease head — {len(spec['disease'])} classes: "
            f"{', '.join(spec['disease'])}"
        )
        st.markdown(
            f"- Condition head — {len(spec['condition'])} classes: "
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


if __name__ == "__main__":
    main()
