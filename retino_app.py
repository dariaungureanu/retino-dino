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
    """Load DINOv2 from hub, then overwrite with SSL teacher weights."""
    model = _build_dinov2()

    if weights_path is None: #skip this step -> keep ImageNet weights
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
@st.cache_resource(show_spinner=False) # the model is only loaded once,
def get_model(task_key: str, variant: str):
    spec = TASKS[task_key]
    ckpt_name = spec["ckpt_da"] if variant == "da" else spec["ckpt_in"]
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    #Load the SSL backbone or hub ImageNet
    backbone_weights = SSL_BACKBONE if variant == "da" else None
    backbone = load_ssl_backbone(backbone_weights)

    if spec["type"] == "octdl":
        d_map = ckpt.get("disease_map", {n: i for i, n in enumerate(spec["disease"])})
        c_map = ckpt.get("condition_map", {n: i for i, n in enumerate(spec["condition"])})
        model = OCTDLModel(backbone, len(d_map), len(c_map))
    elif spec["type"] == "single":
        n = ckpt.get("num_classes", len(spec["classes"]))
        model = SingleHeadModel(backbone, n)
    else:
        n = ckpt.get("num_labels", len(spec["biomarkers"]))
        model = SingleHeadModel(backbone, n)

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # report if keys dont match (head missing means the checkpoint is not compatible with the model architecture)
    head_missing = [k for k in missing if "head" in k]
    backbone_missing = [k for k in missing if "head" not in k]
    head_unexpected = [k for k in unexpected if "head" in k]

    print(f"\n=== {task_key}/{variant} checkpoint diagnostic ===", file=sys.stderr)
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
    return model


# Inference helpers
def to_tensor(image: Image.Image) -> torch.Tensor:
    return EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)


def denormalize(t: torch.Tensor) -> np.ndarray:
    img = t.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = img * np.array(IMG_STD) + np.array(IMG_MEAN)
    return np.clip(img, 0.0, 1.0)


def predict(model, x, task_type):
    """
      Runs the model and converts logits to probabilities:
      - Softmax for OCTDL/MMRDR (mutually exclusive classes - they sum to 1).
      - Sigmoid for Corina/OCT5k (independent biomarkers - each one is "yes/no" on its
      own).
    """
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
    """Wraps the model in _HeadSelector so GradCAM sees a single output."""
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


def _truncate(label, n=14):
    return label if len(label) <= n else label[: n - 1] + "..."


def _bar_chart(probs, classes, predicted_idx, title=""):
    """
    A horizontal bar chart for single-label outputs (softmax).
      - All bars are blue, except the predicted one is red
      - X-axis is locked to 0–100 (probabilities as %).
      - Each bar has its number labeled to the right
    """
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
    """
      For multi-label outputs (sigmoid):
          - Bars are green if prob ≥ 0.5 (predicted "present"), grey otherwise.
          - A red dashed vertical line marks the 0.5 decision threshold.
          - The model can predict any number of biomarkers (zero, one, several).
    """
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


def render_predictions(spec, preds, container, header):
    """
      The dispatcher. Reads spec["type"] and renders the right format:
      - "octdl" - two bar charts (one per head) + two argmax summaries.
      - "single" - one bar chart + the predicted class.
      - "multilabel" - the panel + a "Active biomarkers (≥0.5): X, Y" line.
    """
    container.markdown(f"#### {header}")
    if spec["type"] == "octdl":
        d_idx = int(np.argmax(preds["disease"]))
        c_idx = int(np.argmax(preds["condition"]))
        d_conf = float(preds["disease"][d_idx])
        c_conf = float(preds["condition"][c_idx])
        container.markdown(
            f"**Disease:** `{spec['disease'][d_idx]}` ({d_conf*100:.1f}%)"
        )
        container.markdown(
            f"**Condition:** `{spec['condition'][c_idx]}` ({c_conf*100:.1f}%)"
        )
        container.pyplot(_bar_chart(preds["disease"], spec["disease"], d_idx,
                                    "Disease head"), clear_figure=True)
        _confidence_caption(container, d_conf, len(spec["disease"]))
        container.pyplot(_bar_chart(preds["condition"], spec["condition"], c_idx,
                                    "Condition head"), clear_figure=True)
        _confidence_caption(container, c_conf, len(spec["condition"]))
    elif spec["type"] == "single":
        idx = int(np.argmax(preds["probs"]))
        conf = float(preds["probs"][idx])
        container.markdown(
            f"**Prediction:** `{spec['classes'][idx]}` ({conf*100:.1f}%)"
        )
        container.pyplot(_bar_chart(preds["probs"], spec["classes"], idx),
                         clear_figure=True)
        _confidence_caption(container, conf, len(spec["classes"]))
    else:
        labels = spec["biomarkers"]
        active = [labels[i] for i, p in enumerate(preds["probs"]) if p >= 0.5]
        container.markdown(
            f"**Active biomarkers (≥0.5):** "
            + (", ".join(f"`{a}`" for a in active) if active else "_none_")
        )
        container.pyplot(_multilabel_panel(preds["probs"], labels), clear_figure=True)


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
    """  Extracts the expected label from the filename so the user can sanity-check
    predictions."""
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

        #File uploader, accepts standard image formats
        st.divider()
        st.markdown("**Image source**")
        uploaded = st.file_uploader(
            "Upload an OCT image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        )

        #Sample dropdown, the "(none)" sentinel keeps the dropdown valid when the user wants to upload instead
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

    tab_predict, tab_reports = st.tabs(["Predict", "Reports"])

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
                model_da = get_model(task_key, "da")
            except Exception as e:
                st.error(f"Failed to load DA checkpoint: {e}")
                return

        model_in = None
        if compare:
            with st.spinner("Loading ImageNet baseline..."):
                try:
                    model_in = get_model(task_key, "in")
                except Exception as e:
                    st.warning(f"ImageNet checkpoint not available: {e}")

        #GradCAM target picker
        st.markdown("### Inputs")
        c_orig, c_ctrl = st.columns([2, 3])
        with c_orig:
            cap = source_name or "Uploaded image"
            st.image(image, caption=cap, use_container_width=True)
            if expected_label:
                st.caption(f"Expected (from filename): **{expected_label}**")

        with c_ctrl:
            target_kind, target_idx, target_label = _target_picker(spec)
            st.caption(
                "GradCAM highlights the regions that most influenced the "
                f"selected output: **{target_label}**."
            )

        #Run predictions + GradCAM
        preds_da = predict(model_da, x, spec["type"])
        with st.spinner("Computing GradCAM (DA)..."):
            cam_da = gradcam(model_da, x, spec["type"], target_kind, target_idx)

        preds_in = cam_in = None
        if model_in is not None:
            preds_in = predict(model_in, x, spec["type"])
            with st.spinner("Computing GradCAM (ImageNet)..."):
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

    #Reports tab
    with tab_reports:
        st.markdown(f"### Pre-generated reports - {task_key}")
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


if __name__ == "__main__":
    main()