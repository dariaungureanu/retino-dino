import os
import sys
import torch
import numpy as np
from PIL import Image
import streamlit as st
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

current_dir = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_DIR = os.path.join(current_dir, 'CLASSIFIER')

if CLASSIFIER_DIR not in sys.path:
    sys.path.insert(0, CLASSIFIER_DIR)

try:
    from model import OCTDLMultiTaskModel
except ImportError as e:
    st.error(f"Import error: {e}. Check the CLASSIFIER folder.")
    st.stop()

MODEL_PATH = os.path.join(current_dir, "OLD_PRETRAIN/saved_models", "best_classifier_unfrozen.pth")
# MODEL_PATH = os.path.join(current_dir, "saved_models", "best_classifier.pth")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DISEASE_MAP = {'AMD': 0, 'DME': 1, 'ERM': 2, 'NO': 3, 'RAO': 4, 'RVO': 5, 'VID': 6}
CONDITION_MAP = {'DRIL': 0, 'ERM': 1, 'ME': 2, 'MH': 3, 'MNV': 4, 'MNV_suspected': 5, 'NO': 6, 'drusen': 7}

IDX_TO_DISEASE = {v: k for k, v in DISEASE_MAP.items()}
IDX_TO_CONDITION = {v: k for k, v in CONDITION_MAP.items()}

class MultiTaskWrapper(torch.nn.Module):
    def __init__(self, model, target_index):
        super().__init__()
        self.model = model
        self.target_index = target_index

    def forward(self, x):
        return self.model(x)[self.target_index]

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :]
    grid_size = int(np.sqrt(result.size(1)))
    result = result.reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at: {MODEL_PATH}")
        st.stop()
    model = OCTDLMultiTaskModel(
        checkpoint_path=MODEL_PATH,
        num_diseases=len(DISEASE_MAP),
        num_conditions=len(CONDITION_MAP),
        unfreeze_last_block=True
    )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    for param in model.backbone.parameters():
        param.requires_grad = True

    return model

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def generate_cam(model, input_tensor, head_index):
    wrapper = MultiTaskWrapper(model, head_index)
    target_layers = [model.backbone.blocks[-1].norm1]
    cam = GradCAM(model=wrapper, target_layers=target_layers, reshape_transform=reshape_transform)

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    rgb_img = input_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    rgb_img = (rgb_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    rgb_img = np.clip(rgb_img, 0, 1)

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization

def main():
    st.set_page_config(page_title="OCTDL Screening", layout="wide")

    st.title("Retino-Dino: OCTDL Multi-Task AI")
    st.markdown("Upload an OCT image to detect the primary disease and retinal conditions (biomarkers).")

    with st.spinner("Loading DINOv2 model..."):
        model = load_model()

    uploaded_file = st.file_uploader("Choose an OCT image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        input_tensor = process_image(image)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with torch.no_grad():
            logits_d, logits_c = model(input_tensor)

            prob_d = torch.nn.functional.softmax(logits_d, dim=1)[0]
            prob_c = torch.nn.functional.softmax(logits_c, dim=1)[0]

            pred_d_idx = torch.argmax(prob_d).item()
            pred_c_idx = torch.argmax(prob_c).item()

            pred_disease = IDX_TO_DISEASE[pred_d_idx]
            pred_condition = IDX_TO_CONDITION[pred_c_idx]
            conf_d = prob_d[pred_d_idx].item() * 100
            conf_c = prob_c[pred_c_idx].item() * 100

        st.markdown("---")
        st.subheader("Prediction Results")
        st.write(f"**Disease:** {pred_disease} (Confidence: {conf_d:.2f}%)")
        st.write(f"**Condition:** {pred_condition} (Confidence: {conf_c:.2f}%)")

        with st.spinner("Generating medical attention maps (XAI)..."):
            cam_disease = generate_cam(model, input_tensor, 0)
            cam_condition = generate_cam(model, input_tensor, 1)

        with col2:
            st.subheader(" Disease Attention")
            st.image(cam_disease, use_container_width=True)
            st.caption("Where the model looks to classify the disease.")

        with col3:
            st.subheader(" Condition Attention")
            st.image(cam_condition, use_container_width=True)
            st.caption("Where the model looks to detect the condition.")

if __name__ == "__main__":
    main()