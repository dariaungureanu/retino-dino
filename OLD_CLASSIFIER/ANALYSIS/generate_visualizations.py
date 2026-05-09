import os
import sys
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from OLD_CLASSIFIER.dataset import get_data_splits, OCTDLMultiTaskDataset
from OLD_CLASSIFIER.model import OCTDLMultiTaskModel
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(current_dir)
output_dir = os.path.join(current_dir, "visualizations")
os.makedirs(output_dir, exist_ok=True)

CLASSIFIER_DIR = os.path.join(project_root, "OLD_CLASSIFIER")

DATASET_PATH = r"C:\Datasets\OCTDL_Cleaned"
MODEL_PATH = os.path.join(project_root, "saved_models", "best_classifier_unfrozen.pth")
#MODEL_PATH = os.path.join(project_root, "saved_models", "best_classifier.pth")
SSL_CHECKPOINT_PATH = os.path.join(project_root, "checkpoints_ssl", "checkpoint_latest.pth")

if CLASSIFIER_DIR not in sys.path:
    sys.path.insert(0, CLASSIFIER_DIR)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")


def load_resources():
    csv_path = os.path.join(DATASET_PATH, "OCTDL_clean_metadata.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata not found at: {csv_path}")

    _, _, test_df, disease_map, condition_map = get_data_splits(csv_path)
    idx_to_disease = {v: k for k, v in disease_map.items()}
    idx_to_condition = {v: k for k, v in condition_map.items()}

    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_ds = OCTDLMultiTaskDataset(test_df, DATASET_PATH, val_transform, disease_map, condition_map)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    init_checkpoint = SSL_CHECKPOINT_PATH
    if not os.path.exists(init_checkpoint):
        print(f" Warning: Can't find {SSL_CHECKPOINT_PATH}")
        init_checkpoint = MODEL_PATH

    print("Initializing model structure...")
    model = OCTDLMultiTaskModel(
        checkpoint_path=init_checkpoint,
        num_diseases=len(disease_map),
        num_conditions=len(condition_map),
        unfreeze_last_block=True
    )

    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    model.to(DEVICE)
    model.eval()

    return model, loader, idx_to_disease, idx_to_condition, test_ds, disease_map, condition_map


def plot_tsne(model, loader, idx_to_disease):
    tsne_save_path = os.path.join(output_dir, "tsne_plot.png")
    if os.path.exists(tsne_save_path):
        print("t-SNE already exists. Skipping...")
        return

    print("Extracting features for t-SNE...")
    features = []
    labels = []

    with torch.no_grad():
        for images, d_labels, _ in tqdm(loader):
            images = images.to(DEVICE)
            feats = model.backbone(images)
            features.append(feats.cpu().numpy())
            labels.extend(d_labels.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.array(labels)

    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    palette = sns.color_palette("bright", len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        indices = labels == lbl
        plt.scatter(
            features_2d[indices, 0],
            features_2d[indices, 1],
            label=idx_to_disease[lbl],
            color=palette[i],
            alpha=0.7,
            s=40,
        )

    plt.title("t-SNE Visualization of Disease Latent Space", fontsize=16)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Disease")
    plt.tight_layout()

    plt.savefig(tsne_save_path, dpi=300)
    plt.close()
    print(f"t-SNE plot saved to: {tsne_save_path}")


class MultiTaskWrapper(torch.nn.Module):
    def __init__(self, model, target_index):
        super().__init__()
        self.model = model
        self.target_index = target_index

    def forward(self, x):
        outputs = self.model(x)
        return outputs[self.target_index]  # logits for that head


def reshape_transform(tensor):
    result = tensor[:, 1:, :]  # drop CLS
    grid_size = int(np.sqrt(result.size(1)))
    result = result.reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def denormalize_to_rgb(img_tensor_1x3xhxw):
    """Undo ImageNet normalization for visualization."""
    img = img_tensor_1x3xhxw.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    return img


def collect_predictions(model, loader, device, task_head=0):
    model.eval()
    y_true, y_pred, y_conf, sample_indices = [], [], [], []

    running_idx = 0
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for images, d_labels, c_labels in tqdm(loader, desc=f"Predict head={task_head}"):
            images = images.to(device)
            outputs = model(images)
            logits = outputs[task_head]  # [B, num_classes]

            probs = softmax(logits)
            conf, pred = torch.max(probs, dim=1)

            labels = d_labels if task_head == 0 else c_labels

            bsz = images.size(0)
            batch_indices = list(range(running_idx, running_idx + bsz))
            running_idx += bsz

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            y_conf.extend(conf.cpu().numpy().tolist())
            sample_indices.extend(batch_indices)

    return np.array(y_true), np.array(y_pred), np.array(y_conf), np.array(sample_indices)


def select_top_wrong(y_true, y_pred, y_conf, idxs, topk=8):
    wrong_mask = y_true != y_pred
    wrong = idxs[wrong_mask]
    wrong_conf = y_conf[wrong_mask]
    if len(wrong) == 0:
        return np.array([], dtype=int)
    order = np.argsort(-wrong_conf)
    return wrong[order[:topk]]


def select_wrong_from_class(y_true, y_pred, y_conf, idxs, class_idx, topk=8):
    mask = (y_true == class_idx) & (y_pred != class_idx)
    wrong = idxs[mask]
    conf = y_conf[mask]
    if len(wrong) == 0:
        return np.array([], dtype=int)
    order = np.argsort(-conf)
    return wrong[order[:topk]]


def generate_cam_for_task_compare_pred_true(
    model,
    dataset,
    indices,
    task_name,
    head_index,
    label_map,
    save_name,
):
    """
    For each sample:
      - Original
      - CAM for predicted class
      - CAM for true class
    """
    print(f"Generating GradCAM (pred vs true) for task: {task_name}...")

    for p in model.backbone.parameters():
        p.requires_grad = True

    wrapper = MultiTaskWrapper(model, head_index)
    target_layers = [model.backbone.blocks[-1].norm1]
    cam = GradCAM(model=wrapper, target_layers=target_layers, reshape_transform=reshape_transform)

    num_samples = len(indices)
    if num_samples == 0:
        print("No indices given. Skipping.")
        return

    plt.figure(figsize=(14, 4 * num_samples))

    softmax = torch.nn.Softmax(dim=1)

    for i, idx in enumerate(indices):
        img_tensor, label_d, label_c = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        current_label = label_d if head_index == 0 else label_c
        label_val = int(current_label.item() if isinstance(current_label, torch.Tensor) else current_label)
        if label_val == -100:
            print(f"Skipping sample {idx} (ignore label -100)")
            continue

        with torch.no_grad():
            logits = wrapper(img_tensor)
            probs = softmax(logits)
            pred_class = int(torch.argmax(probs, dim=1).item())
            pred_conf = float(torch.max(probs, dim=1).values.item())

        true_name = label_map.get(label_val, "Unknown/Ignore")
        pred_name = label_map.get(pred_class, "Unknown")

        rgb_img = denormalize_to_rgb(img_tensor)

        cam_pred = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(pred_class)])[0, :]
        cam_true = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(label_val)])[0, :]

        vis_pred = show_cam_on_image(rgb_img, cam_pred, use_rgb=True)
        vis_true = show_cam_on_image(rgb_img, cam_true, use_rgb=True)

        row = i + 1

        plt.subplot(num_samples, 3, (row - 1) * 3 + 1)
        plt.imshow(rgb_img)
        plt.title(f"[{task_name}] TRUE: {true_name}\nPRED: {pred_name} (conf={pred_conf:.2f})")
        plt.axis("off")

        plt.subplot(num_samples, 3, (row - 1) * 3 + 2)
        plt.imshow(vis_pred)
        plt.title(f"CAM for PRED: {pred_name}")
        plt.axis("off")

        plt.subplot(num_samples, 3, (row - 1) * 3 + 3)
        plt.imshow(vis_true)
        plt.title(f"CAM for TRUE: {true_name}")
        plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_gradcam_on_errors(model, dataset, loader, idx_to_disease, idx_to_condition, task_head=0, topk=8):
    """
    Generates GradCAM only for WRONG predictions (highest confidence wrong).
    """
    y_true, y_pred, y_conf, idxs = collect_predictions(model, loader, DEVICE, task_head=task_head)
    selected = select_top_wrong(y_true, y_pred, y_conf, idxs, topk=topk)

    label_map = idx_to_disease if task_head == 0 else idx_to_condition
    task_name = "Disease" if task_head == 0 else "Condition"
    save_name = f"gradcam_{task_name.lower()}_errors_top{topk}.png"

    if len(selected) == 0:
        print(f"No errors for {task_name}. Nothing to plot.")
        return

    generate_cam_for_task_compare_pred_true(
        model=model,
        dataset=dataset,
        indices=selected,
        task_name=task_name,
        head_index=task_head,
        label_map=label_map,
        save_name=save_name,
    )


def plot_gradcam_on_class_errors(
    model,
    dataset,
    loader,
    idx_to_disease,
    idx_to_condition,
    class_name,
    disease_map,
    condition_map,
    task_head=0,
    topk=8,
):
    """
    Plot GradCAM for errors where TRUE == class_name but predicted != class_name.
    Works for disease or condition depending on task_head.
    """
    y_true, y_pred, y_conf, idxs = collect_predictions(model, loader, DEVICE, task_head=task_head)

    if task_head == 0:
        class_idx = disease_map[class_name]
        label_map = idx_to_disease
        task_name = "Disease"
    else:
        class_idx = condition_map[class_name]
        label_map = idx_to_condition
        task_name = "Condition"

    selected = select_wrong_from_class(y_true, y_pred, y_conf, idxs, class_idx=class_idx, topk=topk)
    save_name = f"gradcam_{task_name.lower()}_errors_true_{class_name}_top{topk}.png"

    if len(selected) == 0:
        print(f"No errors found for TRUE={class_name} ({task_name}).")
        return

    generate_cam_for_task_compare_pred_true(
        model=model,
        dataset=dataset,
        indices=selected,
        task_name=task_name,
        head_index=task_head,
        label_map=label_map,
        save_name=save_name,
    )


if __name__ == "__main__":
    model, test_loader, idx_to_disease, idx_to_condition, test_dataset, disease_map, condition_map = load_resources()

    plot_tsne(model, test_loader, idx_to_disease)

    plot_gradcam_on_errors(model, test_dataset, test_loader, idx_to_disease, idx_to_condition, task_head=0, topk=8)
    plot_gradcam_on_errors(model, test_dataset, test_loader, idx_to_disease, idx_to_condition, task_head=1, topk=8)

    plot_gradcam_on_class_errors(model, test_dataset, test_loader,
                                idx_to_disease, idx_to_condition,
                                class_name="AMD", disease_map=disease_map, condition_map=condition_map,
                                task_head=0, topk=8)

    print(f"Done. Check folder: {output_dir}")
