import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import sys
from OLD_CLASSIFIER.dataset import get_data_splits, OCTDLMultiTaskDataset
from OLD_CLASSIFIER.model import OCTDLMultiTaskModel

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
classifier_dir = os.path.join(project_root, 'OLD_CLASSIFIER')

DEFAULT_DATA_PATH = r"C:\Datasets\OCTDL_Cleaned"
RESULT_DIR = os.path.join(project_root, "results_analysis")
DEFAULT_MODEL_PATH = os.path.join(project_root, "saved_models", "best_classifier_unfrozen.pth")
# DEFAULT_MODEL_PATH = os.path.join(project_root, "saved_models", "best_classifier.pth")
SSL_CHECKPOINT_PATH = os.path.join(project_root, "checkpoints_ssl", "checkpoint_latest.pth")

if not os.path.exists(classifier_dir):
    print(f"ERROR: Can't find folder {classifier_dir}")
    sys.exit(1)

if classifier_dir not in sys.path:
    sys.path.insert(0, classifier_dir)


def plot_confusion_matrix(cm, class_names, title, save_path):
    """Draw and save a confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" Saved Matrix: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f" Device: {args.device}")

    csv_path = os.path.join(args.data_path, "OCTDL_clean_metadata.csv")
    _, _, test_df, disease_map, condition_map = get_data_splits(csv_path)

    idx_to_disease = {v: k for k, v in disease_map.items()}
    idx_to_condition = {v: k for k, v in condition_map.items()}

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = OCTDLMultiTaskDataset(test_df, args.data_path, val_transform, disease_map, condition_map)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    init_checkpoint = SSL_CHECKPOINT_PATH
    if not os.path.exists(init_checkpoint):
        print(f" Warning: Can't find {SSL_CHECKPOINT_PATH}")
        init_checkpoint = args.model_path

    print(f" Init model structure using: {init_checkpoint}")
    model = OCTDLMultiTaskModel(checkpoint_path=init_checkpoint, num_diseases=len(disease_map),
                                num_conditions=len(condition_map), unfreeze_last_block=True)

    print(f" Loading Model from: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    # 4. Inference Loop
    all_preds_d, all_labels_d = [], []
    all_preds_c, all_labels_c = [], []

    print(" Running Inference on Test Set...")
    with torch.no_grad():
        for images, labels_d, labels_c in tqdm(test_loader):
            images = images.to(args.device)

            logits_d, logits_c = model(images)

            # Disease predictions
            preds_d = torch.argmax(logits_d, dim=1).cpu().numpy()
            all_preds_d.extend(preds_d)
            all_labels_d.extend(labels_d.numpy())

            # Condition predictions
            preds_c = torch.argmax(logits_c, dim=1).cpu().numpy()
            all_preds_c.extend(preds_c)
            all_labels_c.extend(labels_c.numpy())

    print("\n" + "=" * 30)
    print(" DISEASE CLASSIFICATION REPORT")
    print("=" * 30)

    disease_names = [idx_to_disease[i] for i in range(len(disease_map))]

    print(classification_report(all_labels_d, all_preds_d, target_names=disease_names, digits=4))

    cm_d = confusion_matrix(all_labels_d, all_preds_d)
    plot_confusion_matrix(cm_d, disease_names, "Disease Confusion Matrix", os.path.join(RESULT_DIR, "cm_disease.png"))

    print("\n" + "=" * 30)
    print(" CONDITION CLASSIFICATION REPORT")
    print("=" * 30)

    valid_mask = np.array(all_labels_c) != -100
    clean_preds_c = np.array(all_preds_c)[valid_mask]
    clean_labels_c = np.array(all_labels_c)[valid_mask]

    condition_names = [idx_to_condition[i] for i in range(len(condition_map))]
    unique_labels = sorted(list(set(clean_labels_c)))
    target_names_subset = [idx_to_condition[i] for i in unique_labels]

    print(classification_report(clean_labels_c, clean_preds_c, target_names=target_names_subset, digits=4))

    cm_c = confusion_matrix(clean_labels_c, clean_preds_c)
    plot_confusion_matrix(cm_c, target_names_subset, "Condition Confusion Matrix",
                          os.path.join(RESULT_DIR, "cm_condition.png"))

    print(f"\n Analysis complete. Check the '{RESULT_DIR}' folder for visualization plots.")


if __name__ == "__main__":
    main()