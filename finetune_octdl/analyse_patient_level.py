"""
Group per-image predictions by patient_id and compute patient-level metrics
via majority voting (collect images, majority vote -> patient prediction,
score against the patient's true label).

Usage:
    python finetune_octdl/analyse_patient_level.py \
        --data_path /home/student/Ungureanu_Daria/OCTDL_Cleaned \
        --checkpoint_dir saved_models/octdl_domain_adapted_unfreeze2 \
        --out_dir results/patient_level
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import (
    IGNORE_INDEX, OCTDLMultiTaskDataset, get_data_splits, get_eval_transform,
)
from model import OCTDLMultiTaskModel, load_backbone


def main():
    parser = argparse.ArgumentParser(description="Patient-Level Evaluation")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="results/patient_level")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if args.model_path:
        model_path = args.model_path
    elif args.checkpoint_dir:
        model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    else:
        raise ValueError("Provide --model_path or --checkpoint_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    ckpt = torch.load(model_path, map_location=device)
    config = ckpt["config"]
    disease_map = ckpt["disease_map"]
    condition_map = ckpt["condition_map"]
    inv_disease = {v: k for k, v in disease_map.items()}
    inv_condition = {v: k for k, v in condition_map.items()}

    backbone = load_backbone(config["arch"], config["checkpoint"], device)
    model = OCTDLMultiTaskModel(
        backbone=backbone,
        num_diseases=len(disease_map),
        num_conditions=len(condition_map),
        freeze_backbone=(config["unfreeze_last_n"] < 12),
        unfreeze_last_n=config["unfreeze_last_n"],
        head_hidden=config["head_hidden"],
        head_dropout=config["head_dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load test data with patient IDs
    csv_path = os.path.join(args.data_path, "OCTDL_clean_metadata.csv")
    train_df, _, test_df, _, _ = get_data_splits(csv_path)
    eval_transform = get_eval_transform(config["img_size"])
    test_ds = OCTDLMultiTaskDataset(
        test_df, args.data_path, eval_transform, disease_map, condition_map,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    disease_prior = np.zeros(len(disease_map), dtype=np.int64)
    for cls_name, cls_idx in disease_map.items():
        disease_prior[cls_idx] = int((train_df["label_disease"].astype(str) == cls_name).sum())
    condition_prior = np.zeros(len(condition_map), dtype=np.int64)
    for cls_name, cls_idx in condition_map.items():
        condition_prior[cls_idx] = int((train_df["label_condition_raw"].astype(str) == cls_name).sum())
    print(f"disease prior (train counts): {dict(zip([inv_disease[i] for i in range(len(disease_map))], disease_prior.tolist()))}")

    # Per-image predictions
    softmax = nn.Softmax(dim=1)
    all_preds_d, all_preds_c, all_probs_d = [], [], []

    with torch.no_grad():
        for images, labels_d, labels_c in tqdm(test_loader, desc="Running inference"):
            images = images.to(device)
            logits_d, logits_c = model(images)
            probs_d = softmax(logits_d)
            pred_d = torch.argmax(probs_d, dim=1)
            pred_c = torch.argmax(logits_c, dim=1)

            all_preds_d.extend(pred_d.cpu().numpy())
            all_preds_c.extend(pred_c.cpu().numpy())
            all_probs_d.extend(probs_d.cpu().numpy())

    all_preds_d = np.array(all_preds_d)
    all_preds_c = np.array(all_preds_c)
    all_probs_d = np.array(all_probs_d)

    test_df_reset = test_df.reset_index(drop=True)
    test_df_reset["pred_disease"] = all_preds_d
    test_df_reset["pred_condition"] = all_preds_c
    test_df_reset["true_disease"] = test_df_reset["label_disease"].map(disease_map)
    test_df_reset["true_condition"] = test_df_reset["label_condition_raw"].map(
        lambda x: condition_map.get(x, IGNORE_INDEX)
    )

    # Image-level metrics (for comparison)
    print("image-level metrics")
    img_d_f1 = f1_score(test_df_reset["true_disease"], all_preds_d,
                         average="macro", zero_division=0)
    img_d_acc = accuracy_score(test_df_reset["true_disease"], all_preds_d)
    print(f"disease:   Acc={img_d_acc*100:.1f}%  Macro-F1={img_d_f1:.4f}")

    cond_mask = test_df_reset["true_condition"] != IGNORE_INDEX
    if cond_mask.sum() > 0:
        img_c_f1 = f1_score(
            test_df_reset.loc[cond_mask, "true_condition"],
            all_preds_c[cond_mask],
            average="macro", zero_division=0,
        )
        img_c_acc = accuracy_score(
            test_df_reset.loc[cond_mask, "true_condition"],
            all_preds_c[cond_mask],
        )
        print(f"condition: Acc={img_c_acc*100:.1f}%  Macro-F1={img_c_f1:.4f}")

    def majority_vote_with_prior(votes, prior):
        """Most-frequent prediction across a patient's scans.
        Ties are resolved in favour of the class with the larger training
        prior; ties on prior fall back to the lower class index."""
        counts = Counter(int(v) for v in votes)
        max_count = max(counts.values())
        candidates = [c for c, n in counts.items() if n == max_count]
        if len(candidates) == 1:
            return candidates[0], False
        winner = max(candidates, key=lambda c: (prior[c], -c))
        return winner, True

    print("patient-level metrics (majority vote)")
    patient_results = []
    n_disease_ties = 0
    n_condition_ties = 0
    for patient_id, group in test_df_reset.groupby("patient_id"):
        true_disease = group["true_disease"].mode().iloc[0]
        true_disease_name = inv_disease[true_disease]

        pred_disease, was_tied_d = majority_vote_with_prior(
            group["pred_disease"].values, disease_prior,
        )
        if was_tied_d:
            n_disease_ties += 1
        pred_disease_name = inv_disease[pred_disease]

        avg_probs = all_probs_d[group.index].mean(axis=0)
        pred_disease_avg = int(np.argmax(avg_probs))
        pred_disease_avg_name = inv_disease[pred_disease_avg]

        valid_conds = group[group["true_condition"] != IGNORE_INDEX]
        true_condition = valid_conds["true_condition"].mode().iloc[0] if len(valid_conds) > 0 else IGNORE_INDEX
        if len(valid_conds) > 0:
            pred_condition, was_tied_c = majority_vote_with_prior(
                all_preds_c[valid_conds.index], condition_prior,
            )
            if was_tied_c:
                n_condition_ties += 1
        else:
            pred_condition = IGNORE_INDEX

        n_scans = len(group)
        n_correct = (group["pred_disease"] == group["true_disease"]).sum()

        patient_results.append({
            "patient_id": patient_id,
            "n_scans": n_scans,
            "true_disease": true_disease,
            "true_disease_name": true_disease_name,
            "pred_disease_vote": pred_disease,
            "pred_disease_vote_name": pred_disease_name,
            "pred_disease_avg": pred_disease_avg,
            "pred_disease_avg_name": pred_disease_avg_name,
            "scan_accuracy": n_correct / n_scans,
            "true_condition": true_condition,
            "pred_condition": pred_condition,
        })

    pat_df = pd.DataFrame(patient_results)

    # Disease, patient-level
    print(f"\npatients in test set: {len(pat_df)}")
    print("\n--- Disease (Majority Vote) ---")

    disease_names = [inv_disease[i] for i in range(len(disease_map))]
    pat_true_d = pat_df["true_disease"].values
    pat_pred_d_vote = pat_df["pred_disease_vote"].values
    pat_pred_d_avg = pat_df["pred_disease_avg"].values

    pat_d_acc_vote = accuracy_score(pat_true_d, pat_pred_d_vote) * 100
    pat_d_bal_vote = balanced_accuracy_score(pat_true_d, pat_pred_d_vote) * 100
    pat_d_f1_vote = f1_score(pat_true_d, pat_pred_d_vote, average="macro", zero_division=0)

    print(f"acc={pat_d_acc_vote:.1f}%  Bal.Acc={pat_d_bal_vote:.1f}%  Macro-F1={pat_d_f1_vote:.4f}")
    print("\nper-class report (majority vote):")
    print(classification_report(
        pat_true_d, pat_pred_d_vote,
        target_names=disease_names, zero_division=0,
    ))

    pat_d_acc_avg = accuracy_score(pat_true_d, pat_pred_d_avg) * 100
    pat_d_f1_avg = f1_score(pat_true_d, pat_pred_d_avg, average="macro", zero_division=0)
    print("--- Disease (Average Probabilities) ---")
    print(f"acc={pat_d_acc_avg:.1f}%  Macro-F1={pat_d_f1_avg:.4f}")

    # Condition, patient-level
    valid_pat = pat_df[pat_df["true_condition"] != IGNORE_INDEX]
    if len(valid_pat) > 0:
        print("\n--- Condition (Majority Vote) ---")
        condition_names = [inv_condition[i] for i in range(len(condition_map))]
        pat_c_acc = accuracy_score(valid_pat["true_condition"], valid_pat["pred_condition"]) * 100
        pat_c_f1 = f1_score(valid_pat["true_condition"], valid_pat["pred_condition"],
                            average="macro", zero_division=0)
        print(f"acc={pat_c_acc:.1f}%  Macro-F1={pat_c_f1:.4f}")
        print("\nper-class report:")
        print(classification_report(
            valid_pat["true_condition"], valid_pat["pred_condition"],
            target_names=condition_names, zero_division=0,
        ))

    # Per-patient scan accuracy
    print("\n--- Per-Patient Scan Accuracy ---")
    print("(What fraction of each patient's scans were correctly classified?)")
    for _, row in pat_df.sort_values("scan_accuracy").iterrows():
        status = "ok" if row["pred_disease_vote"] == row["true_disease"] else "no"
        print(f"{status} Patient {row['patient_id']:>6}: "
              f"{row['scan_accuracy']:.0%} ({row['n_scans']} scans) "
              f"True={row['true_disease_name']:>4} Pred={row['pred_disease_vote_name']:>4}")

    print(f"\ntie-broken majority votes: disease={n_disease_ties}/{len(pat_df)} patients, "
          f"condition={n_condition_ties}/{max(len(valid_pat), 1)} patients")

    # Patient-level confusion matrices: majority-vote vs average-probability
    cm_vote = confusion_matrix(pat_true_d, pat_pred_d_vote,
                                labels=list(range(len(disease_map))))
    cm_avg = confusion_matrix(pat_true_d, pat_pred_d_avg,
                               labels=list(range(len(disease_map))))

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, cm, strategy, acc, f1 in [
        (axes[0], cm_vote, "Majority Vote", pat_d_acc_vote, pat_d_f1_vote),
        (axes[1], cm_avg, "Average Probability", pat_d_acc_avg, pat_d_f1_avg),
    ]:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=disease_names, yticklabels=disease_names, ax=ax)
        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("True", fontweight="bold")
        ax.set_title(
            f"patient-Level Disease - {strategy}\n"
            f"Acc={acc:.1f}%  Macro-F1={f1:.4f}",
            fontweight="bold",
        )
    plt.tight_layout()
    cm_path = os.path.join(args.out_dir, "patient_confusion_disease.png")
    fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n{cm_path}")

    # Summary comparison
    print("image-level vs patient-level")
    print(f"{'':>20} {'Image-Level':>15} {'Patient-Level':>15}")
    print(f"{'Disease Acc':>20} {img_d_acc*100:>14.1f}% {pat_d_acc_vote:>14.1f}%")
    print(f"{'Disease Macro-F1':>20} {img_d_f1:>15.4f} {pat_d_f1_vote:>15.4f}")
    print(f"{'N samples':>20} {len(test_df_reset):>15} {len(pat_df):>15}")

    # Save results
    import json
    results = {
        "image_level": {
            "n_images": len(test_df_reset),
            "disease_acc": float(img_d_acc),
            "disease_f1": float(img_d_f1),
        },
        "patient_level": {
            "n_patients": len(pat_df),
            "disease_acc_vote": float(pat_d_acc_vote / 100),
            "disease_f1_vote": float(pat_d_f1_vote),
            "disease_acc_avg": float(pat_d_acc_avg / 100),
            "disease_f1_avg": float(pat_d_f1_avg),
        },
    }
    json_path = os.path.join(args.out_dir, "patient_level_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{json_path}")
    print("\ndone.")


if __name__ == "__main__":
    main()