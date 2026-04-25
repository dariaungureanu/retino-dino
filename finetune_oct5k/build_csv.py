"""
Build CSV from OCT5k bounding box annotations.
=================================================
Converts bounding boxes → multi-label binary vectors (per image).
Resolves images across Images_Automatic/ and Images_Manual/.
Extracts patient/volume IDs for patient-based splitting.
Saves bounding boxes separately for GradCAM validation later.

Usage:
    python finetune_oct5k/build_csv.py \
        --data_path /home/student/Ungureanu_Daria/oct5k \
        --out_csv /home/student/Ungureanu_Daria/oct5k/oct5k_metadata.csv
"""

import argparse
import os
import re

import pandas as pd

BIOMARKERS = [
    "Choroidalfolds", "Fluid", "Geographicatrophy", "Harddrusen",
    "Hyperfluorescentspots", "PRlayerdisruption", "Reticulardrusen",
    "Softdrusen", "SoftdrusenPED",
]

# Short names for display
SHORT_NAMES = {
    "Choroidalfolds": "CF",
    "Fluid": "Fluid",
    "Geographicatrophy": "GA",
    "Harddrusen": "HD",
    "Hyperfluorescentspots": "HFS",
    "PRlayerdisruption": "PRL",
    "Reticulardrusen": "RD",
    "Softdrusen": "SD",
    "SoftdrusenPED": "SDPED",
}


def resolve_image_path(relative_path, data_path):
    """
    Find the actual image file. Check Images_Automatic first, then Images_Manual.
    Returns the full path or None if not found.
    """
    for folder in ["Images_Automatic", "Images_Manual"]:
        full = os.path.join(data_path, folder, relative_path)
        if os.path.isfile(full):
            return os.path.join(folder, relative_path)
    return None


def extract_patient_id(image_path):
    """
    Extract patient/volume ID from path.
    AMD: 'AMD Part1/AMD (3)/Image (14).png' → 'AMD_3'
         'AMD Part1/AMD (3).E2E/Image (5).png' → 'AMD_3'
    DRUSEN: 'DRUSEN/DRUSEN-142234-1.png' → 'DRUSEN_142234'
    """
    parts = image_path.split("/")

    if parts[0] == "DRUSEN":
        # DRUSEN-142234-1.png → patient 142234
        fname = parts[-1]
        match = re.match(r"DRUSEN-(\d+)-", fname)
        if match:
            return f"DRUSEN_{match.group(1)}"
        return f"DRUSEN_{fname}"
    else:
        # AMD Part1/AMD (3)/Image.png or AMD Part1/AMD (3).E2E/Image.png
        if len(parts) >= 2:
            vol = parts[1]  # "AMD (3)" or "AMD (3).E2E"
            # Remove .E2E suffix and extract number
            vol_clean = re.sub(r"\.E2E$", "", vol)
            match = re.match(r"AMD \((\d+)\)", vol_clean)
            if match:
                return f"AMD_{match.group(1)}"
            return f"AMD_{vol_clean}"

    return image_path


def main():
    parser = argparse.ArgumentParser(description="Build CSV for OCT5k")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--bbox_csv", type=str, default=None,
                        help="Path to all_bounding_boxes.csv (default: data_path/Detection/all_bounding_boxes.csv)")
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--min_samples", type=int, default=10,
                        help="Drop biomarkers with fewer than this many positive images")
    args = parser.parse_args()

    if args.bbox_csv is None:
        args.bbox_csv = os.path.join(args.data_path, "Detection", "all_bounding_boxes.csv")
    if args.out_csv is None:
        args.out_csv = os.path.join(args.data_path, "oct5k_metadata.csv")

    # Load bounding boxes
    bbox_df = pd.read_csv(args.bbox_csv)
    print(f"[DATA] Loaded {len(bbox_df)} bounding boxes from {args.bbox_csv}")
    print(f"[DATA] Unique images: {bbox_df['image'].nunique()}")

    # Build multi-label vectors per image
    rows = []
    missing = 0

    for img_rel, group in bbox_df.groupby("image"):
        # Resolve actual path
        resolved = resolve_image_path(img_rel, args.data_path)
        if resolved is None:
            missing += 1
            continue

        # Binary vector: which biomarkers present in this image
        present_classes = set(group["class"].unique())
        labels = {bm: (1 if bm in present_classes else 0) for bm in BIOMARKERS}

        # Patient ID
        patient_id = extract_patient_id(img_rel)

        # Store bounding boxes as JSON string for GradCAM validation later
        bboxes = group[["xmin", "ymin", "xmax", "ymax", "class"]].to_dict("records")

        rows.append({
            "image": resolved,
            "image_csv": img_rel,  # original path from CSV
            "patient_id": patient_id,
            **labels,
            "bbox_count": len(bboxes),
        })

    if missing > 0:
        print(f"[WARN] {missing} images not found!")

    df = pd.DataFrame(rows)
    print(f"\n[DATA] Total images: {len(df)}")
    print(f"[DATA] Unique patients: {df['patient_id'].nunique()}")

    # Biomarker distribution
    print(f"\n[DATA] Biomarker distribution:")
    drop_biomarkers = []
    for bm in BIOMARKERS:
        n_pos = int(df[bm].sum())
        pct = n_pos / len(df) * 100
        status = ""
        if n_pos < args.min_samples:
            status = f" ← WARNING: only {n_pos} images, consider dropping"
            drop_biomarkers.append(bm)
        print(f"  {SHORT_NAMES[bm]:>5} ({bm}): {n_pos} images ({pct:.1f}%){status}")

    if drop_biomarkers:
        print(f"\n[INFO] Biomarkers with <{args.min_samples} images: {drop_biomarkers}")
        print(f"[INFO] These will be kept in CSV but may hurt training.")

    # Patient distribution
    print(f"\n[DATA] Images per patient (top 10):")
    pat_counts = df.groupby("patient_id").size().sort_values(ascending=False)
    for pat, n in pat_counts.head(10).items():
        print(f"  {pat}: {n} images")

    # Save
    df.to_csv(args.out_csv, index=False)
    print(f"\n[SAVED] {args.out_csv}")
    print(f"[INFO] Columns: {list(df.columns)}")

    # Also save the full bounding box data for GradCAM validation
    bbox_out = os.path.join(os.path.dirname(args.out_csv), "oct5k_bboxes.csv")
    bbox_df.to_csv(bbox_out, index=False)
    print(f"[SAVED] {bbox_out} (for GradCAM validation)")


if __name__ == "__main__":
    main()