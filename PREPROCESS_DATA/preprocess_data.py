import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import argparse

### C:\Users\daria\AppData\Local\Programs\Python\Python313\python.exe -m pip install opencv-python
# --- CONFIGURATION ---
# Default paths (can be overridden by command line args)
DEFAULT_SOURCE_ROOT = r"C:\Datasets\OCTDL\OCTDL"
DEFAULT_DEST_ROOT = r"C:\Datasets\OCTDL_Cleaned"
DEFAULT_CSV_NAME = "OCTDL_labels.csv"
IMG_SIZE = 518  # DINOv2 usually works well with 518 or 224. Resizing here saves disk space.


class RemoveTopBackgroundRobust:
    """
    Custom transform to remove text/artifacts from the top of OCT images
    using adaptive thresholding and flood fill.
    """

    def __init__(self, threshold_offset=15):
        self.threshold_offset = threshold_offset

    def __call__(self, img: Image.Image) -> Image.Image:
        # Convert PIL to NumPy array (RGB)
        arr = np.array(img.convert("RGB"))

        # Convert to grayscale for processing
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Adaptive thresholding on the whole image
        thr = max(int(gray.mean()) - self.threshold_offset, 0)
        bg_candidate = (gray < thr).astype(np.uint8)

        # Flood-fill from the top line only
        flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
        tmp = bg_candidate.copy()
        for col in range(w):
            if tmp[0, col]:
                cv2.floodFill(tmp, flood, seedPoint=(col, 0), newVal=1,
                              flags=cv2.FLOODFILL_MASK_ONLY)

        # Extract background mask (1 = background found by floodfill)
        bg = (flood[1:-1, 1:-1] == 1).astype(np.uint8)

        # Create final mask: 1 = keep (foreground), 0 = remove (background)
        mask = 1 - bg

        # Apply mask to original RGB image
        arr_fg = arr * mask[..., None]

        return Image.fromarray(arr_fg)


def build_image_path(row, root_dir):
    """Constructs the full file path based on the dataframe row."""
    fn = str(row["file_name"]).strip()
    disease = str(row["disease"]).strip()

    # Handle missing extensions
    root, ext = os.path.splitext(fn)
    if ext == "":
        fn = fn + ".jpg"

    return os.path.join(root_dir, disease, fn)


def prepare_metadata(df):
    """
    Applies the logic to generate patient_ids and clean condition labels.
    """
    print(" -> Generating metadata (Patient IDs, Clean Conditions)...")

    # 1. Ensure Disease is string
    df["label_disease"] = df["disease"].astype(str)

    # 2. Process Conditions (Keep only those with >= 30 samples)
    cond_counts = df["condition"].value_counts()
    valid_conditions = cond_counts[cond_counts >= 30].index

    def make_label_condition(row):
        cond = row["condition"]
        if cond in valid_conditions:
            return cond
        else:
            return "IGNORE"  # Will be handled in training script

    df["label_condition_raw"] = df.apply(make_label_condition, axis=1)

    # 3. Process Patient IDs
    if "patient_id" in df.columns:
        df["patient_id"] = df["patient_id"].astype(str)
        df["patient_id"] = df["patient_id"].fillna("")
        # If empty, assign unique ID based on index
        mask_empty = df["patient_id"] == ""
        df.loc[mask_empty, "patient_id"] = "anon_" + df.index[mask_empty].astype(str)
    else:
        # Fallback if column doesn't exist
        df["patient_id"] = ["img_" + str(i) for i in range(len(df))]

    return df


def process_images(source_root, dest_root, csv_filename):
    csv_path = os.path.join(source_root, csv_filename)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Prepare metadata columns
    df = prepare_metadata(df)

    # Initialize Cleaner
    cleaner = RemoveTopBackgroundRobust(threshold_offset=15)

    # Create Destination Folder
    os.makedirs(dest_root, exist_ok=True)

    valid_rows = []

    print(f"Starting image processing. Saving to: {dest_root}")
    # Use tqdm for a progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Construct Source Path
            src_path = build_image_path(row, source_root)

            if not os.path.exists(src_path):
                # Try adding .jpg if not found (common issue in datasets)
                if not src_path.endswith(".jpg"):
                    src_path += ".jpg"

                if not os.path.exists(src_path):
                    # Skip if still not found
                    continue

            # Load Image
            img = Image.open(src_path).convert("RGB")

            # Apply Flood Fill Cleaning
            img_clean = cleaner(img)

            # Resize (Optional but recommended for speed)
            if IMG_SIZE:
                img_clean = img_clean.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)

            # Construct Destination Path
            # We preserve the 'Disease' folder structure
            rel_dir = str(row["disease"])
            save_dir = os.path.join(dest_root, rel_dir)
            os.makedirs(save_dir, exist_ok=True)

            # Save File
            file_name = os.path.basename(src_path)
            # Ensure extension is .jpg for consistency
            file_name = os.path.splitext(file_name)[0] + ".jpg"
            dest_path = os.path.join(save_dir, file_name)

            img_clean.save(dest_path, quality=95)

            # Update row with new relative info and add to valid list
            row["file_name"] = file_name  # Update filename in case extension changed
            valid_rows.append(row)

        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue

    # Save the new cleaned CSV
    new_df = pd.DataFrame(valid_rows)
    output_csv_path = os.path.join(dest_root, "OCTDL_clean_metadata.csv")
    new_df.to_csv(output_csv_path, index=False)

    print("------------------------------------------------")
    print(f"Processing Complete!")
    print(f"Original images: {len(df)}")
    print(f"Successfully processed: {len(new_df)}")
    print(f"New CSV saved at: {output_csv_path}")
    print(f"Cleaned images folder: {dest_root}")


if __name__ == "__main__":
    # Setup simple argument parsing
    parser = argparse.ArgumentParser(description="Preprocess OCTDL dataset (FloodFill + Resize)")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE_ROOT, help="Path to original OCTDL folder")
    parser.add_argument("--dest", type=str, default=DEFAULT_DEST_ROOT, help="Path to save cleaned images")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_NAME, help="Name of the label CSV file")

    args = parser.parse_args()

    process_images(args.source, args.dest, args.csv)