import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import argparse

DEFAULT_SOURCE_ROOT = r"C:\Datasets\OCTDL\OCTDL"
DEFAULT_DEST_ROOT = r"C:\Datasets\OCTDL_Cleaned"
DEFAULT_CSV_NAME = "OCTDL_labels.csv"
IMG_SIZE = 512


class RemoveTopBackgroundRobust:
    """
    Custom transform to remove text/artifacts from the top of OCT images
    using adaptive thresholding and flood fill.
    """

    def __init__(self, threshold_offset=15):
        self.threshold_offset = threshold_offset

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        thr = max(int(gray.mean()) - self.threshold_offset, 0)
        bg_candidate = (gray < thr).astype(np.uint8)

        flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
        tmp = bg_candidate.copy()
        for col in range(w):
            if tmp[0, col]:
                cv2.floodFill(tmp, flood, seedPoint=(col, 0), newVal=1,
                              flags=cv2.FLOODFILL_MASK_ONLY)

        bg = (flood[1:-1, 1:-1] == 1).astype(np.uint8)
        mask = 1 - bg
        arr_fg = arr * mask[..., None]

        return Image.fromarray(arr_fg)


def build_image_path(row, root_dir):
    """Constructs the full file path based on the dataframe row."""
    fn = str(row["file_name"]).strip()
    disease = str(row["disease"]).strip()

    root, ext = os.path.splitext(fn)
    if ext == "":
        fn = fn + ".jpg"

    return os.path.join(root_dir, disease, fn)


def prepare_metadata(df):
    """Generate patient_id and clean condition labels."""
    print(" -> Generating metadata (Patient IDs, Clean Conditions)...")

    df["label_disease"] = df["disease"].astype(str)

    # Keep only conditions with >= 30 samples; everything else becomes IGNORE.
    cond_counts = df["condition"].value_counts()
    valid_conditions = cond_counts[cond_counts >= 30].index

    def make_label_condition(row):
        cond = row["condition"]
        if cond in valid_conditions:
            return cond
        else:
            return "IGNORE"

    df["label_condition_raw"] = df.apply(make_label_condition, axis=1)

    if "patient_id" in df.columns:
        df["patient_id"] = df["patient_id"].astype(str)
        df["patient_id"] = df["patient_id"].fillna("")
        mask_empty = df["patient_id"] == ""
        df.loc[mask_empty, "patient_id"] = "anon_" + df.index[mask_empty].astype(str)
    else:
        df["patient_id"] = ["img_" + str(i) for i in range(len(df))]

    return df


def process_images(source_root, dest_root, csv_filename):
    csv_path = os.path.join(source_root, csv_filename)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    print(f"loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    df = prepare_metadata(df)
    cleaner = RemoveTopBackgroundRobust(threshold_offset=15)
    os.makedirs(dest_root, exist_ok=True)

    valid_rows = []

    print(f"starting image processing. Saving to: {dest_root}")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            src_path = build_image_path(row, source_root)

            if not os.path.exists(src_path):
                if not src_path.endswith(".jpg"):
                    src_path += ".jpg"

                if not os.path.exists(src_path):
                    continue

            img = Image.open(src_path).convert("RGB")
            img_clean = cleaner(img)

            if IMG_SIZE:
                img_clean = img_clean.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)

            rel_dir = str(row["disease"])
            save_dir = os.path.join(dest_root, rel_dir)
            os.makedirs(save_dir, exist_ok=True)

            file_name = os.path.basename(src_path)
            file_name = os.path.splitext(file_name)[0] + ".jpg"
            dest_path = os.path.join(save_dir, file_name)

            img_clean.save(dest_path, quality=95)
            row["file_name"] = file_name
            valid_rows.append(row)

        except Exception as e:
            print(f"error processing image {idx}: {e}")
            continue

    new_df = pd.DataFrame(valid_rows)
    output_csv_path = os.path.join(dest_root, "OCTDL_clean_metadata.csv")
    new_df.to_csv(output_csv_path, index=False)
    print("processing Complete!")
    print(f"original images: {len(df)}")
    print(f"successfully processed: {len(new_df)}")
    print(f"new CSV saved at: {output_csv_path}")
    print(f"cleaned images folder: {dest_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess OCTDL dataset using flood fill and resize")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE_ROOT, help="Path to original OCTDL folder")
    parser.add_argument("--dest", type=str, default=DEFAULT_DEST_ROOT, help="Path to save cleaned images")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_NAME, help="Name of the label CSV file")

    args = parser.parse_args()

    process_images(args.source, args.dest, args.csv)