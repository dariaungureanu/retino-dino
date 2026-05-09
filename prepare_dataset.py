import os
import glob
from pathlib import Path


def setup_dinov2_dataset(source_dir, dest_dir):
    train_source = os.path.join(source_dir, 'train')

    print(f"scanning strictly inside {train_source} for images...")

    if not os.path.exists(train_source):
        print(f"ERROR: Could not find {train_source}. Check your paths!")
        return

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(train_source, '**', ext), recursive=True))
        image_paths.extend(glob.glob(os.path.join(train_source, '**', ext.upper()), recursive=True))

    dummy_class_dir = os.path.join(dest_dir, 'train', 'all_scans')
    os.makedirs(dummy_class_dir, exist_ok=True)

    print(f"found {len(image_paths)} training images. Creating clean symlinks...")
    for idx, path in enumerate(image_paths):
        ext = Path(path).suffix
        new_filename = f"scan_{idx:06d}{ext}"
        dest_path = os.path.join(dummy_class_dir, new_filename)

        if not os.path.exists(dest_path):
            os.symlink(os.path.abspath(path), dest_path)

    print("dataset preparation complete. No test/val data leaked!")


if __name__ == "__main__":
    SOURCE_DIR = "../combined/combined"
    DEST_DIR = "../dinov2_dataset"
    setup_dinov2_dataset(SOURCE_DIR, DEST_DIR)