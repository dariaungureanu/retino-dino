import os
import glob
from pathlib import Path


def setup_dinov2_dataset(source_dir, dest_dir):
    print(f"Scanning {source_dir} for images...")
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(source_dir, '**', ext), recursive=True))
        image_paths.extend(glob.glob(os.path.join(source_dir, '**', ext.upper()), recursive=True))

    dummy_class_dir = os.path.join(dest_dir, 'train', 'all_scans')
    os.makedirs(dummy_class_dir, exist_ok=True)

    print(f"Found {len(image_paths)} images. Creating symlinks...")
    for idx, path in enumerate(image_paths):
        ext = Path(path).suffix
        new_filename = f"scan_{idx:06d}{ext}"
        dest_path = os.path.join(dummy_class_dir, new_filename)
        if not os.path.exists(dest_path):
            os.symlink(os.path.abspath(path), dest_path)
    print("Dataset preparation complete.")


if __name__ == "__main__":
    # Pointing to the folders on the university PC
    SOURCE_DIR = "../combined"
    DEST_DIR = "../dinov2_dataset"
    setup_dinov2_dataset(SOURCE_DIR, DEST_DIR)