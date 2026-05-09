import os
import glob
from pathlib import Path


def setup_dinov2_dataset(source_dir, dest_dir):
    train_source = os.path.join(source_dir, 'train')

    print(f"scanning {train_source}")

    if not os.path.exists(train_source):
        print(f"not found: {train_source}")
        return

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(train_source, '**', ext), recursive=True))
        image_paths.extend(glob.glob(os.path.join(train_source, '**', ext.upper()), recursive=True))

    dummy_class_dir = os.path.join(dest_dir, 'train', 'all_scans')
    os.makedirs(dummy_class_dir, exist_ok=True)

    print(f"{len(image_paths)} images, building symlinks")
    for idx, path in enumerate(image_paths):
        ext = Path(path).suffix
        new_filename = f"scan_{idx:06d}{ext}"
        dest_path = os.path.join(dummy_class_dir, new_filename)

        if not os.path.exists(dest_path):
            os.symlink(os.path.abspath(path), dest_path)

    print("done")


if __name__ == "__main__":
    SOURCE_DIR = "../combined/combined"
    DEST_DIR = "../dinov2_dataset"
    setup_dinov2_dataset(SOURCE_DIR, DEST_DIR)