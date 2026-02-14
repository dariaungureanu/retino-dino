from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import get_data_splits, OCTDLMultiTaskDataset

ROOT_DIR = r"C:\Datasets\OCTDL_Cleaned"
CSV_PATH = f"{ROOT_DIR}/OCTDL_clean_metadata.csv"


def test_pipeline():
    print("--- 1. Testing data split function ---")
    try:
        train_df, val_df, test_df, disease_map, condition_map = get_data_splits(CSV_PATH)
        print(f"Split successful.")
        print(f"   Train samples: {len(train_df)}")
        print(f"   Disease mapping: {disease_map}")
        print(f"   Condition mapping: {condition_map}")
    except Exception as e:
        print(f"Error during split: {e}")
        return

    print("\n--- 2. Testing dataset class ---")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    try:
        train_ds = OCTDLMultiTaskDataset(
            dataframe=train_df,
            root_dir=ROOT_DIR,
            transform=transform,
            disease_map=disease_map,
            condition_map=condition_map
        )
        print(f"Dataset initialized. Length: {len(train_ds)}")
    except Exception as e:
        print(f"Error initializing dataset class: {e}")
        return

    print("\n--- 3. Testing item loading (getitem) ---")
    try:
        img, label_dis, label_cond = train_ds[0]

        print(f"Image loaded successfully.")
        print(f"   Tensor shape: {img.shape} (Expected: [3, 224, 224])")
        print(f"   Disease label (int): {label_dis}")
        print(f"   Condition label (int): {label_cond}")

        if not isinstance(label_dis, int) or not isinstance(label_cond, int):
            print("Warning: Labels are not integers.")
        else:
            print("   Label types are correct (int).")

    except Exception as e:
        print(f"Error in getitem: {e}")
        print(f"   Attempted to load: {train_ds.df.iloc[0]['file_name']}")
        return

    print("\n--- 4. Testing dataloader (batching) ---")
    try:
        loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        images, labels_d, labels_c = next(iter(loader))

        print(f"Batch loaded successfully.")
        print(f"   Batch images shape: {images.shape} (Expected: [4, 3, 224, 224])")
        print(f"   Batch disease labels: {labels_d}")
        print(f"   Batch condition labels: {labels_c}")
    except Exception as e:
        print(f"Error in dataloader: {e}")
        return

    print("\nAll tests passed. Dataset is working correctly.")


if __name__ == "__main__":
    test_pipeline()