import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class OCTDLMultiTaskDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, disease_map=None, condition_map=None):

        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.disease_map = disease_map
        self.condition_map = condition_map
        self.IGNORE_INDEX = -100

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = row['file_name']
        disease_folder = row['disease']
        img_path = os.path.join(self.root_dir, disease_folder, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        disease_str = str(row['label_disease'])
        label_disease = self.disease_map[disease_str]

        # unknown conditions -> IGNORE_INDEX (masked in loss)
        condition_str = str(row['label_condition_raw'])
        if condition_str in self.condition_map:
            label_condition = self.condition_map[condition_str]
        else:
            label_condition = self.IGNORE_INDEX

        return image, label_disease, label_condition


def get_data_splits(csv_path, test_size=0.2, val_size=0.1):
    """
    Split dataset based on patient IDs to prevent data leakage.
    Returns train, validation, and test dataframes along with label mappings.
    """
    print(f"loading metadata from: {csv_path}")
    df = pd.read_csv(csv_path)

    unique_diseases = sorted(df['label_disease'].astype(str).unique())
    disease_map = {name: i for i, name in enumerate(unique_diseases)}

    valid_conditions = sorted(df[df['label_condition_raw'] != 'IGNORE']['label_condition_raw'].unique())
    condition_map = {name: i for i, name in enumerate(valid_conditions)}

    print(f"classes (Disease): {disease_map}")
    print(f"classes (Condition): {condition_map}")

    patients = df[['patient_id', 'label_disease']].drop_duplicates()

    total_test_val_ratio = test_size + val_size
    train_pat, temp_pat = train_test_split(
        patients,
        test_size=total_test_val_ratio,
        random_state=42,
        stratify=patients['label_disease']
    )

    relative_test_ratio = test_size / total_test_val_ratio
    val_pat, test_pat = train_test_split(
        temp_pat,
        test_size=relative_test_ratio,
        random_state=42,
        stratify=temp_pat['label_disease']
    )

    train_df = df[df['patient_id'].isin(train_pat['patient_id'])]
    val_df = df[df['patient_id'].isin(val_pat['patient_id'])]
    test_df = df[df['patient_id'].isin(test_pat['patient_id'])]

    print("split Complete:")
    print(f"train: {len(train_df)} images ({len(train_pat)} patients)")
    print(f"val:   {len(val_df)} images ({len(val_pat)} patients)")
    print(f"test:  {len(test_df)} images ({len(test_pat)} patients)")

    return train_df, val_df, test_df, disease_map, condition_map