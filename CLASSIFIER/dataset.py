import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class OCTDLMultiTaskDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, disease_map=None, condition_map=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing the filtered data for this split (train/val).
            root_dir (str): Path to the 'OCTDL_Cleaned' folder.
            transform (callable, optional): Transform to apply to the image.
            disease_map (dict): Mapping {'AMD': 0, 'DME': 1, ...}.
            condition_map (dict): Mapping {'CNV': 0, ...}.
        """
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.disease_map = disease_map
        self.condition_map = condition_map

        # Determine the ignore index (usually -100 for CrossEntropyLoss in PyTorch)
        self.IGNORE_INDEX = -100

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Construct Image Path
        # Structure: root_dir / disease_name / filename
        img_name = row['file_name']
        disease_folder = row['disease']
        img_path = os.path.join(self.root_dir, disease_folder, img_name)

        # 2. Load Image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # 3. Get Disease Label
        disease_str = str(row['label_disease'])
        label_disease = self.disease_map[disease_str]

        # 4. Get Condition Label
        # If the condition is not in our map (e.g. it was filtered out or marked IGNORE), return -100
        condition_str = str(row['label_condition_raw'])

        if condition_str in self.condition_map:
            label_condition = self.condition_map[condition_str]
        else:
            label_condition = self.IGNORE_INDEX

        return image, label_disease, label_condition


def get_data_splits(csv_path, test_size=0.2, val_size=0.1):
    """
    Reads the processed CSV and splits data based on Patient ID to prevent leakage.

    Returns:
        train_df, val_df, test_df (DataFrames)
        disease_map (dict)
        condition_map (dict)
    """
    print(f"Loading metadata from: {csv_path}")
    df = pd.read_csv(csv_path)

    # --- 1. Generate Mappings ---
    # Disease Mapping
    unique_diseases = sorted(df['label_disease'].astype(str).unique())
    disease_map = {name: i for i, name in enumerate(unique_diseases)}

    # Condition Mapping
    # We only map conditions that are NOT "IGNORE"
    valid_conditions = sorted(df[df['label_condition_raw'] != 'IGNORE']['label_condition_raw'].unique())
    condition_map = {name: i for i, name in enumerate(valid_conditions)}

    print(f"   Classes (Disease): {disease_map}")
    print(f"   Classes (Condition): {condition_map}")

    # --- 2. Patient-Level Split ---
    # Get unique patients and their primary disease label for stratification
    patients = df[['patient_id', 'label_disease']].drop_duplicates()

    # First split: Train vs (Val + Test)
    # If val_size + test_size = 0.3, then split is 70% Train / 30% Temp
    total_test_val_ratio = test_size + val_size
    train_pat, temp_pat = train_test_split(
        patients,
        test_size=total_test_val_ratio,
        random_state=42,
        stratify=patients['label_disease']
    )

    # Second split: Val vs Test
    # Adjust ratio relative to the temp set
    # Ex: if Test=0.2 and Val=0.1, Test is 2/3 of Temp.
    relative_test_ratio = test_size / total_test_val_ratio
    val_pat, test_pat = train_test_split(
        temp_pat,
        test_size=relative_test_ratio,
        random_state=42,
        stratify=temp_pat['label_disease']
    )

    # Create DataFrames based on Patient IDs
    train_df = df[df['patient_id'].isin(train_pat['patient_id'])]
    val_df = df[df['patient_id'].isin(val_pat['patient_id'])]
    test_df = df[df['patient_id'].isin(test_pat['patient_id'])]

    print(f"Split Complete:")
    print(f"   Train: {len(train_df)} images ({len(train_pat)} patients)")
    print(f"   Val:   {len(val_df)} images ({len(val_pat)} patients)")
    print(f"   Test:  {len(test_df)} images ({len(test_pat)} patients)")

    return train_df, val_df, test_df, disease_map, condition_map