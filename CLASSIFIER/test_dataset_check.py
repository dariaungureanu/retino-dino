import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Importăm funcțiile din scriptul tău dataset.py
from dataset import get_data_splits, OCTDLMultiTaskDataset

# --- CONFIGURARE ---
# Pune calea unde ai generat datele curățate (output-ul de la pasul anterior)
ROOT_DIR = r"C:\Datasets\OCTDL_Cleaned"
CSV_PATH = f"{ROOT_DIR}/OCTDL_clean_metadata.csv"


def test_pipeline():
    print("--- 1. TEST FUNCȚIA DE SPLIT ---")
    try:
        train_df, val_df, test_df, disease_map, condition_map = get_data_splits(CSV_PATH)
        print(f"✅ Split reușit!")
        print(f"   Train samples: {len(train_df)}")
        print(f"   Mapare Boli: {disease_map}")
        print(f"   Mapare Condiții: {condition_map}")
    except Exception as e:
        print(f"❌ Eroare la split: {e}")
        return

    print("\n--- 2. TEST DATASET CLASS ---")
    # Transformare simplă doar pentru test
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    try:
        # Instanțiem dataset-ul de Train
        train_ds = OCTDLMultiTaskDataset(
            dataframe=train_df,
            root_dir=ROOT_DIR,
            transform=transform,
            disease_map=disease_map,
            condition_map=condition_map
        )
        print(f"✅ Dataset inițializat. Lungime: {len(train_ds)}")
    except Exception as e:
        print(f"❌ Eroare la inițializarea clasei Dataset: {e}")
        return

    print("\n--- 3. TEST ÎNCĂRCARE ITEM (Get Item) ---")
    try:
        # Luăm prima imagine din dataset
        img, label_dis, label_cond = train_ds[0]

        print(f"✅ Imagine încărcată cu succes!")
        print(f"   Shape Tensor: {img.shape} (Trebuie să fie [3, 224, 224])")
        print(f"   Label Disease (int): {label_dis}")
        print(f"   Label Condition (int): {label_cond}")

        # Verificăm tipurile
        if not isinstance(label_dis, int) or not isinstance(label_cond, int):
            print("⚠️ ATENȚIE: Label-urile nu sunt int!")
        else:
            print("   Tipuri label corecte (int).")

    except Exception as e:
        print(f"❌ Eroare la __getitem__: {e}")
        # Printează calea imaginii care a crăpat, dacă e posibil
        print(f"   Încercam să încarc: {train_ds.df.iloc[0]['file_name']}")
        return

    print("\n--- 4. TEST DATALOADER (Batching) ---")
    try:
        # Încercăm să facem un batch de 4 imagini
        loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        images, labels_d, labels_c = next(iter(loader))

        print(f"✅ Batch încărcat cu succes!")
        print(f"   Batch Images Shape: {images.shape} (Trebuie [4, 3, 224, 224])")
        print(f"   Batch Disease Labels: {labels_d}")
        print(f"   Batch Condition Labels: {labels_c}")
    except Exception as e:
        print(f"❌ Eroare la DataLoader: {e}")
        return

    print("\n🎉 SUPER! Dataset-ul funcționează perfect.")


if __name__ == "__main__":
    test_pipeline()