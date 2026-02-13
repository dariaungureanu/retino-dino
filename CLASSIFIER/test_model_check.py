import torch
import os
from model import OCTDLMultiTaskModel

# --- CONFIGURARE ---
# Pune calea exactă către fișierul tău .pth de 4.8GB
CHECKPOINT_PATH = r"/saved_models/model_final.rank_0.pth"

def test_model():
    print("--- 1. TEST ÎNCĂRCARE CHECKPOINT ---")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ EROARE: Nu găsesc fișierul la: {CHECKPOINT_PATH}")
        return

    try:
        # Inițializăm modelul (asta apelează funcția de curățare a cheilor)
        # Setez num_diseases=7 și num_conditions=8 (sau câte ai tu, verifică dataset_check output)
        model = OCTDLMultiTaskModel(
            checkpoint_path=CHECKPOINT_PATH,
            num_diseases=7,
            num_conditions=8,
            freeze_backbone=True
        )
        print("✅ Modelul a fost instanțiat cu succes!")
        print(f"   Backbone type: ViT-Large (DINOv2)")
    except Exception as e:
        print(f"❌ EROARE CRITICĂ la încărcare: {e}")
        return

    print("\n--- 2. TEST FORWARD PASS (Dimensiuni) ---")
    try:
        # Creăm un "batch" fals de 2 imagini random
        # Dimensiunea 518x518 este standard pentru DINOv2 Large
        dummy_input = torch.randn(2, 3, 224, 224)
        print(f"   Input shape: {dummy_input.shape}")

        # Trecem datele prin model
        out_disease, out_condition = model(dummy_input)

        # Verificăm dimensiunile output-ului
        # Ne așteptăm la [Batch_Size, Num_Classes]
        print(f"✅ Forward pass reușit!")
        print(f"   Output Disease Shape:   {out_disease.shape} (Așteptat: [2, 7])")
        print(f"   Output Condition Shape: {out_condition.shape} (Așteptat: [2, 8])")

    except Exception as e:
        print(f"❌ EROARE la procesare (Forward): {e}")
        return

    print("\n🎉 SUPER! Modelul este gata de antrenare.")

if __name__ == "__main__":
    test_model()