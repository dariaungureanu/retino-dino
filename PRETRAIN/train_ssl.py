import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from dataset_ssl import OCTDL_SSL_Dataset

# --- CONFIGURARE DEFAULT ---
DEFAULT_DATA_ROOT = r"C:\Datasets\OCTDL_Cleaned"
SAVE_DIR = "checkpoints_ssl"


class SimpleDINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 256),
        )
        self.last_layer = nn.Linear(256, out_dim, bias=False)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


def main():
    # Argumente flexibile (le poți schimba din terminal)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR)
    parser.add_argument('--epochs', type=int, default=1)  # Default mic pt test
    parser.add_argument('--batch_size', type=int, default=2)  # Default mic pt laptop
    parser.add_argument('--lr', type=float, default=5e-6)
    args = parser.parse_args()

    # Inițializare WandB
    wandb.init(project="Licenta-SSL-Test", config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Dataset
    print(f"📂 Loading data from: {args.data_path}")
    dataset = OCTDL_SSL_Dataset(args.data_path)

    # Drop_last=True e important ca să nu avem batch-uri incomplete care dau erori la norme
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                            drop_last=True)
    print(f"✅ Loaded {len(dataset)} images.")

    # 2. Model (DINOv2 Large)
    print("⬇️ Downloading/Loading DINOv2 ViT-Large...")
    student = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)

    # Freeze layers (antrenăm doar ultimul bloc)
    for name, param in student.named_parameters():
        if "blocks.23" not in name and "norm" not in name:
            param.requires_grad = False
    print("❄️ Backbone frozen (except last block).")

    # Projection Head
    head = SimpleDINOHead(1024).to(device)

    optimizer = optim.AdamW(list(student.parameters()) + list(head.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("🏁 Starting Pre-training...")
    student.train()

    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Ep {epoch + 1}/{args.epochs}")

        count = 0
        for g1, g2 in pbar:
            g1, g2 = g1.to(device), g2.to(device)

            optimizer.zero_grad()

            # Forward
            feat1 = student(g1)
            feat2 = student(g2)

            out1 = head(feat1)
            out2 = head(feat2)

            # Loss
            labels = torch.arange(len(g1)).to(device)
            logits = torch.matmul(out1, out2.T)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            # PE LAPTOP: Oprim după 5 batch-uri ca să nu stăm o oră
            if device.type == 'cpu':
                count += 1
                if count >= 5:
                    print("⚠️ CPU Mode: Skipping rest of epoch for speed test.")
                    break

        avg_loss = total_loss / (len(dataloader) if device.type != 'cpu' else 5)
        wandb.log({"ssl_loss": avg_loss, "epoch": epoch})

        # Save Checkpoint
        save_path = os.path.join(args.save_dir, "checkpoint_latest.pth")
        torch.save(student.state_dict(), save_path)
        print(f"💾 Checkpoint saved: {save_path}")

    print("✅ TEST SUCCESSFUL! Script works.")
    wandb.finish()


if __name__ == "__main__":
    main()