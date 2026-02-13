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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR)
    parser.add_argument('--epochs', type=int, default=20)

    # --- MODIFICARE: Batch Size default marit la 16 sau 32 ---
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-6)
    args = parser.parse_args()

    # Re-init WandB cu nume nou ca să nu amestecăm cu run-ul crăpat
    wandb.init(project="Licenta-SSL-Turbo", config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Dataset
    dataset = OCTDL_SSL_Dataset(args.data_path)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # <<-- SECRETUL VITEZEI
        pin_memory=True,
        drop_last=True,
        persistent_workers=True  # <<-- Ține workers activi
    )
    print(f"✅ Loaded {len(dataset)} images. Batch Size: {args.batch_size}")

    # 2. Model
    print("⬇️ Loading DINOv2 ViT-Large...")
    student = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)

    for name, param in student.named_parameters():
        if "blocks.23" not in name and "norm" not in name:
            param.requires_grad = False
    print("❄️ Backbone frozen (except last block).")

    head = SimpleDINOHead(1024).to(device)
    optimizer = optim.AdamW(list(student.parameters()) + list(head.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("🏁 Starting Pre-training (Turbo Mode)...")
    student.train()
    try:
        for epoch in range(args.epochs):
            total_loss = 0
            pbar = tqdm(dataloader, desc=f"Ep {epoch + 1}/{args.epochs}")

            for g1, g2 in pbar:
                g1, g2 = g1.to(device, non_blocking=True), g2.to(device, non_blocking=True)

                optimizer.zero_grad()
                feat1 = student(g1)
                feat2 = student(g2)
                out1 = head(feat1)
                out2 = head(feat2)

                labels = torch.arange(len(g1)).to(device)
                logits = torch.matmul(out1, out2.T)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(dataloader)
            wandb.log({"ssl_loss": avg_loss, "epoch": epoch})

            # Save every epoch
            save_path = os.path.join(args.save_dir, "checkpoint_latest.pth")
            torch.save(student.state_dict(), save_path)
            print(f"💾 Saved: {save_path}")

    except torch.cuda.OutOfMemoryError:
        print("\n⚠️ CRITICAL: CUDA Out Of Memory! Saving emergency checkpoint...")
        torch.save(student.state_dict(), os.path.join(args.save_dir, "checkpoint_oom.pth"))
        print("💾 Emergency save successful: checkpoint_oom.pth")
        torch.cuda.empty_cache()
        raise  # Aruncăm eroarea ca să știm că a crăpat, dar datele sunt salvate

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user. Saving checkpoint...")
        torch.save(student.state_dict(), os.path.join(args.save_dir, "checkpoint_interrupted.pth"))
        print("💾 Save successful.")

    print("✅ Done!")
    wandb.finish()


if __name__ == "__main__":
    main()  # Aceasta este cheia pe Windows pentru num_workers > 0