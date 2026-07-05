import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from dataset_ssl import OCTDL_SSL_Dataset

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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-6)
    args = parser.parse_args()

    wandb.init(project="Licenta-SSL-Turbo", config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = OCTDL_SSL_Dataset(args.data_path)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    print(f"loaded {len(dataset)} images. Batch size: {args.batch_size}")

    print("loading DINOv2 ViT-Large...")
    student = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)

    for name, param in student.named_parameters():
        if "blocks.23" not in name and "norm" not in name:
            param.requires_grad = False
    print("backbone frozen (except last block).")

    head = SimpleDINOHead(1024).to(device)
    optimizer = optim.AdamW(list(student.parameters()) + list(head.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("starting pre-training...")
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

            save_path = os.path.join(args.save_dir, "checkpoint_latest.pth")
            torch.save(student.state_dict(), save_path)
            print(f"saved: {save_path}")

    except torch.cuda.OutOfMemoryError:
        print("\ncritical: CUDA out of memory. Saving emergency checkpoint...")
        torch.save(student.state_dict(), os.path.join(args.save_dir, "checkpoint_oom.pth"))
        print("emergency save successful: checkpoint_oom.pth")
        torch.cuda.empty_cache()
        raise

    except KeyboardInterrupt:
        print("\ntraining interrupted by user. Saving checkpoint...")
        torch.save(student.state_dict(), os.path.join(args.save_dir, "checkpoint_interrupted.pth"))
        print("save successful.")

    print("pre-training complete.")
    wandb.finish()


if __name__ == "__main__":
    main()
