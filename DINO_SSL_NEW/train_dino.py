import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import wandb
from dataset_dino import DINO_Dataset
DEFAULT_DATA_ROOT = r"C:\Datasets\OCTDL_Cleaned"
SAVE_DIR = "checkpoints_domain_adapt"
class DINOHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, bottleneck_dim=256, out_dim=65536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class DINOModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, dict):
            features = features["x_norm_clstoken"]
        elif isinstance(features, tuple):
            features = features[0]
        return self.head(features)

class DINOLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_outputs, teacher_outputs):
        total_loss = 0.0
        n_loss_terms = 0

        teacher_probs = [
            F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
            for t in teacher_outputs
        ]

        student_log_probs = [
            F.log_softmax(s / self.student_temp, dim=-1)
            for s in student_outputs
        ]

        for iq, tq in enumerate(teacher_probs):
            for iv, sv in enumerate(student_log_probs):
                if iv == iq:
                    continue
                loss = torch.sum(-tq * sv, dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(torch.cat(teacher_outputs, dim=0))
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

def update_teacher(student, teacher, momentum):
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(momentum).add_(ps.data * (1.0 - momentum))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--local_crops_number', type=int, default=6)
    parser.add_argument('--ema_momentum', type=float, default=0.996)
    args = parser.parse_args()

    wandb.init(project="Licenta-SSL-DINOStyle", config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = DINO_Dataset(
        args.data_path,
        global_size=224,
        local_size=96,
        local_crops_number=args.local_crops_number
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    print("Loading DINOv2-BASE (ViT-B/14)...")

    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    embed_dim = 768
    student = DINOModel(backbone, DINOHead(embed_dim)).to(device)
    teacher = copy.deepcopy(student).to(device)

    for param in student.parameters():
        param.requires_grad = True

    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    dino_loss = DINOLoss(out_dim=65536).to(device)

    print("Starting DINO-Style Pre-training...")
    for epoch in range(args.epochs):
        student.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Ep {epoch + 1}/{args.epochs}")

        for crops in pbar:
            crops = [x.to(device, non_blocking=True) for x in crops]
            global_crops = crops[:2]
            local_crops = crops[2:]

            with torch.no_grad():
                teacher_outputs = [teacher(crop) for crop in global_crops]

            student_outputs = [student(crop) for crop in crops]
            loss = dino_loss(student_outputs, teacher_outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_teacher(student, teacher, args.ema_momentum)

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        wandb.log({"ssl_loss": avg_loss, "epoch": epoch + 1})

        torch.save(
            {
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "epoch": epoch + 1,
            },
            os.path.join(args.save_dir, "dinov2_base_adapted_latest.pth")
        )
        print(f"Saved Epoch {epoch + 1}. Avg Loss: {total_loss / len(dataloader):.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()