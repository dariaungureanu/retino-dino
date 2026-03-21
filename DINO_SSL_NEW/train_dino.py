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
import numpy as np
from dataset_dino import DINO_Dataset
DEFAULT_DATA_ROOT = r"C:\Datasets\OCTDL_Cleaned"
# SAVE_DIR = "checkpoints_dino_40k"
SAVE_DIR = "checkpoints_dino_oct_optimized"

def get_parameter_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        #if bias or layernorm, no weight decay
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.0}]

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

class DINOHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, bottleneck_dim=256, out_dim=65536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = torch.nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False)) #weight normalization to stabilize 65536-dim output space
        self.last_layer.weight_g.requires_grad = False #learn direction, not magnitude

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
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--local_crops_number', type=int, default=6)
    parser.add_argument('--ema_momentum', type=float, default=0.996)
    args = parser.parse_args()

    wandb.init(project="Licenta-SSL-OCT-Optimized", config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = DINO_Dataset(
        args.data_path,
        global_size=224,
        local_size=98,
        local_crops_number=args.local_crops_number
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    print("Loading DINOv2-BASE (ViT-B/14)...")

    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    student = DINOModel(backbone, DINOHead(768)).to(device)
    teacher = copy.deepcopy(student).to(device)

    for param in student.parameters():
        param.requires_grad = True

    for p in teacher.parameters():
        p.requires_grad = False

    param_groups = get_parameter_groups(student)
    optimizer = optim.AdamW(param_groups, lr=args.base_lr, weight_decay=0.04)

    dino_loss = DINOLoss(out_dim=65536).to(device)

    scaler = torch.amp.GradScaler('cuda') #Mixed Precision Scaler for VRAM optimization

    #schedulers
    iters_per_epoch = len(dataloader)
    lr_schedule = cosine_scheduler(args.base_lr, args.min_lr, args.epochs, iters_per_epoch, warmup_epochs=5)
    wd_schedule = cosine_scheduler(0.04, 0.4, args.epochs, iters_per_epoch)
    momentum_schedule = cosine_scheduler(0.996, 1.0, args.epochs, iters_per_epoch)

    best_loss = float('inf')
    print("Starting DINO-Style Pre-training...")
    iteration = 0
    for epoch in range(args.epochs):
        student.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Ep {epoch + 1}/{args.epochs}")

        for crops in pbar:

            #update Optimizre params per iteration based on schedules
            current_it = min(iteration, len(lr_schedule) - 1)
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[current_it]
                if i == 0:
                    param_group["weight_decay"] = wd_schedule[current_it]

            crops = [x.to(device, non_blocking=True) for x in crops]

            #batch Forward Passes
            global_tensor = torch.cat(crops[:2])
            local_tensor = torch.cat(crops[2:])
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    teacher_global_out = teacher(global_tensor)
                    teacher_outputs = torch.chunk(teacher_global_out, 2)

                student_global_out = student(global_tensor)
                student_local_out = student(local_tensor)
                student_outputs = list(torch.chunk(student_global_out, 2)) + \
                                  list(torch.chunk(student_local_out, args.local_crops_number))

            student_outputs = [s.float() for s in student_outputs]
            teacher_outputs = [t.float() for t in teacher_outputs]
            loss = dino_loss(student_outputs, teacher_outputs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            update_teacher(student, teacher, momentum_schedule[current_it])

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{lr_schedule[current_it]:.6f}"})
            iteration += 1

        avg_loss = total_loss / len(dataloader)

        wandb.log({
            "ssl_loss": avg_loss,
            "epoch": epoch + 1,
            "lr": lr_schedule[iteration - 1],
            "wd": wd_schedule[iteration - 1],
            "momentum": momentum_schedule[iteration - 1]
        })

        torch.save(
            {
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "epoch": epoch + 1,
            },
            os.path.join(args.save_dir, "dinov2_oct_opt_latest.pth")
        )
        print(f"Saved Epoch {epoch + 1}. Avg Loss: {total_loss / len(dataloader):.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                student.state_dict(),
                os.path.join(args.save_dir, "dinov2_oct_opt_BEST.pth")
            )
            print(f"New: {best_loss:.4f}! Checkpoint BEST saved.")

    wandb.finish()

if __name__ == "__main__":
    main()