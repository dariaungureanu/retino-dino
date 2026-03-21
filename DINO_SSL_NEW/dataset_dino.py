import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DINO_Dataset(Dataset):
    def __init__(self, root_dir, global_size=224, local_size=98, local_crops_number=4):
        self.root_dir = root_dir
        self.local_crops_number = local_crops_number
        self.image_paths = []

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        print(f"[SSL DINO Dataset] Found {len(self.image_paths)} images.")

        #global crop 1
        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter( brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        #global crop 2 - diff augmentation
        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        #local crop
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=(0.05, 0.4)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter( brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")

        crops = []
        # 2 global crops
        crops.append(self.global_transform1(img))
        crops.append(self.global_transform2(img))

        # N local crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(img))

        return crops