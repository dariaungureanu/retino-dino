import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class OCTDL_SSL_Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))

        print(f"✅ [SSL Dataset] Found {len(self.image_paths)} images.")

        self.img_size = 224

        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.4, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size, scale=(0.4, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            path = self.image_paths[idx]
            image = Image.open(path).convert("RGB")

            g1 = self.global_transfo1(image)
            g2 = self.global_transfo2(image)
            return g1, g2
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))