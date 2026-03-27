from torchvision.datasets import ImageFolder
from typing import Callable, Optional

class OCTDataset(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform)
        print(f"===> OCTDataset Initialized: Found {len(self)} images in {root} <===")

    def __getitem__(self, index: int):
        image, _ = super().__getitem__(index)
        return image, 0