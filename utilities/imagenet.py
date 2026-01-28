import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets


class ImageNet1k(Dataset):
    """ImageNet-1k dataset wrapper with dict output compatible with VOC/COCO."""

    def __init__(self, root, phase="train", transform=None):
        super().__init__()
        self.root = os.path.abspath(root)
        self.phase = phase
        split_dir = os.path.join(self.root, phase)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"ImageNet split folder not found: {split_dir}")

        self.transform = transform
        self._folder = datasets.ImageFolder(split_dir, transform=None)
        self.num_classes = len(self._folder.classes)

    def __len__(self):
        return len(self._folder)

    def get_number_classes(self):
        return self.num_classes

    def __getitem__(self, index):
        img, label = self._folder[index]
        target = torch.zeros(self.num_classes, dtype=torch.float32) - 1
        target[label] = 1

        if self.transform is not None:
            img = self.transform(img)

        path, _ = self._folder.samples[index]
        name = os.path.relpath(path, self.root)
        return {"image": img, "name": name, "target": target}

    def get_number_pClasses(self):
        counts = torch.zeros(self.num_classes)
        for _, label in self._folder.samples:
            counts[label] += 1
        return counts, {i: counts[i] for i in range(self.num_classes)}
