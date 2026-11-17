"""Dataset, preprocessing and normalization strategies (paper Sec. IV-D)."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ToGray:
    """Load any image mode as 8-bit grayscale (X-rays are single channel)."""

    def __call__(self, img):
        return img.convert("L")


class PerImageMinMax:
    """Secondary experiment: per-image min-max intensity normalization.
    Uses ONLY the image itself -- no target-dataset statistics."""

    def __call__(self, img):
        a = np.asarray(img, dtype=np.float32)
        lo, hi = float(a.min()), float(a.max())
        if hi - lo < 1e-6:
            return img
        a = (a - lo) / (hi - lo) * 255.0
        return Image.fromarray(a.astype(np.uint8))


def build_transform(train: bool, norm: str = "imagenet", size: int = 224):
    assert norm in ("imagenet", "minmax")
    ops = [ToGray()]
    if norm == "minmax":
        ops.append(PerImageMinMax())
    ops.append(T.Resize((size, size)))
    if train:
        # Mild augmentation only. No horizontal flip (would distort anatomy / laterality).
        ops += [
            T.RandomRotation(10),
            T.RandomAffine(0, translate=(0.05, 0.05)),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    ops += [
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return T.Compose(ops)


def display_image(img: Image.Image, norm: str, size: int = 224) -> np.ndarray:
    """RGB uint8 array of the image as the model sees it (for Grad-CAM overlays)."""
    img = ToGray()(img)
    if norm == "minmax":
        img = PerImageMinMax()(img)
    img = img.resize((size, size))
    return np.stack([np.asarray(img)] * 3, axis=-1)


class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).values
        self.tf = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i])
        img.load()
        return self.tf(img), int(self.labels[i])


def load_manifest(key: str, manifest_dir, split: str = "all") -> pd.DataFrame:
    df = pd.read_csv(Path(manifest_dir) / f"{key}.csv")
    if split != "all":
        df = df[df["split"] == split]
    return df.reset_index(drop=True)


def make_loader(df, train, norm="imagenet", batch_size=32, num_workers=4, size=224):
    ds = CXRDataset(df, build_transform(train, norm, size))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
