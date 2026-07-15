"""CelebA dataset discovery and data-loader construction."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class CelebAImages(Dataset):
    """Load CelebA images without requiring its optional annotation files."""

    def __init__(self, root: str | Path, image_size: int, max_images: int | None = None):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(
                f"CelebA root does not exist: {self.root}\n"
                "Pass the location explicitly with --data-root."
            )

        preferred = [
            self.root / "img_align_celeba",
            self.root / "images",
            self.root / "celeba" / "img_align_celeba",
        ]
        image_root = next((path for path in preferred if path.is_dir()), self.root)
        self.files = sorted(
            path
            for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if max_images is not None:
            self.files = self.files[:max_images]
        if not self.files:
            raise RuntimeError(
                f"No images found below {image_root}. Expected jpg/png files, usually in "
                "<data-root>/img_align_celeba/."
            )

        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB"))


def make_loader(
    *,
    data_root: str | Path,
    image_size: int,
    max_images: int | None,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = CelebAImages(data_root, image_size, max_images)
    print(f"Found {len(dataset):,} images under {dataset.root}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
