"""FFHQ and CelebA image discovery and data-loader construction."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class FaceImages(Dataset):
    """Load aligned face images from FFHQ or CelebA directory layouts."""

    def __init__(
        self,
        root: str | Path,
        dataset_name: str,
        image_size: int,
        max_images: int | None = None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.dataset_name = dataset_name
        if not self.root.exists():
            raise FileNotFoundError(
                f"{dataset_name.upper()} root does not exist: {self.root}\n"
                "Download/extract the images there or pass --data-root."
            )

        preferred = (
            [
                self.root / "images1024x1024",
                self.root / "images",
                self.root / "ffhq" / "images1024x1024",
            ]
            if dataset_name == "ffhq"
            else [
                self.root / "img_align_celeba",
                self.root / "images",
                self.root / "celeba" / "img_align_celeba",
            ]
        )
        self.image_root = next(
            (path for path in preferred if path.is_dir()), self.root
        )
        self.files = sorted(
            path
            for path in self.image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if max_images is not None:
            self.files = self.files[:max_images]
        if not self.files:
            raise RuntimeError(f"No jpg/png images found below {self.image_root}")

        crop = (
            transforms.CenterCrop(178)
            if dataset_name == "celeba"
            else transforms.Lambda(lambda image: image)
        )
        self.transform = transforms.Compose(
            [
                crop,
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
    dataset_name: str,
    data_root: str | Path,
    image_size: int,
    max_images: int | None,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = FaceImages(data_root, dataset_name, image_size, max_images)
    print(
        f"Found {len(dataset):,} {dataset_name.upper()} images under "
        f"{dataset.image_root}"
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
