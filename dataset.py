"""Face and ImageNet-subset data-loader construction."""

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


class ImageNetSubset(Dataset):
    """Load the first sorted ImageNet class directories reproducibly."""

    def __init__(
        self,
        root: str | Path,
        image_size: int,
        num_classes: int,
        max_images: int | None = None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.image_root = self.root / "train" if (self.root / "train").is_dir() else self.root
        if not self.image_root.is_dir():
            raise FileNotFoundError(
                f"ImageNet root does not exist: {self.root}\n"
                "Pass --data-root pointing to ImageNet, arranged as "
                "train/<synset>/*.JPEG."
            )
        class_directories = sorted(path for path in self.image_root.iterdir() if path.is_dir())
        if len(class_directories) < num_classes:
            raise RuntimeError(
                f"Requested {num_classes} ImageNet classes, but found only "
                f"{len(class_directories)} under {self.image_root}"
            )
        self.classes = [path.name for path in class_directories[:num_classes]]
        self.files = [
            path
            for class_directory in class_directories[:num_classes]
            for path in sorted(class_directory.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if max_images is not None:
            self.files = self.files[:max_images]
        if not self.files:
            raise RuntimeError(f"No ImageNet images found below {self.image_root}")
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), antialias=True),
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
    imagenet_classes: int = 100,
    shuffle: bool = True,
) -> DataLoader:
    dataset = (
        ImageNetSubset(data_root, image_size, imagenet_classes, max_images)
        if dataset_name == "imagenet"
        else FaceImages(data_root, dataset_name, image_size, max_images)
    )
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
