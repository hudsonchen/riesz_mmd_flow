#!/usr/bin/env python3
"""Train a one-step face generator with Riesz-kernel MMD flow targets."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from generator_training import train_generator


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--autoencoder", choices=("alae", "sd-vae"), default="alae"
    )
    parser.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument(
        "--alae-source",
        type=Path,
        default=PROJECT_ROOT / "third_party/ALAE",
    )
    parser.add_argument(
        "--alae-checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument("--alae-latent-cache", type=Path, default=None)
    parser.add_argument("--encoding-batch-size", type=int, default=4)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--dataset", choices=("ffhq", "celeba"), default="celeba")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--stats-images", type=int, default=4096)
    parser.add_argument(
        "--generator-arch", choices=("auto", "dit", "mlp"), default="auto"
    )
    parser.add_argument("--generator-hidden", type=int, default=1024)
    parser.add_argument("--generator-depth", type=int, default=4)
    parser.add_argument("--dit-hidden", type=int, default=192)
    parser.add_argument("--dit-depth", type=int, default=6)
    parser.add_argument("--dit-heads", type=int, default=6)
    parser.add_argument("--dit-patch-size", type=int, default=2)
    parser.add_argument("--particles-per-step", type=int, default=64)
    parser.add_argument("--flow-lr", type=float, default=1.0)
    parser.add_argument("--velocity-clip", type=float, default=2.0)
    parser.add_argument("--riesz-epsilon", type=float, default=1e-8)
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.data_root is None:
        args.data_root = (
            Path("/home/zongchen/MMD_Wflow/data/ffhq-dataset")
            if args.dataset == "ffhq"
            else Path("/home/zongchen/MMD_Wflow/data") / args.dataset
        )
    args.data_root = args.data_root.expanduser().resolve()
    args.alae_source = args.alae_source.expanduser().resolve()
    if args.autoencoder == "alae":
        if args.image_size is None:
            args.image_size = 1024 if args.dataset == "ffhq" else 128
        if args.alae_checkpoint is None:
            args.alae_checkpoint = (
                args.alae_source
                / "training_artifacts"
                / args.dataset
                / ("model_submitted.pth" if args.dataset == "ffhq" else "model_final.pth")
            )
        if args.max_images is None:
            args.max_images = 60000 if args.dataset == "ffhq" else 182637
    elif args.image_size is None:
        args.image_size = 64
    if args.alae_checkpoint is not None:
        args.alae_checkpoint = args.alae_checkpoint.expanduser().resolve()
    if args.alae_latent_cache is None:
        count_label = args.max_images if args.max_images is not None else "all"
        args.alae_latent_cache = (
            PROJECT_ROOT / ".cache" / f"{args.dataset}_alae_{count_label}.pt"
        )
    args.alae_latent_cache = args.alae_latent_cache.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = Path("outputs_new") / f"{args.dataset}_mmd_flow"
    args.output_dir = args.output_dir.expanduser().resolve()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    seed_everything(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    train_generator(args, device)


if __name__ == "__main__":
    main()
