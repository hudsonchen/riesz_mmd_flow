#!/usr/bin/env python3
"""Train a one-step face generator with sliced Riesz-MMD flow targets."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from generator_training import train_generator


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--dataset", choices=("ffhq", "celeba"), default="celeba")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
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
    parser.add_argument("--particles-per-step", type=int, default=128)
    parser.add_argument("--flow-lr", type=float, default=1.0)
    parser.add_argument("--velocity-clip", type=float, default=2.0)
    parser.add_argument("--num-projections", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.data_root is None:
        args.data_root = Path("/home/zongchen/MMD_Wflow/data") / args.dataset
    args.data_root = args.data_root.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = Path("outputs") / f"{args.dataset}_mmd_flow"
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
