"""Train an amortized latent generator from sliced Riesz-flow targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.utils import save_image
from tqdm.auto import tqdm

from dataset import make_loader
from models import PretrainedVAE, ResidualLatentGenerator, SmallLatentDiT
from utils.mmd import (
    make_random_projections,
    sliced_riesz_mmd_squared,
    sliced_riesz_velocity,
)


@torch.no_grad()
def estimate_latent_stats(
    vae: PretrainedVAE,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_images: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(1, vae.latent_dim, device=device, dtype=torch.float64)
    total_squared = torch.zeros_like(total)
    count = 0
    progress = tqdm(loader, desc="Estimating latent statistics", unit="batch")
    for images in progress:
        latents = vae.encode(images.to(device, non_blocking=True)).double()
        remaining = max_images - count
        latents = latents[:remaining]
        total += latents.sum(0, keepdim=True)
        total_squared += latents.square().sum(0, keepdim=True)
        count += latents.shape[0]
        if count >= max_images:
            break
    if count < 2:
        raise RuntimeError("At least two images are needed for latent statistics")
    mean = total / count
    variance = total_squared / count - mean.square()
    return mean.float(), variance.clamp_min(1e-6).sqrt().float()


@torch.no_grad()
def save_generated_grid(
    vae: PretrainedVAE,
    generator: nn.Module,
    noise: torch.Tensor,
    mean: torch.Tensor,
    standard_deviation: torch.Tensor,
    path: Path,
) -> None:
    generator.eval()
    latents = generator(noise) * standard_deviation + mean
    images = vae.decode(latents)[:15]
    save_image((images.cpu() + 1) / 2, path, nrow=5)


@torch.no_grad()
def save_reconstruction_check(
    vae: PretrainedVAE, images: torch.Tensor, device: torch.device, path: Path
) -> None:
    originals = images[:5].to(device)
    reconstructions = vae.decode(vae.encode(originals))
    comparison = torch.cat((originals, reconstructions), dim=0)
    save_image((comparison.cpu() + 1) / 2, path, nrow=5)


def save_mmd_curve(history: list[dict[str, float | int]], path: Path) -> None:
    epochs = [int(item["epoch"]) for item in history]
    values = [float(item["mmd_squared"]) for item in history]
    width, height = 900, 600
    left, right, top, bottom = 90, 30, 50, 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    value_min, value_max = min(values), max(values)
    value_span = max(value_max - value_min, 1e-12)
    epoch_span = max(max(epochs) - min(epochs), 1)
    points = [
        (
            left + (epoch - min(epochs)) / epoch_span * (width - left - right),
            top + (value_max - value) / value_span * (height - top - bottom),
        )
        for epoch, value in zip(epochs, values)
    ]
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
    if len(points) > 1:
        draw.line(points, fill="#1f77b4", width=3)
    for x, y in points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#1f77b4")
    draw.text((width // 2 - 75, 15), "Generator MMD convergence", fill="black")
    draw.text((width // 2 - 30, height - 35), "Epoch", fill="black")
    draw.text((8, 18), "Sliced Riesz MMD squared", fill="black")
    draw.text((left - 75, top - 7), f"{value_max:.4g}", fill="black")
    draw.text((left - 75, height - bottom - 7), f"{value_min:.4g}", fill="black")
    for epoch, (x, _) in zip(epochs, points):
        draw.text((x - 4, height - bottom + 10), str(epoch), fill="black")
    image.save(path)


def _checkpoint(
    generator: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    mean: torch.Tensor,
    standard_deviation: torch.Tensor,
    fixed_noise: torch.Tensor,
    diagnostic_noise: torch.Tensor,
    diagnostic_projections: torch.Tensor,
    reference_targets: torch.Tensor,
    history: list[dict[str, float | int]],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "generator": generator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "latent_mean": mean.cpu(),
        "latent_standard_deviation": standard_deviation.cpu(),
        "fixed_noise": fixed_noise.cpu(),
        "diagnostic_noise": diagnostic_noise.cpu(),
        "diagnostic_projections": diagnostic_projections.cpu(),
        "reference_targets": reference_targets.cpu(),
        "history": history,
        "config": {
            "latent_dim": int(generator.latent_dim),
            "architecture": args.generator_arch,
            "hidden_dim": args.generator_hidden,
            "depth": args.generator_depth,
            "dit_hidden_dim": args.dit_hidden,
            "dit_depth": args.dit_depth,
            "dit_heads": args.dit_heads,
            "dit_patch_size": args.dit_patch_size,
            "vae_model": args.vae_model,
            "image_size": args.image_size,
            "dataset": args.dataset,
        },
    }


def train_generator(args: argparse.Namespace, device: torch.device) -> Path:
    print(f"Loading frozen VAE: {args.vae_model}")
    vae = PretrainedVAE(
        args.vae_model, args.image_size, device, args.local_files_only
    )
    loader = make_loader(
        dataset_name=args.dataset,
        data_root=args.data_root,
        image_size=args.image_size,
        max_images=args.max_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    try:
        reference_images = next(iter(loader))
    except StopIteration as error:
        raise RuntimeError(f"The {args.dataset.upper()} loader yielded no images") from error
    save_reconstruction_check(
        vae, reference_images, device, args.output_dir / "vae_reconstructions.png"
    )

    state = None
    if args.resume:
        state = torch.load(
            args.resume.expanduser(), map_location=device, weights_only=False
        )
    selected_architecture = (
        state["config"].get("architecture", "mlp")
        if state
        else ("dit" if args.generator_arch == "auto" and args.dataset == "celeba" else args.generator_arch)
    )
    if selected_architecture == "auto":
        selected_architecture = "mlp"
    args.generator_arch = selected_architecture
    generator_hidden = int(state["config"]["hidden_dim"]) if state else args.generator_hidden
    generator_depth = int(state["config"]["depth"]) if state else args.generator_depth
    args.generator_hidden = generator_hidden
    args.generator_depth = generator_depth
    if selected_architecture == "dit":
        args.dit_hidden = int(state["config"]["dit_hidden_dim"]) if state else args.dit_hidden
        args.dit_depth = int(state["config"]["dit_depth"]) if state else args.dit_depth
        args.dit_heads = int(state["config"]["dit_heads"]) if state else args.dit_heads
        args.dit_patch_size = int(state["config"]["dit_patch_size"]) if state else args.dit_patch_size
        generator = SmallLatentDiT(
            vae.latent_channels,
            vae.latent_spatial,
            args.dit_hidden,
            args.dit_depth,
            args.dit_heads,
            args.dit_patch_size,
        ).to(device)
    else:
        generator = ResidualLatentGenerator(
            vae.latent_dim, generator_hidden, generator_depth
        ).to(device)
    parameter_count = sum(parameter.numel() for parameter in generator.parameters())
    print(f"Using {selected_architecture.upper()} generator ({parameter_count:,} parameters)")
    optimizer = torch.optim.AdamW(
        generator.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = 0
    if state is not None:
        generator.load_state_dict(state["generator"])
        optimizer.load_state_dict(state["optimizer"])
        mean = state["latent_mean"].to(device)
        standard_deviation = state["latent_standard_deviation"].to(device)
        start_epoch = int(state["epoch"]) + 1
    else:
        stats_count = min(args.stats_images, len(loader.dataset))
        mean, standard_deviation = estimate_latent_stats(
            vae, loader, device, stats_count
        )

    if state is not None:
        fixed_noise = state["fixed_noise"].to(device)
        diagnostic_noise = state["diagnostic_noise"].to(device)
        diagnostic_projections = state["diagnostic_projections"].to(device)
        reference_targets = state["reference_targets"].to(device)
        history = state["history"]
    else:
        fixed_noise = torch.randn(args.num_samples, vae.latent_dim, device=device)
        diagnostic_noise = torch.randn(
            args.particles_per_step, vae.latent_dim, device=device
        )
        with torch.no_grad():
            reference_targets = vae.encode(reference_images.to(device)).float()
            reference_targets = (reference_targets - mean) / standard_deviation
            diagnostic_projections = make_random_projections(
                diagnostic_noise, args.num_projections
            )
            initial_mmd = sliced_riesz_mmd_squared(
                generator(diagnostic_noise),
                reference_targets,
                args.num_projections,
                diagnostic_projections,
            )
        history: list[dict[str, float | int]] = [
            {"epoch": 0, "mmd_squared": initial_mmd, "loss": 0.0}
        ]
    save_generated_grid(
        vae,
        generator,
        fixed_noise,
        mean,
        standard_deviation,
        args.output_dir / "generator_initial.png",
    )

    checkpoint_path = args.output_dir / "generator_latest.pt"
    for epoch in range(start_epoch, args.epochs):
        generator.train()
        running_loss = 0.0
        progress = tqdm(
            loader, desc=f"Generator epoch {epoch + 1}/{args.epochs}", unit="batch"
        )
        for step, images in enumerate(progress, start=1):
            with torch.no_grad():
                targets = vae.encode(images.to(device, non_blocking=True)).float()
                targets = (targets - mean) / standard_deviation
            noise = torch.randn(
                args.particles_per_step, vae.latent_dim, device=device
            )
            generated = generator(noise)
            with torch.no_grad():
                velocity = sliced_riesz_velocity(
                    generated.detach(), targets, args.num_projections
                )
                norms = velocity.norm(dim=1, keepdim=True)
                scale = (
                    args.velocity_clip / norms.clamp_min(args.velocity_clip)
                ).clamp(max=1)
                regression_target = generated.detach() + args.flow_lr * velocity * scale
            loss = F.mse_loss(generated, regression_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
            optimizer.step()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{running_loss / step:.5f}")

        generator.eval()
        with torch.no_grad():
            mmd_value = sliced_riesz_mmd_squared(
                generator(diagnostic_noise),
                reference_targets,
                args.num_projections,
                diagnostic_projections,
            )
        average_loss = running_loss / len(loader)
        history.append(
            {
                "epoch": epoch + 1,
                "mmd_squared": mmd_value,
                "loss": average_loss,
            }
        )
        save_generated_grid(
            vae,
            generator,
            fixed_noise,
            mean,
            standard_deviation,
            args.output_dir / f"generator_epoch_{epoch + 1:03d}.png",
        )
        torch.save(
            _checkpoint(
                generator,
                optimizer,
                epoch,
                mean,
                standard_deviation,
                fixed_noise,
                diagnostic_noise,
                diagnostic_projections,
                reference_targets,
                history,
                args,
            ),
            checkpoint_path,
        )
        (args.output_dir / "mmd_flow_metrics.json").write_text(
            json.dumps({"history": history}, indent=2)
        )
        save_mmd_curve(history, args.output_dir / "mmd_curve.png")
        print(
            f"epoch {epoch + 1:03d}/{args.epochs:03d} "
            f"loss={average_loss:.6f} MMD^2={mmd_value:.6f}"
        )

    print(f"Saved generator checkpoint to {checkpoint_path}")
    return checkpoint_path
