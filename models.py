"""Frozen image representation and one-step latent generator."""

from __future__ import annotations

import torch
import torch.nn as nn


class PretrainedVAE(nn.Module):
    """Frozen Diffusers VAE adapter exposing flat deterministic latents."""

    def __init__(
        self,
        model_name: str,
        image_size: int,
        device: torch.device,
        local_files_only: bool,
    ):
        super().__init__()
        try:
            from diffusers.models import AutoencoderKL
        except ImportError as error:
            raise ImportError(
                "The pretrained encoder needs diffusers. Install requirements.txt."
            ) from error
        self.vae = AutoencoderKL.from_pretrained(
            model_name, local_files_only=local_files_only
        ).to(device)
        self.image_size = image_size
        self.latent_channels = int(self.vae.config.latent_channels)
        downsample_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.latent_spatial = image_size // downsample_factor
        self.latent_dim = self.latent_channels * self.latent_spatial**2
        self.scaling_factor = float(self.vae.config.scaling_factor)
        self.eval()
        self.requires_grad_(False)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.mean * self.scaling_factor
        return latents.flatten(1)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.view(
            -1, self.latent_channels, self.latent_spatial, self.latent_spatial
        )
        return self.vae.decode(latents / self.scaling_factor).sample.clamp(-1, 1)


class ResidualLatentGenerator(nn.Module):
    """One-step residual MLP mapping Gaussian noise to normalized VAE latents."""

    def __init__(self, latent_dim: int, hidden_dim: int = 1024, depth: int = 4):
        super().__init__()
        if depth < 2:
            raise ValueError("--generator-depth must be at least 2")
        layers: list[nn.Module] = [
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(depth - 2):
            layers.extend(
                (
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                )
            )
        output = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.residual = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return noise + self.residual(noise)
