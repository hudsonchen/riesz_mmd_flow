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
        self.latent_dim = latent_dim
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


class SmallLatentDiT(nn.Module):
    """Compact residual DiT for spatial VAE latents."""

    def __init__(
        self,
        latent_channels: int,
        latent_spatial: int,
        hidden_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        patch_size: int = 2,
    ):
        super().__init__()
        if latent_spatial % patch_size:
            raise ValueError("The latent spatial size must be divisible by patch size")
        if hidden_dim % num_heads:
            raise ValueError("DiT hidden size must be divisible by the number of heads")
        self.latent_channels = latent_channels
        self.latent_spatial = latent_spatial
        self.latent_dim = latent_channels * latent_spatial**2
        self.patch_size = patch_size
        self.tokens_per_side = latent_spatial // patch_size
        token_count = self.tokens_per_side**2
        self.patch_embed = nn.Conv2d(
            latent_channels, hidden_dim, patch_size, stride=patch_size
        )
        self.position = nn.Parameter(torch.zeros(1, token_count, hidden_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.to_patches = nn.Linear(
            hidden_dim, latent_channels * patch_size**2
        )
        nn.init.zeros_(self.to_patches.weight)
        nn.init.zeros_(self.to_patches.bias)

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        patches = tokens.view(
            batch,
            self.tokens_per_side,
            self.tokens_per_side,
            self.latent_channels,
            self.patch_size,
            self.patch_size,
        )
        return (
            patches.permute(0, 3, 1, 4, 2, 5)
            .reshape(
                batch,
                self.latent_channels,
                self.latent_spatial,
                self.latent_spatial,
            )
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        spatial = noise.view(
            -1, self.latent_channels, self.latent_spatial, self.latent_spatial
        )
        tokens = self.patch_embed(spatial).flatten(2).transpose(1, 2)
        tokens = self.blocks(tokens + self.position)
        residual = self._unpatchify(self.to_patches(self.final_norm(tokens)))
        return (spatial + residual).flatten(1)
