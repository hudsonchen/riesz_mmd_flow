"""Sliced Riesz-MMD diagnostics and particle-flow dynamics."""

from __future__ import annotations

import math

import torch


def _sliced_factor(dimension: int) -> float:
    """Scaling between sliced and multivariate negative-distance kernels."""
    k = (dimension - 1) // 2
    factor = 1.0
    if (dimension - 1) % 2 == 0:
        for index in range(1, k + 1):
            factor *= 2 * index / (2 * index - 1)
    else:
        for index in range(1, k + 1):
            factor *= (2 * index + 1) / (2 * index)
        factor *= math.pi / 2
    return factor


def _random_projections(
    dimension: int,
    count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    projections = torch.randn(count, dimension, device=device, dtype=dtype)
    return projections / projections.norm(dim=1, keepdim=True).clamp_min(1e-12)


def make_random_projections(samples: torch.Tensor, count: int) -> torch.Tensor:
    """Draw unit directions for reproducible sliced-MMD diagnostics."""
    return _random_projections(
        samples.shape[1], count, device=samples.device, dtype=samples.dtype
    )


def riesz_derivative_1d(
    particles: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Compute the 1D Riesz-MMD gradient by sorting (paper Algorithms 1--2).

    Inputs have shape ``(projections, samples)``. The implementation is
    quadratic in neither sample count: sorting dominates its complexity.
    """
    projection_count, particle_count = particles.shape
    target_count = targets.shape[1]

    _, combined_order = torch.sort(torch.cat((particles, targets), dim=1), dim=1)
    target_before = (combined_order >= particle_count).to(particles.dtype).cumsum(1)
    potential = (2 * target_before - target_count) / (
        particle_count * target_count
    )
    _, inverse_order = torch.sort(combined_order, dim=1)
    gradient = potential.gather(1, inverse_order[:, :particle_count])

    _, particle_order = torch.sort(particles, dim=1)
    interaction_by_rank = (
        2 * torch.arange(
            particle_count, device=particles.device, dtype=particles.dtype
        )
        - particle_count
        + 1
    ) / particle_count**2
    interaction = torch.empty_like(gradient)
    interaction.scatter_(
        1, particle_order, interaction_by_rank.expand(projection_count, -1)
    )
    return gradient - interaction


def _mean_absolute_distance_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean pairwise absolute distance for many projections using sorting."""
    y_sorted = y.sort(dim=1).values
    prefix = torch.cat(
        (torch.zeros_like(y_sorted[:, :1]), y_sorted.cumsum(dim=1)), dim=1
    )
    ranks = torch.searchsorted(y_sorted.contiguous(), x.contiguous(), right=False)
    left_sum = prefix.gather(1, ranks)
    total = prefix[:, -1:]
    right_sum = total - left_sum
    distances = (
        ranks * x
        - left_sum
        + right_sum
        - (y.shape[1] - ranks) * x
    )
    return distances.mean(dim=1)


@torch.no_grad()
def sliced_riesz_mmd_squared(
    particles: torch.Tensor,
    targets: torch.Tensor,
    num_projections: int,
    projections: torch.Tensor | None = None,
) -> float:
    """Estimate squared Riesz MMD (energy distance) using sorted slices."""
    if projections is None:
        projections = make_random_projections(particles, num_projections)
    x = projections @ particles.T
    y = projections @ targets.T
    energy = (
        2 * _mean_absolute_distance_1d(x, y)
        - _mean_absolute_distance_1d(x, x)
        - _mean_absolute_distance_1d(y, y)
    )
    return float((_sliced_factor(particles.shape[1]) * energy.mean()).clamp_min(0))


@torch.no_grad()
def sliced_riesz_velocity(
    particles: torch.Tensor,
    targets: torch.Tensor,
    num_projections: int,
) -> torch.Tensor:
    """Return the negative sliced Riesz-MMD gradient using sorting."""
    projections = _random_projections(
        particles.shape[1],
        num_projections,
        device=particles.device,
        dtype=particles.dtype,
    )
    projected_particles = projections @ particles.T
    projected_targets = projections @ targets.T
    projected_gradient = riesz_derivative_1d(
        projected_particles, projected_targets
    )
    gradient = projected_gradient.T @ projections
    gradient *= _sliced_factor(particles.shape[1]) / num_projections
    return -particles.shape[0] * gradient
