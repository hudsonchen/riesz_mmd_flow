"""Riesz-kernel MMD diagnostics and particle-flow dynamics."""

from __future__ import annotations

import torch


def _validate_samples(particles: torch.Tensor, targets: torch.Tensor) -> None:
    if particles.ndim != 2 or targets.ndim != 2:
        raise ValueError("particles and targets must be two-dimensional tensors")
    if particles.shape[1] != targets.shape[1]:
        raise ValueError("particles and targets must have the same feature dimension")
    if particles.shape[0] == 0 or targets.shape[0] == 0:
        raise ValueError("particles and targets must be non-empty")


@torch.no_grad()
def riesz_mmd_squared(particles: torch.Tensor, targets: torch.Tensor) -> float:
    """Return biased MMD squared for the Riesz kernel ``k(x,y)=-||x-y||``.

    This is the multivariate energy distance. The biased empirical estimator is
    used so that the diagnostic is nonnegative and matches the self-interaction
    convention used by the particle velocity.
    """
    _validate_samples(particles, targets)
    cross_distance = torch.cdist(particles, targets).mean()
    particle_distance = torch.cdist(particles, particles).mean()
    target_distance = torch.cdist(targets, targets).mean()
    energy = 2 * cross_distance - particle_distance - target_distance
    return float(energy.clamp_min(0))


@torch.no_grad()
def riesz_velocity(
    particles: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Return the exact negative Riesz-MMD particle gradient.

    For ``k(x,y)=-||x-y||``, each particle is attracted toward every target
    and repelled from every other generated particle. Pairwise distances are
    clamped only to define the derivative at coincident points.
    """
    _validate_samples(particles, targets)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    particle_target_distance = torch.cdist(particles, targets)
    target_inverse_distance = particle_target_distance.clamp_min(epsilon).reciprocal()
    target_inverse_distance.masked_fill_(particle_target_distance == 0, 0)
    target_gradient = (
        particles * target_inverse_distance.sum(dim=1, keepdim=True)
        - target_inverse_distance @ targets
    ) / targets.shape[0]

    particle_distance = torch.cdist(particles, particles)
    particle_inverse_distance = particle_distance.clamp_min(epsilon).reciprocal()
    particle_inverse_distance.masked_fill_(particle_distance == 0, 0)
    self_gradient = (
        particles * particle_inverse_distance.sum(dim=1, keepdim=True)
        - particle_inverse_distance @ particles
    ) / particles.shape[0]

    return self_gradient - target_gradient
