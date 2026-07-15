# CelebA one-step sliced Riesz-MMD generator

This experiment trains an amortized neural generator for CelebA. A frozen
Stable Diffusion VAE maps images to and from latent space. A residual MLP maps
Gaussian noise to normalized VAE latents in one forward pass.

The generator is not trained by backpropagating through the discrepancy.
Instead, each generated latent receives a stop-gradient transport target from
the sliced Riesz-MMD particle velocity. The one-dimensional Riesz gradients use
the sorting algorithm of Hertrich et al. (2024). No Sinkhorn divergence is used.

## Code structure

- `main.py`: command-line configuration and entry point.
- `dataset.py`: CelebA discovery, transformations, and loading.
- `models.py`: frozen VAE adapter and residual latent generator.
- `generator_training.py`: latent statistics, neural training, and outputs.
- `utils/mmd.py`: sliced Riesz-MMD diagnostics and sorting-based velocity.

## Run

```bash
conda activate mmd_flow_hudson
cd /home/zongchen/mmd_flow_hudson/mmd_flow
python main.py --local-files-only
```

The default CelebA directory is `/home/zongchen/MMD_Wflow/data`. The accepted
layouts include:

```text
/home/zongchen/MMD_Wflow/data/img_align_celeba/000001.jpg
/home/zongchen/MMD_Wflow/data/images/000001.jpg
/home/zongchen/MMD_Wflow/data/000001.jpg
```

## Training behavior

Before training, the program estimates coordinate-wise VAE latent statistics
from 4,096 images and saves `vae_reconstructions.png`. The VAE remains frozen.
For every CelebA batch, the program:

1. encodes the real images with the frozen VAE;
2. draws fresh Gaussian noise and evaluates the residual MLP generator;
3. computes sliced Riesz-MMD velocities by projection and sorting;
4. forms fixed regression targets from those velocities; and
5. updates only the generator parameters.

The default generator is a four-layer residual MLP with hidden width 1,024,
LayerNorm, and SiLU activations. Its output layer starts at zero, so the initial
map is the identity on normalized latent space.

## Outputs

Outputs are saved under `outputs/celeba_mmd_flow/`:

- `vae_reconstructions.png`: real images above their VAE reconstructions.
- `generator_initial.png`: samples before generator training.
- `generator_epoch_XXX.png`: a fixed 3-by-5 sample grid after every epoch.
- `generator_latest.pt`: resumable generator and optimizer checkpoint.
- `mmd_flow_metrics.json`: epoch-level loss and sliced Riesz MMD.
- `mmd_curve.png`: epoch-level MMD convergence plot.

Resume training with:

```bash
python main.py --resume outputs/celeba_mmd_flow/generator_latest.pt
```
