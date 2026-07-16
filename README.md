# FFHQ/CelebA one-step Riesz-MMD generator

This experiment trains an amortized neural generator for aligned face images. FFHQ
is the default, while CelebA remains available as a smaller alternative. The
default representation uses the official pretrained StyleALAE for the selected
dataset: 512-dimensional FFHQ latents at 1024x1024, or 256-dimensional CelebA
latents at 128x128. A residual MLP maps Gaussian noise to normalized ALAE
latents in one forward pass. The former Stable Diffusion VAE remains
available with `--autoencoder sd-vae`.

The generator is not trained by backpropagating through the discrepancy.
Instead, each generated latent receives a stop-gradient transport target from
the Riesz-MMD particle velocity. The Riesz kernel is
`k(x, y) = -||x-y||_2` and is evaluated directly in the full latent space; no
random slicing or Sinkhorn divergence is used.

## Code structure

- `main.py`: command-line configuration and entry point.
- `dataset.py`: FFHQ/CelebA discovery, transformations, and loading.
- `models.py`: frozen ALAE/VAE adapters, compact latent DiT, and residual MLP.
- `generator_training.py`: latent statistics, neural training, and outputs.
- `utils/mmd.py`: full-dimensional Riesz-MMD diagnostics and velocity.

## Run

```bash
conda activate mmd_flow_hudson
cd /home/zongchen/mmd_flow_hudson/mmd_flow
python main.py --dataset ffhq --local-files-only
```

The default FFHQ directory is `/home/zongchen/MMD_Wflow/data/ffhq-dataset`. Put the
extracted images in either of these layouts:

```text
/home/zongchen/MMD_Wflow/data/ffhq-dataset/images1024x1024/00000.png
/home/zongchen/MMD_Wflow/data/ffhq-dataset/images/00000.png
/home/zongchen/MMD_Wflow/data/ffhq-dataset/00000.png
```

The first ALAE run builds a latent cache under `.cache/`. FFHQ uses the paper's
60,000-image split at 1024x1024 with 512-dimensional codes; CelebA uses the
official 182,637-image training split at 128x128 with 256-dimensional codes.
Later runs load the cache directly. Control the one-time encoder batch with
`--encoding-batch-size`.

CelebA can still be selected with:

```bash
python main.py --dataset celeba --local-files-only
```

## Training behavior

Before training, the program estimates coordinate-wise ALAE latent statistics
from 4,096 images and saves `vae_reconstructions.png`. The autoencoder remains
frozen.
For every image batch, the program:

1. encodes the real images with the frozen ALAE;
2. draws fresh Gaussian noise and evaluates the residual MLP generator;
3. computes full-dimensional Riesz-MMD velocities from pairwise distances;
4. forms fixed regression targets from those velocities; and
5. updates only the generator parameters.

Optimization follows the paper's FFHQ setup: AdamW with learning rate
`2e-4`, betas `(0.9, 0.95)`, zero weight decay, and gradient clipping at 5.0.

Architecture selection defaults to `auto`, which uses the paper's four-layer
residual MLP with hidden width 1,024 for both face datasets. The optional compact
spatial DiT is available only with the SD VAE representation. Both generators
have zero-initialized output projections, so their
initial maps are the identity on normalized latent space. Override the choice
with `--generator-arch dit` or `--generator-arch mlp`.

## Outputs

FFHQ outputs are saved under `outputs/ffhq_mmd_flow/`; CelebA outputs use
`outputs/celeba_mmd_flow/`.

- `vae_reconstructions.png`: real images above their VAE reconstructions.
- `generator_initial.png`: samples before generator training.
- `generator_epoch_XXX.png`: a fixed 3-by-5 sample grid after every epoch.
- `generator_latest.pt`: resumable generator and optimizer checkpoint.
- `mmd_flow_metrics.json`: epoch-level loss and Riesz MMD.
- `mmd_curve.png`: epoch-level MMD convergence plot.

Resume training with:

```bash
python main.py --dataset ffhq --resume outputs/ffhq_mmd_flow/generator_latest.pt
```
