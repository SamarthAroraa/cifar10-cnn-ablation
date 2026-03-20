# CIFAR-10 CNN Ablation Study

A systematic hyperparameter ablation study on a small CNN trained on CIFAR-10 (50k images, 10 classes, 32x32 px). Each experiment batch fixes all variables except one, then measures val/loss and val/acc after 30 epochs.

## Experiments

| # | Variable | Values Tested | Winner |
|---|----------|---------------|--------|
| 1 | Scheduler | none, step, cosine, plateau, warmup_cosine | **step** |
| 2 | Filter width | [16,32,64,128], [32,64,128,256], [64,128,256,512] | **[32,64,128,256]** |
| 3 | Depth | 1, 2, 3, 4 blocks | **3** |
| 4 | Kernel size | 3, 5, 7 | **3** |
| 5 | Dropout | 0.1, 0.3, 0.5, 0.6, 0.7 | **0.5-0.6** |

## Best Configuration

```python
filters    = [64, 128, 256]
depth      = 3
kernel     = 3
scheduler  = "step" (StepLR, step=10, gamma=0.1)
optimizer  = "adam"
lr         = 3e-4
dropout    = 0.5-0.6
epochs     = 30
batch_size = 64
```

**Best val/acc: ~83%**

## Key Findings

- **Scheduler**: At 30 epochs, discrete LR drops (StepLR) outperform smooth decay. Cosine needs 60+ epochs.
- **Filter width**: Diminishing returns above 256 filters. Bottleneck shifts to data augmentation/regularization.
- **Depth**: depth=3 is the sweet spot for 32x32 images. depth=4 causes resolution-depth mismatch (2x2 final spatial).
- **Kernel size**: Minimal impact at 32x32 resolution. kernel=3 wins on parameter efficiency.
- **Dropout**: Follows inverted-U curve. Standard 0.3 is suboptimal for larger models on small datasets.

## Tracking

Experiments tracked with [Weights & Biases](https://wandb.ai) under project `cifar10-cnn-study`.

## Setup

```bash
pip install torch torchvision wandb matplotlib seaborn opencv-python
```

Download CIFAR-10 automatically on first run via `torchvision.datasets.CIFAR10(download=True)`.
