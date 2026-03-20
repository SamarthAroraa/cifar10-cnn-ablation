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

Experiments tracked with [Weights & Biases](https://wandb.ai) under project [`cifar10-cnn-study`](https://wandb.ai/samartharora176-/cifar10-cnn-study).

## Charts

### Overview (all experiments)

![All experiments — training loss](charts/all-train-loss.png)
![All experiments — validation loss](charts/all-expt-val_loss.png)
![All experiments — validation accuracy](charts/all-val_acc.png)

### Scheduler sweep

![Scheduler sweep — training loss](charts/sched-train_loss.png)
![Scheduler sweep — validation loss](charts/sched-val_loss.png)
![Scheduler sweep — validation accuracy](charts/sched-val_acc.png)

### Filter width sweep

![Filter width sweep — training loss](charts/filter-w-train_loss.png)
![Filter width sweep — validation loss](charts/filter-w-val_loss.png)
![Filter width sweep — validation accuracy](charts/filter-w-val_acc.png)

### Depth sweep

![Depth sweep — training loss](charts/depth-train_loss.png)
![Depth sweep — validation loss](charts/depth-val_loss.png)
![Depth sweep — validation accuracy](charts/depth-val_acc.png)

### Kernel size sweep

![Kernel size sweep — training loss](charts/kernel-train_loss.png)
![Kernel size sweep — validation loss](charts/kernel-val_loss.png)
![Kernel size sweep — validation accuracy](charts/kernel-val_acc.png)

### Dropout sweep

![Dropout sweep — training loss](charts/dropout-training_loss.png)
![Dropout sweep — validation loss](charts/dropout-expts-val_loss.png)
![Dropout sweep — validation accuracy](charts/dropour-val_acc.png)

## Setup

```bash
pip install torch torchvision wandb matplotlib seaborn opencv-python
```

Download CIFAR-10 automatically on first run via `torchvision.datasets.CIFAR10(download=True)`.
