# Task N — Input Scaling: Why Preprocessing Matters

This is the third of five observe-only tasks. You do not implement anything here. You run a provided demo, study the figure it produces, and record what you saw in `answers.py`.


## Background

Tasks k and m worked on tiny made-up datasets: 2D points for the decision-boundary demo, and a 1D noisy sine for the regression demo. This is the first task on *real* data. The dataset is MNIST: 60 000 training images of handwritten digits, each a 28×28 grid of pixels, sorted into 10 classes (the digits 0 through 9). Each image is flattened into a vector of 784 numbers before it enters the network, and the model is the same `784 -> 128 -> 10` classifier with one ReLU that you trained in task j.

### Raw pixels are large numbers

Each pixel in an MNIST image is an integer brightness value in the range `[0, 255]`, where 0 is black and 255 is white. That range is large compared to the weight magnitudes that come out of He initialisation, which are small numbers centred on zero. The trouble starts in the very first layer.

When the first `Linear` layer computes `W @ x + b`, an input value as large as 255 produces a pre-activation that is far bigger than the values the rest of the network is set up to handle. Big pre-activations lead to big gradients during the backward pass, because the size of the input scales the size of the gradient flowing back into the weights.

Big gradients are dangerous when combined with a large learning rate. Recall the SGD update rule:

```
w  <-  w - lr * grad
```

If `grad` is large and `lr` is also large, the step `lr * grad` can be so big that it shoots straight past the minimum instead of settling into it. The loss spikes upward, or grows without bound, and training *diverges* — the model gets worse, not better.

### Scaling widens the usable learning-rate range

The practical consequence is that the safe range of learning rates depends on the scale of the inputs. With raw `[0, 255]` inputs, only very small learning rates are safe; anything larger blows up. If you shrink the inputs first, a much wider band of learning rates trains stably. This matters because it makes training far less sensitive to the exact learning rate you pick — you no longer have to guess a tiny value to avoid divergence.

### Two provided transforms

A *transform* is a small object that takes an input array and returns a modified version of it. You did not write these; they are provided in `src/transforms/`. Two of them rescale the inputs:

- **`ToArray()`** maps each pixel from `[0, 255]` to `[0, 1]` by dividing by 255. This is min-max rescaling: it shifts the smallest possible value to 0 and the largest to 1. The formula is

  ```
  (x - 0) / (255 - 0) = x / 255
  ```

  It is a purely mechanical change. The inputs get smaller, but their relative structure — which pixel is brighter than which — is untouched.

- **`Normalize(mean, std)`** standardises the input to roughly zero mean and unit standard deviation:

  ```
  (x - mean) / std
  ```

  It is applied *after* `ToArray`, using the MNIST training-set statistics `mean=0.1307` and `std=0.3081`. After both transforms, a typical pixel value sits near zero with a spread of about 1. Centring the inputs around zero helps gradient-based optimisation, because errors above and below zero then contribute symmetrically rather than all pulling in one direction.

These transforms are chained into a pipeline with `Compose`, which simply applies each transform in order:

```
Compose([ToArray(), Normalize()])
```

This mirrors the standard torchvision preprocessing pipeline `Compose([ToTensor(), Normalize(mean, std)])`, so the pattern will look familiar if you later move on to PyTorch.

### Scaling touches the inputs only

It is worth being precise about what input scaling does and does not change. It rescales the *inputs* — the pixel values fed into the first layer. It does not touch the labels, the architecture, the loss function, the optimizer, or anything else about the model. The exact same `784 -> 128 -> 10` classifier is trained in every case below. Only the numbers going in are different.


## Run the Experiment

Run the demo:

```
make demo-task-n
```

This trains the same MNIST classifier under three different input-scaling pipelines, sweeping several learning rates for each, and writes the figure:

```
results/figures/input_scaling.png
```

The figure has **three rows and three columns**. Each column is one scaling pipeline, in this order:

1. **Raw** — the unscaled `[0, 255]` pixel values, no transform.
2. **`ToArray()`** — pixels rescaled to `[0, 1]`.
3. **`Compose([ToArray(), Normalize()])`** — rescaled and then standardised to zero mean, unit std.

Within each column there is one curve per learning rate in the sweep (roughly `0.001`, `0.01`, `0.1`, and `1.0`). The rows are:

- **Row 1 — Val accuracy** (linear y-axis). The validation-accuracy curves you already know from task j. All three columns share the same y-range `[0, 1]`.
- **Row 2 — Train grad norm** (log y-axis). The global L2 norm of all parameter gradients, computed each batch and accumulated per epoch. Large raw inputs push the gradient norm upward; scaling brings it back down. The three columns share a log y-axis so you can compare magnitudes directly.
- **Row 3 — Train update magnitude** (log y-axis). `lr × ‖grad‖ / ‖weights‖` — the fractional change applied to the weights each step. The green band marks the healthy range `[1e-3, 1e-2]`. Runs that land inside the band tend to train stably; runs above it tend to overshoot.

### What to look for

**Raw column.** Most learning rates either diverge or sit flat near chance accuracy (about 0.1, since there are 10 classes). Only the smallest learning rate or two produce a curve that actually rises. The gradient-norm row explains why: raw inputs produce pre-activations roughly 255× larger than scaled inputs, pushing the gradient norm 1–2 orders of magnitude above the scaled columns. The update-magnitude row shows these runs sitting well above the healthy band.

**Scaled columns (`ToArray` and `Normalize`).** Many more learning rates produce rising curves, and those curves are tighter and more reliable. The gradient-norm row sits lower, and the update-magnitude row shows several learning rates landing inside the green healthy band.

**The key message.** Scaling *widens the band of learning rates that work*. That is the point — not merely that scaling helps in absolute terms, but that it makes training robust to the choice of learning rate. The gradient and update-magnitude rows make the mechanism visible: scaling controls how big the gradients are, which controls how big the weight updates are.

**One caveat.** Even after scaling, the very largest learning rate may still be unstable. Scaling shifts the stability window; it does not remove the need to pick a sensible learning rate.


## Deliverables

- Run `make demo-task-n` and open `results/figures/input_scaling.png`.
- Study each panel, then fill in the following keys in `answers.py`:

```python
n_observations: dict[str, bool | None]
# Keys (all start as None — mark True if you observed it, False if not):
#   toarray_maps_pixels_to_unit_interval
#   normalize_centers_to_zero_mean_unit_std
#   scaling_changes_inputs_not_labels
#   raw_only_trains_at_the_smallest_lr
#   raw_and_scaled_train_equally_well_at_every_lr
#   scaled_inputs_train_across_a_wider_lr_range
#   a_larger_lr_is_always_better_after_scaling
#   raw_inputs_produce_larger_gradient_norm_than_scaled
#   update_magnitude_is_identical_across_all_three_pipelines
#   scaling_keeps_update_magnitude_in_the_healthy_band_at_more_lrs

n_best_lr_for_raw: float
# What is the smallest learning rate in the sweep at which the raw
# (unscaled) inputs produce a rising validation-accuracy curve?

n_best_scaling: str
# Which scaling achieves the highest validation accuracy overall?
# One of: "raw" / "toarray" / "normalized"
```

- Run `make submit-n` to generate `submission.json` and upload it on the course webpage.
