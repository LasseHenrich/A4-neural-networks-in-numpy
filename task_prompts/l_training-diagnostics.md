# Task L — Training Diagnostics: Watching the Gradient and Weight Signals

This is the second of six observe-only tasks. You do not implement anything here. You run a provided demo, study the figure it produces, and record what you saw in `answers.py`.


## Background

The training loss tells you whether your model is learning at the bottom line, but it does not tell you *why* or *how*. Two runs with identical loss curves can be in completely different states underneath — one is learning efficiently, the other has barely started moving but happens to land on the same loss because the network is symmetric near initialization. The diagnostic signals — gradient norms, weight norms, the ratio between them, and the effective update size — let you see inside the training loop.

### Four scalar signals

**Gradient norm.** At the end of each batch the backward pass computes a gradient for every parameter. The *gradient norm* is the L2 norm of all those gradients gathered into one number. A gradient norm that is stable and reasonable means the backward signal is reaching the parameters. A gradient norm that grows without bound is gradient *explosion*: parameters will be updated by huge steps and training diverges. A gradient norm that shrinks to near zero is gradient *vanishing*: parameters stop receiving a useful signal and training stalls. One subtlety to keep in mind: if the loss is not going down, the gradient norm does not go down either — it stays pinned near its initial-loss magnitude. A stubbornly high gradient norm can therefore mean either "exploding" or "not learning at all".

**Weight norm.** The *weight norm* is the L2 norm of all parameters (the actual stored weights, not the gradients). It gives you the scale of the model. A model that has barely moved from its initialization has a weight norm close to the He-init baseline. A model that has learned typically has a larger weight norm; a model whose weights are exploding has a much larger one. A weight norm that does not change at all over epochs is a clear sign that no useful updates are happening.

**Gradient-to-weight ratio.** Comparing gradient norm and weight norm in absolute terms is hard because both depend on the architecture size. The *gradient-to-weight ratio* — gradient norm divided by weight norm — normalises the architecture away: it measures how large the gradient signal is *relative to the current magnitude of the parameters*. A ratio in the 10⁻² range is typical for healthy training. A ratio that is constant across epochs while the loss is not dropping is a tell that nothing useful is happening even though the numbers look "reasonable". Watch this signal alongside the loss curve, not on its own.

**Update magnitude.** The *update magnitude* is `lr × gradient_norm / weight_norm`. It combines the gradient-to-weight ratio with the learning rate to give the *fractional change* applied to the weights at each step: an update magnitude of 10⁻³ means each step moves the parameters by roughly 0.1% of their current scale. This is the single most actionable diagnostic of the four.

### Rule of thumb: aim for an update magnitude near 10⁻³

A well-known heuristic from Karpathy's CS231n course notes recommends that the update-to-weight ratio — exactly the update magnitude defined above — should land somewhere around **10⁻³**. This is a *rough* heuristic, not a hard target: anywhere in the 10⁻³ to 10⁻² range is typical for a well-tuned network. (You will see in the demo below that the healthy run lands comfortably in this band on both datasets — near ~10⁻² early in training and settling toward ~10⁻³ as it converges — and trains smoothly throughout.) What you really want to detect is *order-of-magnitude* deviations — an update magnitude near 1.0 means each step is overwriting the weights and training will diverge; an update magnitude near 10⁻⁶ means the parameters are not moving and the model is stalled. This one number folds learning-rate appropriateness, initialisation scale, and gradient flow into one signal.

Two refinements come up in practice:

- **Track per layer, not just globally.** A healthy global update magnitude can hide a layer whose update magnitude is 10⁻⁸ — the global average is dominated by the layers that *are* learning. Real diagnostic pipelines compute the ratio for each parameter tensor separately so layer-specific problems (especially vanishing gradients in deep stacks) show up. The diagnostic metrics in `src/evaluation/` collapse all parameters into one aggregate to keep this teaching plot readable; extending them to per-layer is a straightforward change to the metric implementations.
- **Read the diagnostics with the loss curve, not instead of it.** The signals above describe *how* the optimiser is moving, but only the loss says whether the moves are productive. A few specific joint patterns:

| Loss | Update magnitude | Likely state | What to do |
|---|---|---|---|
| Falling smoothly | 10⁻³ to 10⁻², steady | Healthy training | Nothing |
| Flat or NaN | ≫ 10⁻², gradient norm spiking | Step too large | Lower lr or clip gradients |
| Flat | ≪ 10⁻³, weight norm not moving | Step too small | Raise lr, fix init, check that gradients reach the layer |
| Falling globally, but layerwise grad norms decay toward zero in deep layers | Global update magnitude looks fine | Vanishing gradients hidden by the aggregate | Better init, residuals, or a per-layer view |
| Falling, but weight norm growing without bound | In range early, drifts up | Under-regularised | Weight decay or dropout |

### The histogram metric

A fifth diagnostic, `GradientHistogram`, collects all gradient values into a histogram rather than collapsing them to a single norm. It complements the scalar signals by showing the *distribution shape* — symmetric, skewed, or piled near zero (potential dying ReLU units). This task focuses on the four scalar signals, which are easier to display as epoch-by-epoch trajectories.

The implementations of all five diagnostic metrics live in `src/evaluation/`.


## Run the Experiment

Run the demo:

```
make demo-task-l
```

This trains four configurations on two datasets and writes the figure:

```
results/figures/training_diagnostics.png
```

The figure is a grid. Each **column** is one dataset:

1. **toy (moons)** — a 2-D binary classification dataset.
2. **MNIST subset** — 2000 training samples and 500 validation samples from MNIST.

Each **row** is one signal:

1. Training loss (log scale)
2. Training accuracy (linear scale)
3. Gradient norm (log scale)
4. Weight norm (log scale)
5. Grad / weight ratio (log scale)
6. Update magnitude (log scale)

Within each panel, four curves are drawn — one per condition:

| Colour | Condition | Description |
|---|---|---|
| Blue | `healthy` | lr = 0.05, single hidden layer of 64 units, default He init |
| Red | `lr_too_high` | lr = 10.0 on toy moons, lr = 1.5 on the MNIST subset, same architecture |
| Orange | `lr_too_low` | lr = 1e-5, same architecture |
| Purple | `deep_bad_init` | lr = 0.05, four hidden layers of 64 units, weights scaled to 0.1× He |

A note on the two `lr_too_high` rates. "Too high" is not an absolute number of the learning rate alone — it depends on the *scale* of the inputs and how many of them feed each neuron. The MNIST inputs are 784 pixels rescaled to `[0, 1]`, so a single neuron sums far more (and larger) terms than on the 2-D toy data, which makes the effective step much larger for the same `lr`. lr = 10 is *moderately* too high on toy moons (the run survives but never settles), whereas on MNIST the same value collapses the network within one epoch — every ReLU unit dies and the gradient vanishes, so nothing is left to diagnose. Dropping the MNIST rate to lr = 1.5 keeps the run *sustainedly* unstable for all 30 epochs so the "too high" signatures stay visible. This input-scale dependence is exactly the phenomenon task n explores in depth.

What to look for, reading the rows top-to-bottom:

- **Loss and accuracy (rows 1–2).** The `healthy` condition drives the loss smoothly down and the accuracy up to around 0.90 on both datasets. The other three are all stalled-or-broken in different ways: `lr_too_high`'s loss oscillates without ever settling and its accuracy stays low and noisy, well short of `healthy`; `lr_too_low` is essentially flat at the initial loss; `deep_bad_init` barely improves and stays near chance. *None of them look identical in the diagnostics below — that is the point of monitoring them.*
- **Gradient norm (row 3).** `lr_too_high` produces the highest *peak* gradient norm — a large spike in the first epoch as the oversized steps fling the parameters around. Counter-intuitively, `lr_too_low` has one of the highest *sustained* gradient norms: because the loss never decreases, its gradient stays pinned near its initial-loss magnitude for all 30 epochs. `healthy`'s gradient norm decays as the loss drops. `deep_bad_init`'s gradient is the tiniest throughout — roughly an order of magnitude below `healthy` — vanishing through the four-layer stack of 10× under-scaled weights. This is the vanishing-gradient condition, and this row is the *only* place its vanishing is plainly visible: the ratio and update-magnitude rows below are fooled by it.
- **Weight norm (row 4).** `healthy` grows modestly from its He-init baseline (although this is difficult to see in the toy example). `lr_too_low` barely moves at all. `lr_too_high` is driven *upward* by its oversized steps — on toy moons it explodes roughly fourfold within the first two epochs and then sits at that inflated level; on the MNIST subset it climbs steadily across all 30 epochs. `deep_bad_init` starts about 5× below the others (because every W was rescaled by 0.1) and grows only slowly.
- **Grad / weight ratio (row 5).** `healthy` lands within an order of magnitude of 10⁻²: low-10⁻² on toy moons, nearer 10⁻¹ on the MNIST subset. `lr_too_low`'s ratio is the *highest* of the four — gradient and weight both stay frozen near their initial values, so the ratio is simply pinned at its (relatively large) starting level. `lr_too_high`'s ratio is *not* the highest despite its large gradient: on toy moons the weight norm has exploded by an even larger factor, so the ratio normalizes downward. The surprise is `deep_bad_init`: its ratio sits almost exactly on top of `healthy`, *not* among the lowest. Scaling every layer's W by the same 0.1 shrinks the gradient, but it shrinks the weight norm in the denominator by a comparable factor, so the rescale very nearly *cancels* in the ratio. The vanishing is real — this row just cannot see it. It shows up only in the absolute gradient norm (row 3) and in the flat loss.
- **Update magnitude (row 6, the headline number).** This row is `lr × grad/weight ratio` and is the diagnostic with the cleanest interpretation — *most* of the time. `healthy` lands in the 10⁻³ to 10⁻² band on both datasets — near ~10⁻² early on and settling toward ~10⁻³ as it converges, right around the Karpathy/CS231n target, and the loss confirms training is fine. `lr_too_low` is many orders of magnitude *below* this band (each step moves the weights by roughly 10⁻⁶ — 30 epochs make no useful progress), and it is the lowest of the four throughout. `lr_too_high` is subtler than its name suggests. It spikes to an update magnitude near **1.0** in its first epoch — each step essentially *overwriting* the weights, the unmistakable mark of too large a step — but it does not stay there: the oversized steps inflate the weight norm (row 4 — roughly four-fold on toy moons), and that swollen denominator throttles the fractional update back down. Within a few epochs it settles to the *top* of the healthy band on toy moons and just above it on MNIST (a few × 10⁻²) — only modestly above the healthy ceiling, not the order-of-magnitude blow-up the raw `lr` (10 and 1.5) would suggest. In steady state this row alone makes it look *almost* acceptable; what gives it away is the epoch-1 spike, the still-inflated weight norm, and a loss that oscillates and never settles (rows 1, 4). `deep_bad_init` is the other cautionary case: it tracks `healthy` almost exactly — same `lr`, and a grad/weight ratio that matches `healthy` for the cancellation reason just described — yet its loss never falls and its accuracy stays near chance. Both make the point the refinements above warn about: *an in-or-near-band update magnitude is necessary but not sufficient* — read this row with the loss, never instead of it. For `deep_bad_init` the giveaway is the lowest-of-all absolute gradient norm (row 3) and the flat loss and accuracy (rows 1–2); localizing the dead early layers would need the per-layer view that this aggregate deliberately collapses away.


## Deliverables

- Run `make demo-task-l` and open `results/figures/training_diagnostics.png`.
- Study each row and column, then fill in the following keys in `answers.py`:

```python
l_observations: dict[str, bool | None]
# Keys (all start as None — mark True if you observed it, False if not):
#   healthy_update_magnitude_lands_between_1e-3_and_1e-2
#   healthy_grad_weight_ratio_near_001_or_01
#   lr_too_high_weight_norm_explodes_upward
#   a_higher_lr_always_produces_a_higher_update_magnitude_in_a_useful_range
#   lr_too_low_update_magnitude_is_far_below_healthy
#   lr_too_low_gradient_norm_stays_pinned_at_initial_value
#   deep_bad_init_weight_norm_starts_much_lower_than_healthy
#   all_conditions_converge_to_the_same_weight_norm
#   deep_bad_init_aggregate_update_magnitude_tracks_healthy_despite_vanishing

l_condition_with_highest_peak_grad_norm: str
# Which condition produces the highest peak gradient norm?
# One of: "healthy" / "lr_too_high" / "lr_too_low" / "deep_bad_init"

l_condition_with_lowest_update_magnitude: str
# Which condition produces the lowest update magnitude throughout training?
# One of: "healthy" / "lr_too_high" / "lr_too_low" / "deep_bad_init"

l_condition_with_lowest_gradient_norm: str
# Which condition has the lowest gradient norm throughout training (the
# vanishing-gradient case)?
# One of: "healthy" / "lr_too_high" / "lr_too_low" / "deep_bad_init"
```

- Run `make submit-l` to generate `submission.json` and upload it on the course webpage.
