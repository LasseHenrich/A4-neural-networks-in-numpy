# Task M — Regression: The Same Network Without the Softmax

This is the second of five observe-only tasks. You do not implement anything here. You run a provided demo, study the figure it produces, and record what you saw in `answers.py`.


## Background

Every network you have trained so far has been a *classifier*. It takes an input, produces one score per class, passes those scores through softmax to turn them into a probability distribution, and uses CrossEntropy to measure how far that distribution is from the true class. The output is a guess at *which category* the input belongs to.

*Regression* is a different kind of task. Instead of choosing a category, the network outputs a single real-valued number, and we want that number to be as close as possible to a true target value. Predicting tomorrow's temperature, the price of a house, or the height of a curve at a given position are all regression problems: the answer is a quantity, not a label.

### What changes in the network

The surprising part is how little has to change. The network you already built becomes a regressor with three small edits:

1. **Set `num_classes=1`.** The output layer now has a single unit, so the model produces one number per input instead of one score per class.
2. **Remove the softmax.** Softmax exists only to turn class scores into probabilities, and there are no classes here. For regression the raw output of the final `Linear` layer *is* the prediction. Nothing is applied on top of it.
3. **Swap CrossEntropy for MSE.** CrossEntropy compares two probability distributions, which makes no sense for a single number. Instead we use *mean squared error* (MSE).

Everything else — the `Linear` layers, the ReLU hidden units, the backward pass, the optimizer — stays exactly the same. The same machinery that learned decision boundaries in task k also learns to fit a curve. Only the output shape and the loss change.

### Mean squared error

MSE measures how far the prediction is from the target. For a single sample with prediction `p` and target `y`, the error is the squared difference `(p - y)^2`. Squaring does two things: it makes every error positive (so errors above and below the target both count), and it punishes large misses much more heavily than small ones. The loss over a batch is the *mean* of these squared differences:

```
MSE = (1 / N) * sum_over_samples( (p_i - y_i)^2 )
```

A perfect fit has MSE 0. The larger the MSE, the further the predictions sit from the targets on average.

### The target function for this demo

The demo fits a *noisy sine curve*. The true underlying function is

```
y = sin(2 * pi * x)   for x in [0, 1]
```

which is one full smooth wave. The training data is a small, noisy sample of this curve: 25 points, each with a little Gaussian noise added, `y = sin(2 * pi * x) + epsilon`. The model never sees the clean curve. It only sees the noisy scatter and must figure out the smooth shape underneath. The test set is a dense grid of clean points along the true curve, with no noise, so we can measure how well the model recovered the real function rather than the noise.

### Capacity controls under- vs over-fitting

As in task k, *capacity* is the width and depth of the network taken together, and it controls how complex a function the model can represent. In task k more capacity meant a more intricate decision boundary. Here, capacity controls something you can see just as clearly: whether the model *under-fits*, fits well, or *over-fits*.

- **Under-fit (too little capacity).** With too few hidden units the model cannot bend enough to match the shape of the sine. It ends up approximating a near-straight line or a single-kink hinge that misses the rise and fall of the curve entirely. Both training and test error stay high.
- **Good fit (the right capacity).** With a modest network the model traces the true underlying curve closely. It follows the overall shape — rising, peaking, falling, recovering — without chasing every noisy point. Training and test error are both low and close together.
- **Over-fit (too much capacity).** With far more units than the problem needs, the model has enough flexibility to memorise the *exact* noisy training points. Instead of the smooth sine it produces a wiggly curve that bends to pass through each noisy point. Its training error is the lowest of all three configurations — it nailed the points it saw — but its test error is *higher* than the good-fit model's, because the wiggles do not match the clean underlying curve. That gap is over-fitting made visible.

### The train/test gap is the signal

The clearest sign of over-fitting is the *gap* between training error and test error. A model that genuinely learned the underlying function does about as well on the clean test grid as on the noisy training points: a small gap. A model that memorised the noise does great on training and worse on test: a large gap. Low training error alone tells you nothing — it is the comparison between the two that reveals whether the model generalised or merely memorised.


## Run the Experiment

Run the demo:

```
make demo-task-m
```

This trains three regressors of increasing capacity on the same noisy sine data and writes the figure:

```
results/figures/regression_fit.png
```

The figure has one panel per model configuration, in this order:

1. **`hidden=[1]`** — a single hidden unit. Far too little capacity (the underfit case).
2. **`hidden=[8, 8]`** — two hidden layers of 8 units each (the good-fit case).
3. **`hidden=[64, 64, 64]`** — three hidden layers of 64 units each. Far more capacity than the problem needs (the overfit case).

Each panel shows three things drawn on the same axes:

- The **noisy training points** as a scatter plot.
- The **true noiseless sine** as a dashed line.
- The **model's prediction** as a solid line.

Each panel title lists the configuration along with that model's **train MSE** and **test MSE**.

What to look for:

- **`hidden=[1]` (underfit).** The solid prediction line is a single-kink hinge — flat on one side, sloping on the other. It cannot follow the rise and fall of the dashed sine at all. Both MSE values are high.
- **`hidden=[8, 8]` (good fit).** The solid line traces the full shape of the dashed sine closely — rising to the peak, crossing zero, descending to the trough, and recovering at the right end. It passes through the middle of the noisy scatter rather than hitting every point. Train and test MSE are both low and close together.
- **`hidden=[64, 64, 64]` (overfit).** The solid line is visibly wiggly, bending out of its way to pass through individual noisy points instead of following the smooth dashed curve. Compare its two MSE numbers: the train MSE is the lowest of all three configurations, but the test MSE is higher than the good-fit model's. That gap is over-fitting made visible.


## Deliverables

- Run `make demo-task-m` and open `results/figures/regression_fit.png`.
- Study each panel, then fill in the following keys in `answers.py`:

```python
m_observations: dict[str, bool | None]
# Keys (all start as None — mark True if you observed it, False if not):
#   regression_uses_single_output_no_softmax
#   underfit_model_misses_curve_shape
#   mid_model_tracks_true_curve_best
#   more_hidden_units_always_lower_test_mse
#   overfit_model_wiggles_through_noisy_points
#   every_model_passes_through_all_noisy_points
#   overfit_has_lowest_train_but_higher_test_mse

m_best_generalizing_config: str
# Which configuration has the lowest *test* MSE?
# One of: "[1]" / "[8,8]" / "[64,64,64]"

m_lowest_train_mse_config: str
# Which configuration has the lowest *training* MSE?
# One of: "[1]" / "[8,8]" / "[64,64,64]"
```

- Run `make submit-m` to generate `submission.json` and upload it on the course webpage.
