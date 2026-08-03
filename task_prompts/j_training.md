# Task J — Training: Mini-Batch Stochastic Gradient Descent

Training a network involves taking a randomly initialized model and fitting it to a dataset using an iterative optimization process. Each iteration:

1. Samples a mini-batch from the training set.
2. Measures the difference between `model(X_batch)` and `y_batch` using a loss function (e.g. `CrossEntropy`).
3. Computes how much to change each parameter (`backward`) and applies the update with an optimizer.
4. Optionally folds the batch into any running evaluation metrics.

The update step is delegated to an `Optimizer` instance (`src/optim/optimizer.py`, with `SGD` in `src/optim/sgd.py` and `Adam` in `src/optim/adam.py`, all provided). The optimizer is constructed from `model.parameters()` and exposes two methods used in the training loop:

- `optimizer.zero_grad()` — clears every gradient buffer to zero.
- `optimizer.step()` — applies one update to every parameter using the current contents of its gradient buffer.

Splitting the optimizer out of the model keeps the two concerns separate: the model owns the layers and their parameters, the optimizer owns the update rule and the learning rate.

> **Note on `zero_grad`.** `Linear.backward` accumulates into its gradient buffers (`dW`, `db`) rather than overwriting them. Calling `zero_grad` before each forward–backward pass clears those buffers so that the update is based only on the current batch's gradients, not the sum of all previous batches'. This matches the standard PyTorch pattern exactly — PyTorch accumulates gradients for the same reason.

You will implement this in two functions: `run_epoch`, which runs a single pass over a dataset and returns a dict of per-pass measurements, and `train_ffnn`, which calls `run_epoch` on both the training and validation splits each epoch and collects the results.


## `run_epoch`

```
run_epoch(
    model: Sequential,
    loader: DataLoader,
    loss_fn: Module,
    optimizer: Optimizer | None = None,
    metrics: MetricCollection | None = None,
) -> dict[str, float]:
```

- **Inputs:**
  - `model` -- a `Sequential` instance.
  - `loader` -- a `DataLoader` that yields `(xb, yb)` mini-batches. A shuffled loader is used for training; an ordered loader is used for evaluation.
  - `loss_fn` -- a `Module` instance used to compute the loss and its gradient.
  - `optimizer` -- an `Optimizer` instance bound to `model.parameters()`. When provided, a backward pass and parameter update are performed after each batch (training mode). When `None`, only the forward pass runs (evaluation mode).
  - `metrics` -- an optional `MetricCollection`. When provided, every batch is folded into it; when `None`, no metrics are tracked.
- **Returns:** a dict containing `"loss"` (the sample-weighted average loss over the full pass) and one entry per metric in `metrics`, keyed by the metric's `name`.

`run_epoch` proceeds in this order:

1. Switch the model into the matching mode: `model.train()` when `optimizer is not None`, otherwise `model.eval()`. This mirrors PyTorch and lets layers like `Dropout` behave correctly on each pass.
2. If `metrics` is not `None`, call `metrics.reset()` so the same object can be reused across training and validation passes and across epochs.
3. Iterate over `loader`. For each mini-batch `(xb, yb)`:

```
total_loss = 0.0
n = 0
for xb, yb in loader:
    outputs = model(xb)
    batch_loss = float(loss_fn(outputs, yb))
    total_loss += batch_loss * len(xb)        # sample-weighted accumulation
    n += len(xb)
    if optimizer is not None:
        optimizer.zero_grad()
        doutputs = loss_fn.backward()
        model.backward(doutputs)
        optimizer.step()
    if metrics is not None:
        metrics.update(outputs, yb)
```

4. Build the result dict starting with `{"loss": total_loss / n}` and merge in `metrics.compute()` when `metrics` is provided.

Weighting `batch_loss` by `len(xb)` ensures the reported loss is a true sample-weighted average, since the final batch in an epoch may be smaller than the others.


## `train_ffnn`

```
train_ffnn(
    model: Sequential,
    train_dataset: Dataset,
    val_dataset: Dataset,
    optimizer: Optimizer,
    epochs: int,
    batch_size: int,
    loss_fn: Module,
    metrics: MetricCollection | None = None,
    seed: int = 0,
) -> dict[str, list[float]]:
```

- **Inputs:**
  - `model` -- a pre-built `Sequential` (typically an `MLP`) constructed by the caller.
  - `train_dataset` -- `Dataset` of training samples; wrapped in a shuffled `DataLoader` internally.
  - `val_dataset` -- `Dataset` of validation samples; wrapped in an ordered `DataLoader` internally.
  - `optimizer` -- a pre-instantiated `Optimizer` already tracking the model's parameters.
  - `epochs` -- number of full passes over the training set.
  - `batch_size` -- mini-batch size used by both loaders.
  - `loss_fn` -- a `Module` instance used to compute the loss and its gradient.
  - `metrics` -- optional `MetricCollection` reused across both passes; when `None`, the history contains only `"train_loss"` and `"val_loss"`.
  - `seed` -- seed for the training `DataLoader`'s shuffle RNG so successive runs are reproducible.
- **Returns:** a dict mapping `"train_"`/`"val_"`-prefixed measurement names to lists of length `epochs`. `"train_loss"` and `"val_loss"` are always present; additional keys come from each metric's `name` field.

`train_ffnn` should:

1. Build a shuffled `DataLoader` for `train_dataset` (using `seed` for the shuffle RNG) and an ordered `DataLoader` for `val_dataset`.
2. For each epoch:
   1. Call `run_epoch` on the training loader, passing the optimizer and the metrics collection, to update the model and collect training measurements.
   2. Call `run_epoch` on the validation loader without an optimizer (but with the same metrics collection) to collect validation measurements. `run_epoch` resets `metrics` at the top of each pass, so there is no cross-contamination between training and validation.
   3. For each key in the training result, append the value to `history["train_<key>"]`, creating the list on first encounter. Do the same for the validation result under `"val_<key>"`.

The caller is responsible for constructing `model`, `optimizer`, `loss_fn`, and `metrics` — `train_ffnn` does not know which optimizer or which loss is in use.

You do not need to modify `run_lr_experiment` — it is already provided in `src/experiments/lr_sweep.py`. It builds an `MLP`, an `SGD`, a `CrossEntropy`, and a `MetricCollection([Accuracy()])` (via the `get_network` helper in `src/training/get_network.py`), then calls `train_ffnn` for each learning rate.


## Deliverables

- Implement `run_epoch` and `train_ffnn` in `src/training/train_ffnn.py`.
- Run `make test-j` and confirm the smoke tests pass.
- Run `make demo-task-j` to produce `results/figures/lr_experiment.png`. The script prints the final validation and test accuracy for each learning rate; confirm they look like real learning curves (rising validation accuracy, reasonable final test accuracy on MNIST) before recording your answers in `answers.py`.
- Run `make submit-j` to generate `submission.json` and upload it on the course webpage.
