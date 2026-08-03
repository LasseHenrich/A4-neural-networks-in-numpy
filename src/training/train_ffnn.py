"""Mini-batch SGD training loop for feed-forward networks."""

import numpy as np

from data import DataLoader, Dataset
from evaluation import MetricCollection
from nn import Module, Sequential
from optim import Optimizer


def run_epoch(
    model: Sequential,
    loader: DataLoader,
    loss_fn: Module,
    optimizer: Optimizer | None = None,
    metrics: MetricCollection | None = None,
) -> dict[str, float]:
    """
    Runs one full pass over `loader` and returns a dict of scalar
    measurements for the pass.

    The returned dict always contains `"loss"` (the sample-weighted
    average loss). If `metrics` is provided, its per-metric values are
    merged in under each metric's name.

    When `optimizer` is provided the pass is a training pass: each
    batch zeroes the gradient buffers, runs the backward pass, and
    applies one optimizer step. When `optimizer` is None the pass is
    an evaluation pass: only the forward pass runs.

    The model is switched into the matching mode at the top of the
    pass via `model.train()` / `model.eval()`, mirroring PyTorch.

    `metrics` is reset at the start of each pass so the same object
    can be reused across training and validation passes and across
    epochs.
    """
    if optimizer is None:
        model.eval()
    else:
        model.train()
    
    if metrics is not None:
        metrics.reset()
    
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
            
    result = { "loss": total_loss / n }
    if metrics is not None:
        result |= metrics.compute()
    return result


def train_ffnn(
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
    """
    Trains `model` with mini-batch SGD and returns per-epoch
    measurements for training and validation.

    Wraps `train_dataset` in a shuffled DataLoader and `val_dataset`
    in an ordered DataLoader, then for each epoch:
      1. Calls `run_epoch` on the training loader (with optimizer) to
         update the model and collect training measurements.
      2. Calls `run_epoch` on the validation loader (no optimizer) to
         collect validation measurements.
      3. Merges both result dicts into `history` with `"train_"` and
         `"val_"` prefixes, creating keys on first encounter.

    The same `metrics` object is passed to both passes; `run_epoch`
    resets it at the start of each pass so there is no
    cross-contamination.

    Arguments:
        model          -- a Sequential
        train_dataset  -- Dataset yielding (input, target) pairs for
                          training; wrapped in a shuffled DataLoader
                          internally
        val_dataset    -- Dataset yielding (input, target) pairs for
                          validation; wrapped in an ordered DataLoader
                          internally
        optimizer      -- a pre-instantiated Optimizer already
                          tracking the model's parameters
        epochs         -- number of full passes over the training set
        batch_size     -- mini-batch size
        loss_fn        -- loss module used for forward and backward
                          passes
        metrics        -- optional MetricCollection; when None the
                          history contains only "train_loss" and
                          "val_loss"
        seed           -- seed for the training DataLoader's shuffle RNG

    Returns:
        dict mapping metric names (with "train_" / "val_" prefix) to
        lists of per-epoch values. "train_loss" and "val_loss" are
        always present; additional keys come from each metric's name.
    """
    history: dict = {}
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, seed=seed)
    
    for _ in range(epochs):
        train_result = run_epoch(model=model, loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics)
        val_result = run_epoch(model=model, loader=val_loader, loss_fn=loss_fn, metrics=metrics)

        for key, value in train_result.items():
            history.setdefault(f"train_{key}", []).append(value)
        for key, value in val_result.items():
            history.setdefault(f"val_{key}", []).append(value)
            
    return history
