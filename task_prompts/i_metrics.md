# Task I — Evaluation Metrics

Training loss tells us whether optimization is making progress, but it does not say how often the model is correct or where it makes mistakes. This task introduces two complementary ways to measure performance: **accuracy**, which counts the fraction of correct predictions, and the **confusion matrix**, which breaks errors down by class so you can see exactly which classes are confused with which.

To support multiple metrics without coupling them to the training loop, we use a small framework built around a common interface. You will implement the `accuracy` helper function and several methods inside `confusion_matrix.py` that fill in this framework.

---

## The Streaming `Metric` Framework

Because data flows through the network in mini-batches, a metric cannot wait until all predictions are collected. Instead, each metric accumulates running statistics over a pass through the data:

- `update(outputs, targets)` — folds one mini-batch into the metric's running state.
- `compute()` — finalizes and returns the metric value after all batches have been seen.
- `reset()` — clears all accumulated state before starting a new pass.

This pattern lets the training loop call `update` once per batch and `compute` once at the end, without holding all predictions in memory at the same time.

The `Metric` abstract base class (provided in `src/evaluation/metric.py`) enforces this interface using `ABC` and `@abstractmethod`, which you saw in task h:

```python
class Metric(ABC):
    name: str

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update(self, outputs: np.ndarray, targets: np.ndarray) -> None: ...

    @abstractmethod
    def compute(self) -> float | np.ndarray: ...
```

Any subclass that does not implement all three methods cannot be instantiated. `name` identifies the metric in results dictionaries.

`MetricCollection` (also provided) groups a list of `Metric` objects and drives them together. Its `update`, `compute`, and `reset` methods simply delegate to each metric in the list:

```python
class MetricCollection:
    def update(self, outputs, targets):
        for metric in self.metrics:
            metric.update(outputs, targets)

    def compute(self):
        return {metric.name: metric.compute() for metric in self.metrics}
```

Adding a new metric is just appending another object to the list — the training loop does not change.

Note that `update(outputs, targets)` is the *performance-metric* contract used by `Accuracy` and `ConfusionMatrix`; the diagnostic metrics introduced in later tasks (gradient norms, weight norms, etc.) intentionally ignore those arguments and instead read from the model they were given at construction time.

---

## Accuracy

Accuracy is the fraction of predictions that match the true labels. It was introduced in assignment 2; a full re-derivation is not needed here. The key property is that computing it for a single batch is identical to computing it for a full dataset: count the matches and divide by the total.

The provided `Accuracy` class (in `src/evaluation/accuracy.py`) wraps the bare `accuracy` helper into the streaming interface. On each `update` call it takes the argmax of the model's output logits to get predicted class indices, then accumulates a weighted sum of per-batch accuracy values. At `compute` time it divides to recover the overall fraction:

```python
class Accuracy(Metric):
    name = "acc"

    def update(self, outputs: np.ndarray, targets: np.ndarray) -> None:
        batch_n = len(targets)
        preds = outputs.argmax(axis=1)
        self._total += accuracy(preds, targets) * batch_n
        self._n += batch_n

    def compute(self) -> float:
        return self._total / self._n
```

You will implement the `accuracy` helper that `Accuracy.update` calls.

---

## The Confusion Matrix

For a K-class problem, the **confusion matrix** is a K×K table. Entry `[i, j]` counts the number of samples whose true class is `i` and whose predicted class is `j`.

```
                   Predicted
              Cat   Dog   Fish
           ┌──────────────────
True  Cat  │  50     3     2
      Dog  │   4    44     1
      Fish │   1     2    47
```

The diagonal entries are correct predictions. Off-diagonal entries are mistakes: entry `[0, 1] = 3` means 3 cats were predicted as dogs.

You build the confusion matrix over a mini-batch by taking the argmax of the output logits to get predicted indices, then incrementing the matrix at position `(true_class, predicted_class)` for each sample. `np.add.at` performs this scatter-add operation without buffering:

```python
preds = outputs.argmax(axis=1)
np.add.at(matrix, (targets, preds), 1)
```

`np.add.at(matrix, (targets, preds), 1)` increments `matrix[targets[i], preds[i]]` by 1 for each index `i`. Unlike `matrix[targets, preds] += 1`, it handles the case where the same `(i, j)` position appears more than once in a single batch.

### Per-Class Precision

For class `k`, precision is the fraction of samples **predicted** as `k` that are actually class `k`:

```
precision[k] = matrix[k, k] / sum(matrix[:, k])
```

The denominator is the sum of column `k` — everything predicted as `k`. When the denominator is zero (the model never predicted class `k`), precision is defined as `0.0`.

In NumPy, you can compute all K precisions at once and guard the division with `np.divide(..., where=col_sums > 0)`, which leaves positions zero where the denominator is zero:

```python
out = np.zeros_like(diag)
np.divide(diag, col_sums, out=out, where=col_sums > 0)
```

### Per-Class Recall

For class `k`, recall is the fraction of **actual** class-`k` samples that the model correctly predicted as `k`:

```
recall[k] = matrix[k, k] / sum(matrix[k, :])
```

The denominator is the sum of row `k` — everything that truly belongs to class `k`. Apply the same divide-by-zero guard, returning `0.0` when the row sum is zero.

### F1 Score

F1 is the harmonic mean of precision and recall. It gives equal weight to both, and is lower than the arithmetic mean — a model with high precision but near-zero recall will have a low F1:

```
F1[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
```

When `precision[k] + recall[k] = 0`, define `F1[k] = 0.0`.

---

## What to Implement

### `src/evaluation/accuracy.py` — `accuracy`

```python
def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
```

- `y_pred` — `(N,)` array of integer predicted class indices.
- `y_true` — `(N,)` array of integer true class indices.
- Raise `ValueError` if `y_pred.shape != y_true.shape`.
- Raise `ValueError` if either array is empty (`y_pred.size == 0`).
- Return the fraction of positions where `y_pred[i] == y_true[i]`, as a Python `float`.

The `Accuracy` class above this function is provided and does not need to be modified.

### `src/evaluation/confusion_matrix.py`

#### `ConfusionMatrix.update`

```python
def update(self, outputs: np.ndarray, targets: np.ndarray) -> None:
```

- `outputs` — logits of shape `(N, K)`; take `argmax` over axis 1 to get predictions.
- `targets` — integer class labels of shape `(N,)`.
- Increment `self._matrix` at each `(targets[i], preds[i])` position using `np.add.at`.

#### `_per_class_precision`

```python
def _per_class_precision(cm: np.ndarray) -> np.ndarray:
```

- Returns a `(K,)` float array of per-class precision values.
- Use the diagonal and column sums of `cm`.
- Return `0.0` for any class whose column sum is zero.

#### `_per_class_recall`

```python
def _per_class_recall(cm: np.ndarray) -> np.ndarray:
```

- Returns a `(K,)` float array of per-class recall values.
- Use the diagonal and row sums of `cm`.
- Return `0.0` for any class whose row sum is zero.

#### `f1`

```python
def f1(cm: np.ndarray, class_label: int | None = None) -> float | np.ndarray:
```

- Call `_per_class_precision` and `_per_class_recall`.
- Compute the harmonic mean using the formula above, guarding against `P + R == 0`.
- If `class_label` is not `None`, return `float(per_class[class_label])`; otherwise return the full `(K,)` array.

The `__init__`, `reset`, `compute` methods of `ConfusionMatrix` and the `accuracy_from_matrix(cm)`, `precision`, `recall` wrapper functions are provided and do not need to be modified.

---

## Deliverables

- Implement `accuracy` in `src/evaluation/accuracy.py`.
- Implement `ConfusionMatrix.update`, `_per_class_precision`, `_per_class_recall`, and `f1` in `src/evaluation/confusion_matrix.py`.
- Run `make test-i` and confirm the smoke tests pass.
- Run `make submit-i` to generate `submission.json` and upload it on the course webpage.
