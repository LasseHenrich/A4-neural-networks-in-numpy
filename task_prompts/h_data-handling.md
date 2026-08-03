# Task H — Data Handling: Dataset and DataLoader

So far the training loop has operated directly on raw NumPy arrays passed in as function arguments. This works, but it ties the training code to the specific format of the data. It is useful instead to define a **dataset abstraction**: an object that knows how to retrieve individual samples, regardless of whether those samples come from an in-memory array, a file on disk, or a database. The training loop can then ask a **data loader** to produce mini-batches from the dataset; the loader handles shuffling and batching.

This separation mirrors the design PyTorch uses with `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`. In this task you will implement lightweight NumPy equivalents.

---

## The `Dataset` Protocol

A dataset is any object that supports two operations:

- `len(dataset)` — returns the total number of samples.
- `dataset[i]` — returns the `i`-th sample as a `(x, y)` pair.

In Python, `len(obj)` calls `obj.__len__()` and `obj[i]` calls `obj.__getitem__(i)`. By defining these two special methods you make your class work with Python's built-in `len` and indexing syntax, the same way a list or tuple does.

`Dataset` is an **abstract base class** — it defines the interface but leaves the implementation to subclasses. A subclass that forgets to implement `__len__` or `__getitem__` should raise a `TypeError` immediately when instantiated, rather than failing silently at runtime. Python's `abc` module provides this guarantee:

```python
from abc import ABC, abstractmethod

class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]: ...
```

Marking a method `@abstractmethod` means that any direct subclass that does not override it cannot be instantiated. `Dataset` itself cannot be instantiated either — it is purely an interface specification.

---

## `ArrayDataset`

`ArrayDataset` is the concrete `Dataset` for the common case where the whole dataset fits in memory as two NumPy arrays: `X` (features, shape `(n, ...)`) and `y` (labels, shape `(n,)`). Retrieving sample `i` is just `X[i]` and `y[i]`.

```
dataset = ArrayDataset(X_train, y_train)
x0, y0 = dataset[0]   # first sample
print(len(dataset))    # 50000 (for MNIST training set)
```

---

## The `DataLoader`

A `DataLoader` wraps a `Dataset` and, each time you iterate over it, produces a sequence of mini-batches. It handles two bookkeeping concerns the training loop should not deal with directly:

**Shuffling.** At the start of each new iteration the loader optionally shuffles the sample indices so successive epochs see the data in a different order. Shuffling is controlled by a `shuffle` flag and a `seed` for reproducibility.

**Batching.** The loader groups the (possibly reordered) indices into contiguous blocks of `batch_size` and gathers the corresponding rows from the dataset. The final batch may be smaller than `batch_size` if the dataset length is not evenly divisible by `batch_size`.

The loader exposes the standard Python iteration protocol through `__iter__`, so callers write:

```python
loader = DataLoader(dataset, batch_size=128, shuffle=True, seed=0)
for X_batch, y_batch in loader:
    # X_batch shape: (128, 784) — or smaller for the last batch
    # y_batch shape: (128,)
    ...
```

Internally `__iter__` works as follows:

```
indices = [0, 1, 2, ..., len(dataset) - 1]
if shuffle:
    rng.shuffle(indices)
for each block of batch_size indices:
    gather the corresponding samples from the dataset
    yield (X_batch, y_batch)
```

To gather a batch, iterate over the block of indices and call `dataset[i]` for each `i`, then assemble the results:

```python
xs, ys = zip(*(dataset[i] for i in batch_idx))
X_batch = np.stack(xs)
y_batch = np.array(ys)
```

This approach works for any `Dataset` subclass.

---

## What to Implement

### `src/data/dataset.py`

#### `ArrayDataset(Dataset)`

Concrete dataset backed by two NumPy arrays.

- `__init__(self, X: np.ndarray, y: np.ndarray) -> None`
  - Store `X` and `y` on `self`.
  - Raise `ValueError` if `len(X) != len(y)`.
- `__len__(self) -> int` — return `len(self.X)`.
- `__getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]` — return `(self.X[idx], self.y[idx])`.

### `src/data/dataloader.py`

#### `DataLoader`

- `__len__(self) -> int` — return the number of batches: `ceil(len(self.dataset) / self.batch_size)`. You may use `math.ceil` or the equivalent integer expression `(n + batch_size - 1) // batch_size`.
- `__iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]`
  - Build an index array `np.arange(len(self.dataset))`.
  - If `shuffle` is `True`, shuffle it in place with `self.rng.shuffle(indices)`.
  - Walk through `indices` in non-overlapping blocks of `self.batch_size`. For each block, gather the corresponding samples from `self.dataset` and yield `(X_batch, y_batch)`.

---

## Deliverables

- Implement `ArrayDataset` in `src/data/dataset.py`.
- Implement `__len__` and `__iter__` methods of`DataLoader` in `src/data/dataloader.py`.
- Run `make test-h` and confirm the smoke tests pass.
- Run `make submit-h` to generate `submission.json` and upload it on the course webpage.
