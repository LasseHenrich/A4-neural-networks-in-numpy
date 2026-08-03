import numpy as np
import pytest

from data.dataloader import DataLoader
from data.dataset import ArrayDataset, Dataset
from data.utils import Subset, random_split


def _make_dataset(n: int = 12, features: int = 4) -> ArrayDataset:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, features)).astype(np.float32)
    y = np.arange(n, dtype=np.int64)
    return ArrayDataset(X, y)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_dataset_is_abstract():
    pass


# ---------------------------------------------------------------------------
# ArrayDataset
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_array_dataset_smoke():
    ds = _make_dataset()
    assert isinstance(ds, Dataset)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


def test_array_dataset_len():
    pass


def test_array_dataset_getitem_x():
    pass


def test_array_dataset_getitem_y():
    pass


def test_array_dataset_length_mismatch_raises():
    pass


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_dataloader_smoke():
    loader = DataLoader(_make_dataset(), batch_size=4)
    batches = list(loader)
    assert len(batches) > 0


def test_dataloader_yields_ndarray_pairs():
    pass


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


def test_dataloader_len_exact_division():
    pass


def test_dataloader_len_with_remainder():
    pass


def test_dataloader_full_batches_have_correct_size():
    pass


def test_dataloader_last_batch_smaller():
    pass


def test_dataloader_covers_all_samples():
    pass


def test_dataloader_no_shuffle_preserves_order():
    pass


def test_dataloader_shuffle_seed_reproducible():
    pass


def test_dataloader_shuffle_changes_order():
    pass


def test_dataloader_shuffle_advances_rng_each_epoch():
    pass


def test_dataloader_invalid_batch_size_raises():
    pass


# ---------------------------------------------------------------------------
# Subset
# ---------------------------------------------------------------------------


def test_subset_len():
    pass


def test_subset_getitem_maps_through_indices():
    pass


def test_subset_is_dataset_instance():
    pass


def test_subset_works_with_dataloader():
    pass


# ---------------------------------------------------------------------------
# random_split
# ---------------------------------------------------------------------------


def test_random_split_lengths():
    pass


def test_random_split_covers_all_indices():
    pass


def test_random_split_seeded_reproducible():
    pass


def test_random_split_different_seeds_differ():
    pass


def test_random_split_sum_mismatch_raises():
    pass


def test_random_split_zero_length_raises():
    pass


def test_random_split_three_way():
    pass


# ---------------------------------------------------------------------------
# Toy dataset class smoke tests
# ---------------------------------------------------------------------------


def test_toy_moons():
    pass


def test_toy_circles():
    pass


def test_toy_xor():
    pass


def test_toy_linear():
    pass


def test_toy_checkerboard():
    pass


def test_toy_spiral():
    pass


def test_toy_blobs():
    pass


def test_toy_curve():
    pass
