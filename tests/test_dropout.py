"""Tests for the Dropout layer."""

import numpy as np
import pytest

from nn.dropout import Dropout


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_dropout_forward_smoke_train():
    pass


def test_dropout_forward_smoke_eval():
    pass


# ---------------------------------------------------------------------------
# Eval mode: identity behaviour
# ---------------------------------------------------------------------------


def test_dropout_eval_forward_is_identity():
    pass


def test_dropout_eval_backward_is_identity():
    pass


# ---------------------------------------------------------------------------
# Train mode: mask statistics and scaling
# ---------------------------------------------------------------------------


def test_dropout_train_zeros_fraction():
    pass


def test_dropout_train_survivors_scaled():
    pass


def test_dropout_train_mask_reused_in_backward():
    pass


def test_dropout_train_backward_gradient_values():
    pass


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_dropout_same_seed_same_mask():
    pass


def test_dropout_different_seeds_different_masks():
    pass


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------


def test_dropout_invalid_p_negative():
    pass


def test_dropout_invalid_p_one():
    pass


def test_dropout_invalid_p_above_one():
    pass


def test_dropout_p_zero_valid():
    pass
