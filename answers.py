"""
Answers to observation questions in the task prompts.

Fill in each variable below with the value you observed. Replace `None`
with your answer. Types are shown in the comments.

Run `uv run python submit.py <task>` to bundle your code and these
answers into submission.json.
"""

# ---------------------------------------------------------------------------
# Task J - Training and the Learning Rate
# ---------------------------------------------------------------------------

# After running
#   make demo-task-j
# open results/figures/lr_experiment.png and answer the following.

# Which learning rate produced the highest final test accuracy?
# Type: float - one of 0.01, 0.1, 1.0
j_best_lr: float | None = 1.0

# Did lr = 0.01 fall meaningfully short of the best test accuracy
# (more than 2 percentage points below) within the 20-epoch budget?
# Type: bool
j_lr_001_underfits: bool | None = True

# Did lr = 1.0 produce a noisy or unstable validation curve compared
# to lr = 0.1? "Unstable" here means the validation accuracy drops
# from one epoch to the next at least once during training.
# Type: bool
j_lr_10_unstable: bool | None = True


# ---------------------------------------------------------------------------
# Task K - Decision Boundaries
# ---------------------------------------------------------------------------

# After running
#   make demo-task-k
# open results/figures/decision_boundaries.png and answer the following.

k_observations: dict[str, bool | None] = {
    "stacked_linear_no_activation_stays_straight": True,
    "all_configs_equal_accuracy_on_xor": False,
    "linear_model_solves_linear_but_not_xor": True,
    "relu_encloses_inner_circle": True,
    "linear_model_separates_moons": False,
    "large_relu_traces_spiral_arms": True,
    "relu_boundary_is_piecewise_linear": True,
    "small_relu_underfits_checkerboard": True,
}

# How many of the six datasets does the no-hidden-layer model solve
# (test accuracy > 0.90)?
# Type: int
k_num_datasets_linear_solves: int | None = 1

# Which is the smallest ReLU architecture that achieves accuracy > 0.75
# on the checkerboard dataset?
# Type: str — one of "[4]" / "[16]" / "[64,64]" / "none"
k_smallest_arch_solving_checkerboard: str | None = "[64,64]"


# ---------------------------------------------------------------------------
# Task L - Training Diagnostics
# ---------------------------------------------------------------------------

# After running
#   make demo-task-l
# open results/figures/training_diagnostics.png and answer the following.

l_observations: dict[str, bool | None] = {
    "lr_too_low_update_magnitude_is_far_below_healthy": None,
    "deep_bad_init_weight_norm_starts_much_lower_than_healthy": None,
    "healthy_grad_weight_ratio_near_001_or_01": None,
    "deep_bad_init_aggregate_update_magnitude_tracks_healthy_despite_vanishing": None,  # noqa: E501
    "healthy_update_magnitude_lands_between_1e-3_and_1e-2": None,
    "all_conditions_converge_to_the_same_weight_norm": None,
    "lr_too_high_weight_norm_explodes_upward": None,
    "lr_too_low_gradient_norm_stays_pinned_at_initial_value": None,
    "a_higher_lr_always_produces_a_higher_update_magnitude_in_a_useful_range": None,
}

# Which condition produces the highest peak gradient norm?
# Type: str — one of "healthy" / "lr_too_high" / "lr_too_low" /
#             "deep_bad_init"
l_condition_with_highest_peak_grad_norm: str | None = None

# Which condition produces the lowest update magnitude throughout
# training?
# Type: str — one of "healthy" / "lr_too_high" / "lr_too_low" /
#             "deep_bad_init"
l_condition_with_lowest_update_magnitude: str | None = None

# Which condition has the lowest gradient norm throughout training (the
# vanishing-gradient case)?
# Type: str — one of "healthy" / "lr_too_high" / "lr_too_low" /
#             "deep_bad_init"
l_condition_with_lowest_gradient_norm: str | None = None


# ---------------------------------------------------------------------------
# Task M - Regression
# ---------------------------------------------------------------------------

# After running
#   make demo-task-m
# open results/figures/regression_fit.png and answer the following.

m_observations: dict[str, bool | None] = {
    "more_hidden_units_always_lower_test_mse": None,
    "regression_uses_single_output_no_softmax": None,
    "every_model_passes_through_all_noisy_points": None,
    "mid_model_tracks_true_curve_best": None,
    "overfit_has_lowest_train_but_higher_test_mse": None,
    "underfit_model_misses_curve_shape": None,
    "overfit_model_wiggles_through_noisy_points": None,
}

# Which configuration has the lowest *test* MSE?
# Type: str — one of "[1]" / "[8,8]" / "[64,64,64]"
m_best_generalizing_config: str | None = None

# Which configuration has the lowest *training* MSE?
# Type: str — one of "[1]" / "[8,8]" / "[64,64,64]"
m_lowest_train_mse_config: str | None = None


# ---------------------------------------------------------------------------
# Task N - Input Scaling
# ---------------------------------------------------------------------------

# After running
#   make demo-task-n
# open results/figures/input_scaling.png and answer the following.

n_observations: dict[str, bool | None] = {
    "scaling_changes_inputs_not_labels": None,
    "raw_inputs_produce_larger_gradient_norm_than_scaled": None,
    "toarray_maps_pixels_to_unit_interval": None,
    "raw_and_scaled_train_equally_well_at_every_lr": None,
    "scaling_keeps_update_magnitude_in_the_healthy_band_at_more_lrs": None,
    "normalize_centers_to_zero_mean_unit_std": None,
    "scaled_inputs_train_across_a_wider_lr_range": None,
    "update_magnitude_is_identical_across_all_three_pipelines": None,
    "raw_only_trains_at_the_smallest_lr": None,
    "a_larger_lr_is_always_better_after_scaling": None,
}

# What is the smallest learning rate in the sweep at which the raw
# (unscaled) inputs produce a rising validation-accuracy curve?
# Type: float - one of 0.001, 0.01, 0.1, 1.0
n_best_lr_for_raw: float | None = None

# Which scaling achieves the highest validation accuracy overall?
# Type: str — one of "raw" / "toarray" / "normalized"
n_best_scaling: str | None = None


# ---------------------------------------------------------------------------
# Task O - Hyperparameter Experiments
# ---------------------------------------------------------------------------

# After running
#   make demo-task-o
# open results/figures/hyperparameters.png and answer the following.

o_observations: dict[str, bool | None] = {
    "regularization_increases_training_accuracy": None,
    "adam_always_reaches_higher_final_val_accuracy_than_sgd": None,
    "adam_converges_in_fewer_epochs_than_plain_sgd": None,
    "weight_decay_reduces_the_train_val_gap": None,
    "momentum_speeds_up_plain_sgd": None,
    "dropout_reduces_the_train_val_gap": None,
    "unregularized_model_has_the_largest_train_val_gap": None,
}

# Which optimizer reached high validation accuracy in the fewest epochs?
# Type: str — one of "sgd" / "momentum" / "adam"
o_fastest_optimizer: str | None = None

# Which regularization setting achieved the highest validation accuracy?
# Type: str — one of "none" / "dropout" / "weight_decay"
o_best_regularizer_for_val: str | None = None


# ---------------------------------------------------------------------------
# Task P - Weight Visualization
# ---------------------------------------------------------------------------

# After running
#   make demo-task-p
# open results/figures/learned_features.png and answer the following.

p_observations: dict[str, bool | None] = {
    "with_few_units_each_filter_is_more_distinct": None,
    "filters_are_not_pure_random_noise": None,
    "with_many_units_filters_look_more_redundant_or_distributed": None,
    "first_layer_weight_columns_reshape_to_28x28": None,
    "increasing_units_makes_each_filter_a_sharper_digit": None,
    "some_filters_resemble_strokes_or_digit_templates": None,
    "each_first_layer_unit_detects_exactly_one_digit_class": None,
}

# What is the side length (in pixels) of each filter image?
# Type: int
p_filter_image_side: int | None = None

# Does test accuracy increase from H=4 to H=64?
# Type: bool
p_more_units_higher_test_accuracy: bool | None = None
