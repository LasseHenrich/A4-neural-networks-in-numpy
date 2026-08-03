# Task O — The Knobs: Optimizers and Regularization

This is the fourth of five observe-only tasks. You do not implement anything here. You run a provided demo, study the figure it produces, and record what you saw in `answers.py`.


## Background

So far you have trained networks without thinking much about two of the choices baked into the training loop: which optimizer applies the parameter updates, and whether anything is done to keep the model from memorising the training data. You built the pieces for both in earlier tasks. You built SGD and Adam in task g, with momentum and weight decay as options on SGD. You built Dropout in task e. This task is where those pieces pay off. The demo compares them on real MNIST data so you can see, rather than assume, how each one changes training.

The background splits into two sub-topics, matching the two panels of the figure.

### Optimizers and convergence speed

An optimizer is the rule that turns gradients into parameter updates. All three optimizers below see the same gradients; they differ only in what they do with them.

**Plain SGD** updates each parameter by

```
w  <-  w - lr * grad
```

It works, but it moves at the same pace in every direction. When the loss surface is elongated — steep in some directions and shallow in others, which is the usual case — a single learning rate is a poor fit for all directions at once, and progress can be slow.

**Momentum** keeps a running average of past gradients, called the *velocity*, and steps in the direction of that average instead of the raw gradient:

```
velocity  <-  momentum * velocity + grad
w         <-  w - lr * velocity
```

This has two effects. It smooths out noisy gradient updates, because averaging cancels out the jitter from one batch to the next. And it accelerates in directions where the gradients consistently point the same way, building up "momentum" the way a rolling ball does. In practice it converges faster than plain SGD.

**Adam** (Adaptive Moment Estimation) tracks two running averages: the mean of the gradients, like momentum, and the mean of the *squared* gradients. The second quantity estimates the *scale* of the gradients in each direction, and Adam divides the update by that scale. The result is that each parameter gets its own effective learning rate, adjusted automatically to its gradient history. Adam typically converges in fewer epochs than plain SGD and is more robust to the choice of learning rate, which is why it is a common default. You built the full update in task g, so the derivation is not repeated here.

All three optimizers are compared on the same model, the same data, and the same starting weights. The only thing that changes between the three runs is the optimizer. Any difference in the curves reflects convergence behaviour — not model capacity, not the data.

### Regularization and the train/val gap

**Overfitting** happens when a model has enough capacity to memorise the training data rather than learning a rule that generalises. The sign of overfitting is a large gap between training accuracy and *validation* accuracy: the model scores well on the images it was trained on and worse on held-out images it has never seen. This is the same train/test gap you met on the regression demo in task m, now measured on MNIST as a train/val gap.

This demo deliberately *causes* overfitting. It trains on a small subset of MNIST — a few thousand images instead of the full 60 000 — while using a network large enough to memorise that subset. The full validation set, which the model never trains on, then measures how well the model generalises. With so little training data and so much capacity, the unregularized model overfits badly, and the gap is wide.

Two techniques reduce the gap.

- **Dropout** (task e): during training, each hidden unit is independently switched off with probability `p`. This stops any single unit from becoming too important and forces the network to learn redundant representations that generalise better. At test time all units are active, with their outputs scaled to match the expected training behavior — the inverted-dropout scaling you implemented. Dropout usually *lowers* training accuracy, because the network is deliberately impaired during training, while *raising* validation accuracy.

- **Weight decay** (L2 regularization, built into the SGD constructor in task g): a small penalty proportional to the square of each weight is added to the loss. This pushes the weights toward small values, which smooths out the function the network learns and makes it less able to fit every quirk of the training set. Like dropout, weight decay typically lowers training accuracy while improving validation accuracy.

There is one counter-intuitive point that is easy to miss, so it is worth stating plainly. Regularization almost always makes *training* accuracy **worse**. The network is constrained, so it can no longer fit the training data as tightly. But that same constraint forces it to learn something more general, so *validation* accuracy goes **up**. The goal of regularization is not a high training score — it is a smaller gap between training and validation.


## Run the Experiment

Run the demo:

```
make demo-task-o
```

This runs two sub-experiments on a small subset of normalized MNIST inputs — the same `Compose([ToArray(), Normalize()])` pipeline from task l — and writes a single figure with two panels side by side:

```
results/figures/hyperparameters.png
```

**Panel A — Optimizers.** Training and validation accuracy across epochs for three optimizers: plain SGD, SGD with momentum, and Adam. Each optimizer contributes two curves, one for training accuracy and one for validation accuracy. All three runs use the same model, data, and starting weights. Compare how quickly each one reaches good accuracy, and where it ends up.

**Panel B — Regularization.** Training and validation accuracy across epochs for three regularization settings: no regularization, dropout, and weight decay. Each setting contributes two curves — solid for training, dashed for validation — trained on a smaller subset of MNIST than Panel A. The visual point is the *gap* between each condition's solid and dashed curves: widest when the model is unconstrained, narrower when regularization is applied.

What to look for:

- **Panel A.** Adam and momentum should reach high validation accuracy in fewer epochs than plain SGD. Plain SGD gets there too, just more slowly. Note that "faster to converge" does not guarantee "higher final accuracy" — by the end of training the three can land close together.
- **Panel B.** The unregularized model should show the largest gap between its training and validation curves. Dropout narrows that gap most visibly: its training curve is noisier and settles lower than the unregularized training curve, while the validation curve holds at a similar or slightly higher level — the constraint of randomly disabling units forces more distributed learning. Weight decay has a subtler effect on these curves; look carefully at whether its validation curve sits slightly above the unregularized baseline. Note that regularization can reduce the gap by pulling the training curve down, by pushing the validation curve up, or both.


## Deliverables

- Run `make demo-task-o` and open `results/figures/hyperparameters.png`.
- Study both panels, then fill in the following keys in `answers.py`:

```python
o_observations: dict[str, bool | None]
# Keys (all start as None — mark True if you observed it, False if not):
#   adam_converges_in_fewer_epochs_than_plain_sgd
#   momentum_speeds_up_plain_sgd
#   adam_always_reaches_higher_final_val_accuracy_than_sgd
#   unregularized_model_has_the_largest_train_val_gap
#   dropout_reduces_the_train_val_gap
#   regularization_increases_training_accuracy
#   weight_decay_reduces_the_train_val_gap

o_fastest_optimizer: str
# Which optimizer reached high validation accuracy in the fewest epochs?
# One of: "sgd" / "momentum" / "adam"

o_best_regularizer_for_val: str
# Which regularization setting achieved the highest validation accuracy?
# One of: "none" / "dropout" / "weight_decay"
```

- Run `make submit-o` to generate `submission.json` and upload it on the course webpage.
