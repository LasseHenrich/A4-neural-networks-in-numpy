# Assignment 4 — Neural Networks in NumPy

A from-scratch neural network implemented using NumPy.
You will build the network piece by piece across sixteen tasks.

## Setup

```bash
make install
```

## Tasks

Read the prompt files in `task_prompts/` in order.
Tasks b–j require implementation; tasks k–p are observe-only experiments.

| Prompt | Topic | Files to implement |
|--------|-------|--------------------|
| `b_linear.md` | Linear layer | `src/nn/linear.py` |
| `c_activation.md` | ReLU, LeakyReLU, GELU | `src/nn/activations.py` |
| `d_loss.md` | CrossEntropy and MSE | `src/nn/loss.py` |
| `e_regularization.md` | Dropout | `src/nn/dropout.py` |
| `f_models.md` | Sequential and MLP | `src/nn/sequential.py`, `src/nn/mlp.py` |
| `g_optim.md` | SGD and Adam | `src/optim/sgd.py`, `src/optim/adam.py` |
| `h_data-handling.md` | Dataset and DataLoader | `src/data/dataset.py`, `src/data/dataloader.py` |
| `i_metrics.md` | Accuracy and ConfusionMatrix | `src/evaluation/accuracy.py`, `src/evaluation/confusion_matrix.py` |
| `j_training.md` | Training loop | `src/training/train_ffnn.py` |

## Running tests

```bash
make test-b   # Linear layer (task b)
make test-c   # Activation functions (task c)
make test-d   # Loss functions (task d)
make test-e   # Dropout (task e)
make test-f   # Sequential and MLP (task f)
make test-g   # SGD and Adam (task g)
make test-h   # Dataset and DataLoader (task h)
make test-i   # Accuracy and ConfusionMatrix (task i)
make test-j   # Training loop (task j)
make test     # All tests
```

## Running experiment demos

```bash
make demo-task-j   # learning-rate sweep (task j)
make demo-task-k   # decision boundaries (task k)
make demo-task-l   # training diagnostics (task l)
make demo-task-m   # regression (task m)
make demo-task-n   # input scaling (task n)
make demo-task-o   # hyperparameter experiments (task o)
make demo-task-p   # first-layer weight visualization (task p)
```

## Submitting

```bash
make submit-b   # submit task b (Linear layer)
make submit-c   # submit task c (activations)
make submit-d   # submit task d (loss functions)
make submit-e   # submit task e (Dropout)
make submit-f   # submit task f (Sequential and MLP)
make submit-g   # submit task g (SGD and Adam)
make submit-h   # submit task h (Dataset and DataLoader)
make submit-i   # submit task i (metrics)
make submit-j   # submit task j (training loop)
make submit     # submit all tasks
```

This writes `submission.json`, which you upload to the course webpage.

## Reference mode

A compiled reference implementation is included in `src_reference/`. You
can use it to run demos and tests without having implemented anything
yourself.

```bash
make use-reference   # switch src/ to the reference implementation
make use-student     # switch back to your own implementation
```
