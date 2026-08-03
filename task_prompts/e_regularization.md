# Task E — Regularization: Dropout and L2

A network with enough parameters can memorize its training set: it drives the training loss very low while doing poorly on data it has not seen. This gap between training performance and held-out (validation/test) performance is called **overfitting**. The model has fit not just the real structure in the data but also its noise.

**Regularization** is the umbrella term for techniques that fight overfitting by discouraging the model from relying too heavily on any single pattern. This task introduces two of the most common: **dropout** and **L2 regularization**.

In this task you implement **dropout only**, as a new layer. L2 is explained here so you understand the full regularization picture, but you will implement it in the next task, where it attaches to the optimizer as a weight penalty. There is nothing to code for L2 in this task.

---

## Part 1 — Dropout

### The idea

During training, dropout randomly "switches off" a fraction of the units in a layer's output on every forward pass. Concretely, each element of the input is set to zero independently with probability `p` (the *drop probability*), and the surviving elements are passed through.

```
input:          [ 1.2,  -0.7,  3.1,   0.4,  -2.0 ]
random keep:    [  1,     0,    1,     0,     1   ]   (p = 0.4 dropped)
output (raw):   [ 1.2,   0.0,  3.1,   0.0,  -2.0 ]
```

Why would deleting information help? Because the network can no longer depend on any one unit always being present. A unit cannot assume its neighbours will be there to correct for it, so each unit is pushed to learn a feature that is useful on its own. This breaks up fragile "co-adaptations" between units. Another way to see it: every forward pass uses a slightly different random sub-network, so training with dropout is loosely like training a huge ensemble of networks that share weights and then averaging them. Ensembles generalize better than any single member, and dropout gives you that benefit cheaply.

Dropout is only active **during training**. At evaluation time we want the full, deterministic network — we are no longer trying to regularize, we just want the model's best prediction — so dropout does nothing and passes its input straight through.

### Keeping the scale consistent: inverted dropout

There is one subtlety. If we zero out a fraction `p` of the units during training but use all of them at evaluation, then the *typical magnitude* of the layer's output changes between the two modes. A unit downstream that learned to expect a certain input scale during training would suddenly see larger inputs at evaluation.

We fix this with **inverted dropout** (the convention PyTorch's `nn.Dropout` uses). Let the *keep probability* be

$$
q = 1 - p .
$$

During training we divide the surviving units by `q`, scaling them *up*:

$$
\text{mask}_i =
\begin{cases}
\dfrac{1}{q} & \text{with probability } q \quad(\text{unit kept})\\[2mm]
0 & \text{with probability } p \quad(\text{unit dropped})
\end{cases}
\qquad
y_i = x_i \cdot \text{mask}_i .
$$

Because a kept unit is scaled by $1/q$ and is kept a fraction $q$ of the time, the expected value of each output equals the original input:

$$
\mathbb{E}[y_i] = q \cdot \frac{x_i}{q} + p \cdot 0 = x_i .
$$

So the *expected* scale is unchanged. The payoff is that **evaluation needs no rescaling at all** — the forward pass in evaluation mode is simply the identity, $y = x$.

### Backward pass

Dropout has no trainable parameters, so backward only needs to route the upstream gradient back to the input. The same mask that was applied in the forward pass is applied to the gradient: a unit that was zeroed contributes no gradient, and a unit that was kept (and scaled by $1/q$) has its gradient scaled by the same factor.

$$
\frac{\partial y_i}{\partial x_i} = \text{mask}_i
\qquad\Longrightarrow\qquad
\frac{\partial \mathcal{L}}{\partial x_i}
= \frac{\partial \mathcal{L}}{\partial y_i}\cdot \text{mask}_i .
$$

In evaluation mode the forward pass was the identity, so backward simply passes the gradient through unchanged.

### Train vs. evaluation mode

The `Module` base class carries a boolean `training` flag, toggled by `train()` and `eval()` (these propagate through `Sequential` to every child layer). Dropout reads this flag to decide what to do:

```
training == True   ->  sample a fresh mask, scale by 1/q, cache the mask
training == False  ->  return the input unchanged
```

This is exactly why the flag exists. Earlier layers (`Linear`, `ReLU`) behave identically in both modes, so for them `train()`/`eval()` is a no-op. Dropout is the first layer whose behaviour actually depends on the mode.

### What to implement

Implement the `Dropout` class in `src/nn/dropout.py`. It inherits from `Module`.

#### `__init__(self, p: float = 0.5, seed: int | None = None) -> None`

- **Input:**
  - `p` — the probability of dropping each unit. Must satisfy $0 \le p < 1$ (a value of `1.0` would drop everything and divide by zero). Raise `ValueError` otherwise.
  - `seed` — optional seed so the dropout mask is reproducible in tests.
- Call `super().__init__()` first.
- Store `p` on `self.p`.
- Create a generator `self.rng = np.random.default_rng(seed)`. Reusing one generator across forward calls means successive batches see different masks while the whole run stays reproducible from the seed.
- Allocate `self.mask = None`; you set it during a training forward pass.

#### `forward(self, x: np.ndarray) -> np.ndarray`

- **Input:** `x` — array of any shape.
- If `self.training` is `False`, return `x` unchanged (and there is no mask to cache).
- Otherwise, with $q = 1 - p$, build the scaled keep mask and cache it on `self.mask`:
  - `self.mask = (self.rng.random(x.shape) < q) / q`
  - This produces an array of `0` (dropped) and `1/q` (kept) values with the same shape as `x`.
- **Return:** `x * self.mask`.

#### `backward(self, dout: np.ndarray) -> np.ndarray`

- **Input:** `dout` — upstream gradient, same shape as the most recent input.
- If `self.training` is `False`, return `dout` unchanged.
- Otherwise, raise `RuntimeError` if `forward` has not been called yet (no cached mask), then return `dout * self.mask`.

Dropout has no trainable parameters, so you do **not** implement `parameters`; the base `Module` default (an empty iterator) is correct.

---

## Part 2 — L2 regularization (background only — nothing to implement here)

Dropout regularizes by perturbing *activations*. L2 regularizes the *weights* directly, by adding a penalty term to the loss that grows as the weights grow. The optimizer then has to balance fitting the data against keeping the weights small. Smaller weights mean a smoother, less extreme function that is less able to memorize noise.

Let $\mathcal{L}_{\text{data}}$ be the usual loss (e.g. cross-entropy) and let $W$ range over all the weight matrices in the network. The regularized loss is:

**L2 regularization** (also called *weight decay* or *ridge*) adds the sum of squared weights:

$$
\mathcal{L} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2}\sum_{W} \lVert W \rVert_2^2
= \mathcal{L}_{\text{data}} + \frac{\lambda}{2}\sum_{W}\sum_{i,j} W_{ij}^2 .
$$

Its gradient contribution is simply $\lambda W$, so each update nudges every weight a little toward zero — hence "weight decay." L2 shrinks weights smoothly but rarely makes them exactly zero.

Here $\lambda \ge 0$ is the **regularization strength**: $\lambda = 0$ recovers the unregularized model, and larger $\lambda$ penalizes large weights more aggressively.

The natural place for this penalty is the parameter-update step, because it only touches the weights and the gradients. You will implement L2 in **task G**, where it is added to the optimizer as a weight-decay term. **You do not implement L2 in this task** — for now it is enough to understand what it does and how it differs from dropout.

---

## Deliverables

- Implement the `Dropout` class in `src/nn/dropout.py`.
- Run `make test-e` and confirm the smoke tests pass.
- Run `make submit-e` to generate `submission.json` and upload it on the course webpage.
