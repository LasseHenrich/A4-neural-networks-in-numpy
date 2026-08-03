# Task C — Activation functions: ReLU, Leaky ReLU, and GELU

Between each layer of a neural network you'll find a non-linear function. This is the key feature that allows a neural network to approximate non-linear functions. If we were to not have a non-linear function between layers, we'd have a linear model, and all multi-layer linear models can be reduced to a single linear calculation (I'll leave the proof as a fun exercise that has no applicability to the course).

In this task you will implement three activation functions in `src/nn/activations.py`. All three are common choices for the hidden layers of a multilayer perceptron: ReLU is the standard default, Leaky ReLU is a small variation that fixes one of ReLU's failure modes, and GELU is the smooth activation favoured by modern transformer architectures. Each class inherits from the `Module` base class.

---

## `ReLU`

A rectified linear unit (ReLU) is the most commonly used activation function in hidden layers of modern networks.

#### `__init__(self) -> None:`

- Call `super().__init__()` first.
- ReLU itself doesn't contain any trainable parameters, however, we will still need to save information about the input during `forward` so that `backward` can use it. Allocate `self.mask = None`; in `forward` you will set it to the boolean mask `x > 0`.

#### `forward(self, x: np.ndarray) -> np.ndarray:`

- **Input:** array of shape `(num_samples, input_dim)`
- Cache the boolean mask `x > 0` on `self.mask`.
- **Returns:** The input with the ReLU function, $a^{(i)} = \mathrm{ReLU}(x) = \mathrm{max}(0, x)$, applied in an element-wise fashion. Element-wise multiplication of `x` with the cached mask achieves this.

#### `backward(self, dout: np.ndarray) -> np.ndarray:`

- **Input:** gradient of the cost function with respect to every output value of its own units from the previous call to `forward`.
- Raise `RuntimeError` if `forward` has not been called yet.
- **Return:** The gradient of the cost function with respect to each input, as defined below.

The partial derivative of the ReLU function is below, setting the derivative at $x = 0$ to $0$:

$$
\frac{\partial \mathrm{ReLU(x)}}{\partial x} =
\begin{cases}
    1 & \text{if } x > 0, \\
    0 & \text{if } x \leq 0
\end{cases}
$$

This is exactly the mask you cached in `forward`. Multiplying `dout` by `self.mask` element-wise zeroes out the gradient at positions where the input was non-positive.

Below is the derivation of the cost function with respect to the input:

$$
\frac{\partial \mathcal{L}}{\partial z_i} = \frac{\partial \mathcal{L}}{\partial a_i} \cdot \frac{\partial a_i}{\partial z_i}
$$

`ReLU` has no parameters, so you do **not** need to implement `parameters` or `step`; the base `Module` defaults are correct.

---

## `LeakyReLU`

Standard ReLU has a weakness: once a unit's input is non-positive, its output and its gradient are both zero. If a unit gets stuck in that region it stops learning entirely — the "dying ReLU" problem. Leaky ReLU fixes this by letting a small, fixed slope $\alpha$ (the *negative slope*, default $0.01$) through for non-positive inputs, so the gradient is never exactly zero:

$$
\mathrm{LeakyReLU}(x) =
\begin{cases}
    x & \text{if } x > 0, \\
    \alpha x & \text{if } x \leq 0
\end{cases}
$$

#### `__init__(self, negative_slope: float = 0.01) -> None:`

- Call `super().__init__()` first.
- Store the negative slope on `self.negative_slope` so `forward` and `backward` can read it.
- Allocate `self.mask = None`; as with ReLU you will set it to the boolean mask `x > 0` in `forward`.

#### `forward(self, x: np.ndarray) -> np.ndarray:`

- **Input:** array of any shape.
- Cache the boolean mask `x > 0` on `self.mask`.
- **Returns:** the input where positive, and the input scaled by `self.negative_slope` where non-positive. `np.where(self.mask, x, self.negative_slope * x)` does this in one element-wise call.

#### `backward(self, dout: np.ndarray) -> np.ndarray:`

- **Input:** upstream gradient, same shape as the cached input.
- Raise `RuntimeError` if `forward` has not been called yet.
- **Return:** the gradient of the loss with respect to the input.

The derivative is $1$ where the input was positive and $\alpha$ where it was not:

$$
\frac{\partial \mathrm{LeakyReLU}(x)}{\partial x} =
\begin{cases}
    1 & \text{if } x > 0, \\
    \alpha & \text{if } x \leq 0
\end{cases}
$$

Build this local gradient with `np.where(self.mask, 1.0, self.negative_slope)` and multiply it by `dout` element-wise.

---

## `GELU`

The Gaussian Error Linear Unit (GELU) is a smooth activation used throughout modern transformer architectures (BERT, GPT-style models). Unlike ReLU's hard kink at $0$, GELU transitions smoothly between suppressing and passing its input. We use the **tanh approximation**, which keeps the implementation in pure NumPy:

$$
\mathrm{GELU}(x) = 0.5\,x\left(1 + \tanh\!\big(C\,(x + A\,x^3)\big)\right)
$$

where $C = \sqrt{2/\pi}$ and $A = 0.044715$. These two constants are provided as the class attributes `GELU.C` and `GELU.A`.

#### `__init__(self) -> None:`

- Call `super().__init__()` first.
- Allocate `self.x = None`. Unlike the ReLU-family activations, GELU's gradient depends on the input value itself (not just its sign), so you will cache the full input array in `forward`.

#### `forward(self, x: np.ndarray) -> np.ndarray:`

- **Input:** array of any shape.
- Cache the input on `self.x`.
- **Returns:** $\mathrm{GELU}(x)$ using the formula above, with `np.tanh`. Reference the constants as `self.C` and `self.A`.

#### `backward(self, dout: np.ndarray) -> np.ndarray:`

- **Input:** upstream gradient, same shape as the cached input.
- Raise `RuntimeError` if `forward` has not been called yet.
- **Return:** the gradient of the loss with respect to the input.

Write $u = C\,(x + A\,x^3)$ so that $\mathrm{GELU}(x) = 0.5\,x\,(1 + \tanh u)$. Applying the product rule (and the chain rule on $\tanh$, whose derivative is $1 - \tanh^2 u$) gives the local gradient:

$$
\frac{\partial \mathrm{GELU}(x)}{\partial x} = 0.5\,(1 + \tanh u) + 0.5\,x\,(1 - \tanh^2 u)\cdot u'
\qquad\text{where}\qquad
u' = C\,(1 + 3A\,x^2)
$$

Compute this from the cached `self.x`, then multiply by `dout` element-wise. A finite-difference check against the forward pass is a good way to confirm your derivative is correct.

---

None of these activations have trainable parameters, so for all three you do **not** need to implement `parameters` or `step`; the base `Module` defaults are correct.

## Deliverables

- Implement all three classes in `src/nn/activations.py`.
- Run `make test-c` and confirm the smoke tests pass.
- Run `make submit-c` to generate `submission.json` and upload it on the course webpage.
