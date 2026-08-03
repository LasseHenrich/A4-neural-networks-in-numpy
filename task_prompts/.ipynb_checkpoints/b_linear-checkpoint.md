# Task B — Linear class

The linear layer contains multiple linear units. Within a network a linear layer will have an input size equal to the output size of the previous layer, and an output size equal to the number of linear units within the layer.

Below you will implement the `__init__`, `forward`, `backward`, and `parameters` methods of the layer in `src/nn/linear.py`. Assume that the input shape to `forward` or `backward` will be of `(num_samples, in_features)`.

The `Linear` class inherits from the `Module` base class, so `Linear(x)` is short for `Linear.forward(x)`.

#### `__init__(self, in_features: int, out_features: int, seed: int | None = None) -> None:`

Call `super().__init__()` first, then initialize the following attributes:

The layer should contain the following attributes at minimum:

- `self.W` — A two dimensional numpy array of shape `(in_features, out_features)` that contains the weights of all linear units in the layer. The initial values of these weights should be initialized to random values from a Gaussian distribution with $\mu = 0$ and $\sigma = \sqrt{\frac{2}{\mathrm{in\_features}}}$. This initialization method is known as He initialization. Use `np.random.default_rng(seed)` to construct a generator so the initialization is reproducible.
- `self.b` — A one dimensional numpy array of shape `(out_features,)` that contains the biases for each linear unit, initialized to zero. It will broadcast across the batch dimension during the forward pass.
- _Gradient and cache attributes_: To increase readability we declare attributes that will be used to store intermediate data required for each method but are not directly set in `__init__`. Initialize the gradients `self.dW` and `self.db` to zero arrays of the right shape, and set `self.x = None`, which will hold the input from the most recent `forward` call.

#### `forward(self, x: np.ndarray) -> np.ndarray:`

- **Input:** Some ndarray of shape `(num_samples, in_features)`
- Save the passed input to `self.x`, as this is required when calculating the gradient in `backward`.
- **Returns:** The linear transformation of passed inputs.

The output of the $j^{\text{th}}$ linear unit for one input $x$ is defined as:

$$
z_j = w_{1j}x_1 + w_{2j}x_2 + \ldots + w_{nj}x_n + b_j
$$

Note that for an input of size `(num_samples, in_features)` a calculation of this kind will be done `num_samples * out_features` times for a given call to forward.

#### `backward(self, dout: np.ndarray) -> np.ndarray:`

- **Input:** gradient of the loss function with respect to every output value of its own units from the previous call to `forward`.
- Calculate and internally save the gradient of the loss function with respect to each weight and bias in the layer (`self.dW` and `self.db`).
- Raise `RuntimeError` if `forward` has not been called yet.
- **Returns:** the gradient of the loss function with respect to its own inputs.

Below are the gradient calculations for individual weight, bias, and input values, where $i$ is the weight index within a linear unit, $j$ is the index of the unit within the layer, and $k$ is the index of a sample within the input.

$$
\frac{\partial \mathcal{L}}{\partial w_{ij}} = \frac{\partial \mathcal{L}}{\partial z_j} \cdot \frac{\partial z_j}{\partial w_{ij}} = \frac{\partial \mathcal{L}}{\partial z_j} * x_i \;\; \text{or} \;\; \sum_k \frac{\partial \mathcal{L}}{\partial z_{jk}} * x_{ik}
$$

$$
\frac{\partial \mathcal{L}}{\partial b_j} = \frac{\partial \mathcal{L}}{\partial z_j} \cdot \frac{\partial z_j}{\partial b_j} = \frac{\partial \mathcal{L}}{\partial z_j} \;\; \text{or} \;\; \sum_k \frac{\partial \mathcal{L}}{\partial z_{jk}}
$$

$$
\frac{\partial \mathcal{L}}{\partial x_{ik}} = \sum_j \frac{\partial \mathcal{L}}{\partial z_{jk}} \cdot \frac{\partial z_{jk}}{\partial x_{ik}} = \sum_j \frac{\partial \mathcal{L}}{\partial z_{jk}} * w_{ij}
$$

All of this can be done via matrix multiplications.

#### `parameters(self) -> Iterable[tuple[np.ndarray, np.ndarray]]:`

Yield the layer's `(parameter, gradient)` pairs in the order `(self.W, self.dW)` then `(self.b, self.db)`. The optimizer (introduced in Task G) iterates over these pairs and applies one update step:

$$
\begin{align*}
w_{ij} \leftarrow w_{ij} - \eta \frac{\partial \mathcal{L}}{\partial w_{ij}}
&\;\;\;\text{and}\;\;\;&
b_{j} \leftarrow b_{j} - \eta \frac{\partial \mathcal{L}}{\partial b_{j}}
\end{align*}
$$

where $\eta$ is the learning rate.

#### Tips, Hints, and Sanity Checks

- Your gradient calculations during `backward` should be the same `shape` as the attribute it is in respect to (e.g. `self.dW.shape == self.W.shape`).
- Consider the desired `shape` of any attribute you are calculating. Considering the shapes of other attributes and how `@` and `+` operations transform shape, how could we create a value with the desired shape?
- When writing gradients in `backward`, accumulate in-place: either `self.dW += new_grads` or `self.dW[...] = self.dW + new_grads`. Both update the existing array, so any other variable holding a reference to `self.dW` — such as the optimizer — automatically sees the updated gradient. A bare `self.dW = new_grads` rebinds the Python variable to a new array and silently breaks that reference.

## Deliverables

- Implement the four methods in `src/nn/linear.py`.
- Run `make test-b` and confirm the smoke tests pass.
- Run `make submit_b` to generate `submission.json` and upload it on the course webpage.
