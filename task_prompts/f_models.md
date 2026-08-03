# Task F — Model classes

This task introduces two classes that compose the layers from earlier tasks into complete networks: `Sequential` and `MLP`.

---

## Part 1 — `Sequential`

The `Sequential` class integrates all of the classes we have written so far together, taking a list of layers and moving outputs forward and gradients backwards. It plays the same role as `torch.nn.Sequential` in PyTorch.

`Sequential` lives in `src/nn/sequential.py` and inherits from the `Module` base class.

Note that the **loss is not part of the Sequential**. The Sequential's `forward` returns logits; you call the loss separately. Parameter updates are handled by the optimizer (introduced in Task G):

```
logits  = model(x)
loss    = loss_fn(logits, y)
optimizer.zero_grad()
dlogits = loss_fn.backward()
model.backward(dlogits)
optimizer.step()
```

This separation lets you swap losses or models independently and matches PyTorch's design.

<br />

#### `__init__(self, layers: list[Module]) -> None:`

- **Input:**
  - `layers` — An ordered list of `Module` instances.
- Call `super().__init__()` first, then store the list on `self.layers`.
- **Return:** None

You do **not** need to construct the `Linear`,`ReLU`, `Dropout`, or other layers, inside `Sequential`. These are assumed to be instantiated before they are passed to the `Linear` constructor.

<br />

#### `forward(self, x: np.ndarray) -> np.ndarray`

- **Input:** Some input to the network of shape `(num_samples, data_size)`.
- Use the `forward` method of each component to pass `x` through all network components in sequential order.
- Simple python structures recommended.
- **Return:** The logits (no softmax applied) for each sample in the input.

<br />

#### `backward(self, dout: np.ndarray) -> np.ndarray`

- **Input:** The gradient of the loss function with respect to the network output, or rather the logits.
- Use the `backward` method of each component to pass the gradient in reverse order.
- Pass the gradient back through the components of the network, starting from the last layer and going to the input layer.
- **Return:** The gradient of the loss function with respect to the model's input.

Implement the `backward` method, which takes the gradient of the loss function and uses it as input for the `backward` method of the last layer. Any given output of a `backward` method is used as input for the `backward` method of the previous layer. Continue this until there is no previous layer.

<br />

#### `parameters(self) -> Iterable[tuple[np.ndarray, np.ndarray]]`

Yield every `(parameter, gradient)` pair from every layer that owns trainable parameters. Layers without parameters (e.g. `ReLU`) contribute nothing — using `yield from layer.parameters()` for every layer is enough.

The  optimizer's `step()` iterates over `model.parameters()` and applies `param -= lr * grad` to every `(parameter, gradient)` pair, so it updates the weights and biases of every `Linear` in the model.

---

## Part 2 — `MLP`

The `MLP` class is a convenience wrapper that builds a fully-connected feed-forward network from a compact description: a list of layer widths and a single activation module. Internally it constructs a `Sequential` from alternating `Linear` and activation layers, so all of the forward, backward, and parameter-update logic you wrote in Part 1 is reused automatically.

`MLP` lives in `src/nn/mlp.py` and inherits from `Module`.

<br />

#### `__init__(self, layer_sizes: list[int], activation: str = "relu", dropout: float = 0.0, seed: int | None = None) -> None`

- **Input:**
  - `layer_sizes` — A list of integers `[in, h1, h2, ..., out]` that fully describes the network. Each consecutive pair `(layer_sizes[i], layer_sizes[i+1])` defines one `Linear` layer. Must have at least two elements (input size and output size).
  - `activation` — A string naming the hidden-layer activation to use. Accepted values are `"relu"`, `"leaky_relu"`, `"gelu"`, and `"identity"`. The MLP constructs a **fresh activation instance** for each hidden layer from this name, so each layer has its own independent cache. Defaults to `"relu"`.
  - `dropout` — Drop probability passed to a `Dropout` layer inserted after each hidden-layer activation. `0.0` (the default) disables dropout entirely; the `Dropout` layer is not added at all.
  - `seed` — Optional integer seed. Layer `i` receives `seed + i` so each `Linear` is initialized independently.
- Call `super().__init__()` first.
- Raise `ValueError` if `layer_sizes` has fewer than two elements.
- Build a `Sequential` from the resulting layer list and store it on `self.net`.
- **Return:** None

The resulting layer list for `layer_sizes = [784, 128, 10]`, `activation = "relu"`, and `dropout = 0.5` looks like:

```
Linear(784, 128)
ReLU()
Dropout(0.5)
Linear(128, 10)
```

With `dropout = 0.0` the `Dropout` layer is omitted and the list is the same as before.

<br />

#### `forward(self, x: np.ndarray) -> np.ndarray`

- **Input:** A batch of shape `(num_samples, layer_sizes[0])`.
- Delegate to `self.net.forward(x)`.
- **Return:** Prediction array of shape `(num_samples, layer_sizes[-1])`.

<br />

#### `backward(self, dout: np.ndarray) -> np.ndarray`

- **Input:** The gradient of the loss with respect to the network's output logits.
- Delegate to `self.net.backward(dout)`.
- **Return:** The gradient of the loss with respect to the network's input.

<br />

#### `parameters(self)`

- Delegate to `self.net.parameters()`.
- Yields every `(parameter, gradient)` pair from every `Linear` layer inside the network; `activation` layers contribute nothing.

---

## Deliverables

- Implement the four methods in `src/nn/sequential.py`.
- Implement `MLP` in `src/nn/mlp.py`.
- Run `make test-f` and confirm the smoke tests pass.
- Run `make submit-f` to generate `submission.json` and upload it on the course webpage.
