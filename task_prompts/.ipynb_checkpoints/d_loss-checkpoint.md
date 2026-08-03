# Task D — Loss functions

The loss function is a measure of how far off the results of the network are from the actual results. The particular function that is used is often dependent on the application and type of prediction being made. When we're trying to predict some real-valued number, we would often use average mean squared error. For categorical data, which is what is found in the MNIST dataset, we'll use a cross entropy loss.

## Categorical encoding

First though, we need to consider how we'll represent this categorical information. It's often the case when working with processed training data that different categories will be represented by a number, which can be thought of as the index of some list of categorical labels. This is what the MNIST dataset provides, however, given that the labels of the MNIST dataset are those of handwritten digits, we have the convenient and _coincidental_ benefit that the index for each category has a very literal relationship to the category label.

Categorical models will often output a discrete probability distribution over all categories, literally a list of numbers that sum to one and is as long as the number of categories, each number estimating the likelihood that the input is a member of a particular category rather than one of the others. Given this output from our model, it can be helpful to think of our labels as being in this format. Since we assume the labels of our training data to be correct with $p=1.0$ each label will be a list with a single `1` in the index of the correct category. This is often called **one-hot encoding**. For example, for the category at index 3, which in our case is _coincidentally_ the digit "3", we have:

`3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`

The above list is of length 10 because we are classifying into 10 different categories.

To be clear, during implementation it is not necessary to directly convert each label into this format, but the output of the loss function should be the same as if they were in this format. Our `CrossEntropy` class will take integer class indices, matching PyTorch's `nn.CrossEntropyLoss`.

## Softmax

One problem we have is that the raw output of the network, called logits and denoted by $\hat y$, almost never sums to 1 and the values will likely not all be in $[0,1]$, as is a requirement for a probability distribution. We can, however, force this property by normalizing the output. To do this we'll apply a **softmax** function to $\hat y$. The $i^{th}$ entry the distribution $q$, which is $\hat y$ with softmax applied is:

$$
q_i = \frac{e^{\hat y_i}}{\sum_{j} e^{\hat y_j}}
= \frac{e^{-\max_{j} \hat y_j}}{e^{-\max_{j} \hat y_j}} \cdot \frac{e^{\hat y_i}}{\sum_{j} e^{\hat y_j}}
= \frac{e^{\hat y_i - \max_{j} \hat y_j}}{\sum_{k} e^{\hat y_k - \max_{j} \hat y_j}}
$$

For numerical stability we will multiply both the numerator and denominator by $e^{-\max_j \hat y_j}$. This is due to the potential for exponents of large numbers to overflow into `inf`, which would break the training.

Since $e^x$ is monotonically increasing, or rather only increases as $x$ increases, we still have the same relative ordering of entries in the output, although the relative difference between the entries has changed. This relative ordering is important, as during inference we can still find the most likely category by using the largest value in the output without applying the softmax.

## Cross-entropy

The loss function $\mathcal{L}(p, q)$ we'll be using is called **Cross-entropy Loss**. Cross-entropy $H(p, q)$ measures the difference between two probability distributions and is expressed as such:

$$
H(p, q) = -\sum^{K}_{k=1} p_k \log( q_k)
$$

where
- $p$ is the true label with $p_k$ being the $k^{th}$ element of $p$.
- $q$ is the network output with the softmax applied, with $q_k$ being the $k^{th}$ element of $q$.
- And $K$ is the number of elements in $p$ and $q$

And the loss is the average cross-entropy across $N$ samples:

$$
\mathcal{L}(P, Q) = -\frac{1}{N}\sum^{N}_{n=1}\sum^{K}_{k=1} p_{n,k} \log( q_{n,k})
$$

where
- $N$ is the number of samples being calculated in the loss function.
- $P$ is the true distribution for a list of samples.
- $Q$ is the network's predicted distributions for the samples corresponding to $P$.
- $p_{n,k}$ is the $k^{th}$ element of the $n^{th}$ sample label in $P$.
- $q_{n,k}$ is the $k^{th}$ element of the $n^{th}$ network output with softmax in $Q$.


## Calculating the gradients of the loss

For training we must calculate the gradient of $\mathcal{L}(p, q)$ with respect to the network's outputs, or in mathematical notation:

$$
\frac{\partial \mathcal{L}}{\partial \hat y_i} = \frac{\partial \mathcal{L}}{\partial q_i} \cdot \frac{\partial q_i}{\partial \hat y_i} + \sum_{i \neq j} \frac{\partial \mathcal{L}}{\partial q_i} \cdot \frac{\partial q_i}{\partial \hat y_j}
$$

The partial derivative of $\mathcal{L}$ with respect to a single softmax value $q_i$, and the partial derivative of the softmax entry $q_i$ with respect to the network output for that entry $\hat y_j$ are stated below:

$$
\begin{align*}
\frac{\partial \mathcal{L}}{\partial q_i} =
\begin{cases}
    -\frac{1}{N} \cdot \frac{1}{q_i} & \text{if } i = y \\
    0 & \text{if } i \neq y
\end{cases}
& \; \; \; \; \; &
\frac{\partial q_i}{\partial \hat y_j} =
\begin{cases}
    q_i(1-q_i) & \text{if } i = j \\
    -q_i q_j & \text{if } i \neq j
\end{cases}
\end{align*}
$$

Putting these together we'll consider the cases of the logit with the correct class ($i = y$) and the logits of the incorrect classes ($i \neq y$).

$$
\frac{\partial \mathcal{L}}{\partial \hat y_i} =
\begin{cases}
    -\frac{1}{N} \cdot \frac{1}{q_y} \cdot q_y(1 - q_y) = -\frac{1 - q_y}{N} = \frac{q_y - 1}{N} & \text{if } i = y \\
    -\frac{1}{N} \cdot \frac{1}{q_y} \cdot (-q_i q_y) = \frac{q_i}{N} & \text{if } i \neq y
\end{cases}
$$

Considering what we know about one hot encoding, we can generalize this to:

$$
\frac{\partial \mathcal{L}}{\partial \hat y_i} = \frac{1}{N} (q_i - p_i)
$$

## Mean squared error

Cross-entropy is built for classification, where the target is a category and the network outputs a probability distribution. Many problems instead ask the network to predict a real-valued number (or a vector of them) — for example the price of a house or the coordinates of a point. For these **regression** problems we use **Mean Squared Error (MSE)**.

MSE measures the average of the squared differences between the prediction $\hat y$ and the target $y$. For a single sample with $K$ output values:

$$
\mathcal{L}(\hat y, y) = \frac{1}{K} \sum^{K}_{k=1} (\hat y_k - y_k)^2
$$

and across $N$ samples we average over every element, so the denominator is the total number of elements $N \cdot K$:

$$
\mathcal{L}(\hat Y, Y) = \frac{1}{N K} \sum^{N}_{n=1} \sum^{K}_{k=1} (\hat y_{n,k} - y_{n,k})^2
$$

Squaring does two things: it makes every difference positive (so errors do not cancel out), and it penalizes large errors much more heavily than small ones.

The gradient is straightforward because there is no softmax in the way. Differentiating the loss with respect to a single prediction $\hat y_i$ gives:

$$
\frac{\partial \mathcal{L}}{\partial \hat y_i} = \frac{2}{NK} (\hat y_i - y_i)
$$

In words: the gradient points away from the target, and its magnitude is proportional to how far off the prediction is.

## Comparing the two loss functions

Both classes return a single scalar from `forward` and the gradient of that scalar with respect to the network's output from `backward`, so they plug into the same training loop. The differences are in what they expect and what they are used for:

- **Task type.** Cross-entropy is for **classification**: the target is a category. MSE is for **regression**: the target is a real-valued number.
- **Targets.** `CrossEntropy` takes integer class indices of shape `(B,)`. `MSE` takes a real-valued target array with the **same shape** as the prediction.
- **Output of the network.** Cross-entropy applies a **softmax** internally to turn raw logits into a probability distribution before measuring the loss. MSE applies no such transformation — it compares the raw predictions to the targets directly.
- **Why not swap them.** Using MSE on a classification problem is possible but works poorly: it treats the output values as independent numbers rather than competing probabilities, and the gradient signal when the network is confidently wrong is weak. Cross-entropy paired with softmax gives a stronger, well-behaved gradient for classification, which is why it is the standard choice (and the one we use for MNIST).

## Writing the `CrossEntropy` Class

The `CrossEntropy` class inherits from the `Module` base class and lives in `src/nn/loss.py`. It has no trainable parameters, so you do not need to implement `parameters` or `step`.

#### `__init__(self) -> None`

- Call `super().__init__()` first.
- Declare attributes which will be set in `forward` and used by `backward`. Allocate `self.probs = None` and `self.y = None`.

#### `forward(self, logits: np.ndarray, y: np.ndarray) -> float`

- **Input:**
    - `logits` — The raw network output of shape `(B, C)`, before softmax is applied.
    - `y` — A `(B,)` array of integer class labels in `[0, C)`.
- Save the predicted probabilities (`self.probs`) and labels (`self.y`) for `backward`.
- Be sure to apply softmax to the input using the max-subtraction trick described above.
- **Return:** Value of the loss function as a Python `float`.

#### `backward(self) -> np.ndarray`

Note that, unlike a layer's `backward(dout)` which receives an upstream gradient from the next component, the loss's `backward()` takes **no argument** — the loss is where backpropagation begins, so it computes and returns dL/dz directly from its own cached state.

- **Input:** Nothing.
- Raise `RuntimeError` if `forward` has not been called yet.
- You will probably have to make a copy of your stored predicted probabilities.
- **Return:** Partial derivative of the loss function with regard to every input into the loss, of shape `(B, C)`.

## Writing the `MSE` Class

The `MSE` class also inherits from `Module` and lives in `src/nn/loss.py`. Like `CrossEntropy` it has no trainable parameters, so you do not need to implement `parameters` or `step`.

#### `__init__(self) -> None`

- Call `super().__init__()` first.
- Declare an attribute that will be set in `forward` and used by `backward`. Allocate `self.diff = None`.

#### `forward(self, pred: np.ndarray, target: np.ndarray) -> float`

- **Input:**
    - `pred` — The network's predictions, of any shape.
    - `target` — The target values, the **same shape** as `pred`.
- Save what you need for `backward`. Caching the difference `pred - target` (`self.diff`) is enough.
- Do **not** apply a softmax; compare the raw predictions to the targets directly.
- **Return:** The mean squared error across all elements as a Python `float`.

#### `backward(self) -> np.ndarray`

- **Input:** Nothing.
- Raise `RuntimeError` if `forward` has not been called yet.
- **Return:** Partial derivative of the loss with respect to every prediction, the same shape as `pred`. Remember the factor of `2 / N`, where `N` is the total number of elements.

## Deliverables

- Implement the methods in `src/nn/loss.py`.
- Run `make test-d` and confirm the smoke tests pass for both the `CrossEntropy` and `MSE` classes.
- Run `make submit_d` to generate `submission.json` and upload it on the course webpage.
