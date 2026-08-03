# Task A — Introduction and Theoretical Explanation

In this assignment we'll be implementing a basic fully-connected feed-forward neural network using NumPy. We will then be using it to classify characters in the MNIST dataset.

By fully-connected we mean that the output for each unit in the prior layer is given as input for each unit in the following layer.

By feed-forward we mean that the model has a series of layers in which the output of one layer, either fully or partially, is used as input for a subsequent layer, but not any prior layer.

Below is a diagram of a very simple fully-connected feed-forward neural network. Each layer in the diagram is represented by a column of dots, which represent units.

<div style="text-align: center;">
  <img src="imgs/nn-diagram.png" alt="Fully-connected feed-forward neural network diagram" width="400"/>
  <p style="text-align: center;">
    Source: Gow, Stephen & Niranjan, Mahesan & Pearman-Kanza, Samantha & Frey, Jeremy. (2022).
    A Review of Reinforcement Learning in Chemistry. <em>Digital Discovery. 1.</em> 10.1039/D2DD00047D.
  </p>
</div>

Mathematically we can think of these networks as a series of composed functions, expressed as:

$$ o = \color{#00b446}{f_3(}\color{#4040b8}{f_2(}\color{purple}{f_1(}\color{#f95b00}{x}\color{purple}{)}\color{#4040b8}{)}\color{#00b446}{)} $$

where
- $x$ is the input
- $f_i$ is the $i^{th}$ fully-connected layer
- $o$ is the output of the network

Note that the above diagram has three layers in total, two hidden layers and an output layer. Each layer has multiple units which receive input and send their output to the following layer. Each unit is comprised of two components: a linear function and an activation function. In mathematical notation we could state this as:

$$ o = a(w_1x_1 + w_2x_2 + \ldots + w_nx_n + b) $$

where
- $o$ is the output of the unit
- $a$ is the activation function, a non-linear function that we discuss more later in the homework
- $w_i$ is the $i^{th}$ weight of the linear function
- $x_i$ is the $i^{th}$ component of the input
- $b$ is the bias term

or rather, expressing the output of the whole layer,

$$ o^{(i)} = a(W^{(i)}x^{(i)} + b^{(i)}) $$

where
- $o^{(i)}$ is the output of the layer, a one dimensional vector with the same length as the number of units in the layer
- $x^{(i)}$ is the input into the $i^{th}$ layer, which is often equal to the output of the previous layer $o^{(i-1)}$
- $W^{(i)}$ is a matrix with dimensions `(input_size, num_units)` that holds the weights of all linear units in the layer
- $b^{(i)}$ is the bias term which is a one dimensional vector with the same length as the number of units in the layer

Let's combine these formulations to express the above network in terms of **functional composition**. The components of a given layer are shown in the same color.

$$
o = \color{#00b446}a^{(3)}(W^{(3)}\color{#4040b8}a^{(2)}(W^{(2)}\color{purple}a^{(1)}(W^{(1)}(\color{#f95b00}x\color{purple}) + b^{(1)})\color{#4040b8} + b^{(2)})\color{#00b446} + b^{(3)})
$$

From this you should have an idea of how an input is evaluated by this kind of neural network. In this homework we start by writing classes for each of the network's components and then combining them at the end with a `Model` class. Each of the classes we'll be implementing today will have a `forward` method which will do the portion of the above computation for the respective component.

We will also implement a `backward` method which is used during training and will compute the gradient of `forward` as a part of a larger gradient computation.

To make a network useful and accurately classifying inputs we have to adjust and tune all of the weights and biases in each unit. To do this we'll use a method called stochastic gradient descent (SGD) with mini-batches. This technique uses a loss function, often denoted with $\mathcal{L}$, which measures how 'wrong' the network is. The goal of training is to adjust and tune the weights and biases of the network such that $\mathcal{L}$ is minimized. To this end we calculate the gradient of $\mathcal{L}$ with respect to some single weight or bias, which tells us which way we should adjust a parameter to reduce the value of $\mathcal{L}$. In mathematical notation, where $z$ is the output of a linear function, or rather the pre-activation value of the unit:

$$ \frac{\partial\mathcal{L}}{\partial w} = \frac{\partial\mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w} \;\;\; \text{or} \;\;\; \frac{\partial\mathcal{L}}{\partial b} = \frac{\partial\mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial b}$$

To calculate the gradient of the weights we'll utilize the chain rule of derivation:

$$ \frac{d}{dx}f(g(h(x))) = f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x)$$

and when calculating the gradient of the activation function we'll use the additive rule of derivation as well:

$$ \frac{d}{dx}(f(x) + g(x)) = f'(x) + g'(x)$$

Note from the above rules that we will need to be able to calculate the first derivative of every component with respect to one of the inputs. Practically, our implementation will calculate the derivative for every input in parallel, or rather the gradient, which is what `backward` does.

Training a network is an iterative process where we find the network's current outputs on some labeled data samples (`forward`), find the gradients of each parameter (`backward`), and adjust each parameter so the value of the loss function decreases for said inputs (`step`), usually until the values of the loss function stabilize at a minimum.

## The `Module` base class

Every layer, loss, and model in this assignment inherits from a tiny `Module` base class, defined in `src/nn/module.py`. This class is provided — you do not need to modify it. It mirrors the calling conventions of PyTorch's `nn.Module`:

- `module(x)` is short for `module.forward(x)`.
- `module.parameters()` yields `(param, grad)` pairs that `step` consumes for SGD updates.
- Subclasses override `forward` and `backward`. Subclasses that hold trainable weights also override `parameters`.
- The default `step(lr)` walks `self.parameters()` and applies one in-place SGD step `param -= lr * grad`, so subclasses that expose their parameters get the SGD update for free.
- `Module.__init__` sets up the `training` flag (used by layers like `Dropout`), so **every subclass constructor you write must call `super().__init__()` as its very first line.**

In the tasks that follow you will inherit from `Module` and override the relevant methods.

There are no tasks to complete in this section. Read on to the next task to begin implementing.
