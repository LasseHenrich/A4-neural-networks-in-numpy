# Task P — Representation Learning: What the Network Learned

This is the fifth of five observe-only tasks. You do not implement anything here. You run a provided demo, study the figures it produces, and record what you saw in `answers.py`.


## Background

By now you have trained networks on MNIST and watched their accuracy numbers climb. Those numbers tell you *how well* a network does, but not *what* it learned to do it. This task opens up the trained model and looks directly at its first layer to see the patterns it discovered on its own. Nothing here is new machinery — the layer you are inspecting is the same `Linear` layer you built in task b.

### Weights as feature detectors

The first hidden layer of a network trained on MNIST is a `Linear(784, H)` layer. Its weights are stored in `model.layers[0].W`, an array of shape `(784, H)`. There are H columns, one per hidden unit, and each column is a weight vector of length 784 — exactly one weight per input pixel.

Recall the forward pass: each hidden unit computes the dot product of the input with its weight column (plus a bias). That dot product is large when the input lines up with the weight column and small when it does not. During training, gradient descent adjusted these weights so that each unit's dot product is large for inputs that look like the pattern the unit is "looking for." In other words, each column encodes what kind of input excites that unit. That is what makes the column a *feature detector*.

### Visualizing weights as images

Here is the convenient part. An MNIST input is a 28×28 image flattened to a length-784 vector. A first-layer weight column also has 784 entries, one aligned with each input pixel. So we can take a column, reshape it back to `(28, 28)`, and display it as an image. The result is a picture of the pattern that unit responds to.

Pixels with a large positive weight light up bright in the unit's output when the corresponding input pixel is bright. Pixels with a large negative weight do the opposite — input brightness there pushes the unit's output down. This is a direct window into what the first layer learned.

The demo draws each filter with a *diverging colormap*: one color (say red) for positive weights, another (say blue) for negative weights, and white at zero. Each filter is scaled symmetrically around zero, using its own largest absolute weight, so the full color range is used no matter how big or small that filter's weights happen to be. This lets you compare the *shape* of one filter against another without one filter's magnitude washing out the rest.

### What first-layer units actually detect

With a moderate or large number of units, the learned filters often resemble strokes, edges, blobs, or partial digit templates. They detect *primitive* features — local patterns in the image — not whole digits.

It is tempting to expect each first-layer unit to detect exactly one digit class: a "3-detector," a "7-detector," and so on. That is not what happens, and it is worth saying clearly. Ten-class digit detection is the output layer's job. The first layer learns lower-level features that cut across classes; later layers combine those features into class scores. A single first-layer filter is more like "a curved stroke in the upper left" than "the digit 3."

### How H changes the filters

The number of first-layer units H is the capacity knob from task j, and it controls how the network's representational budget is spread out.

- **Few units (for example, H=4).** The network has to cover the full diversity of MNIST with only a handful of filters. Each filter is forced to be a broad, distinct pattern — something like an orientation detector or a rough template — because it has to be useful across many different inputs. The filters look visually different from one another.

- **Many units (for example, H=64).** There is no pressure for each unit to cover a lot of ground. Filters become more specialised, but also more *redundant*: many of them end up detecting slightly different versions of the same low-level feature. You will see filters that look alike repeated across the grid.

### More units, higher test accuracy

With very few units (H=4), the network is underpowered: it does not have enough filters to represent the variety in the data, and its test accuracy is noticeably lower. As H grows, accuracy rises over the range tested here. The filters are the mechanism that connects representation quality to performance — more and better filters give the later layers more to work with.

### Weight decay as a visualization aid

The demo produces two figures so you can see the same filters under two different regularization conditions.

The first figure uses a weight decay of 1e-4, which is a typical value for neural network training in practice. The second uses 1e-2, which is roughly 100× stronger than typical. To put that in context, commonly used weight decay values for training neural networks range from 1e-5 to 1e-4; a value of 1e-2 would be considered unusually aggressive in most real training runs.

With weak regularization (wd=1e-4), the first-layer filters — especially at high H — are dominated by high-frequency speckle. The network is free to fit small, idiosyncratic patterns from individual training examples, and those patterns show up as pixel-level noise in the weight images. The large-scale structure is present but hard to see through the noise.

With strong regularization (wd=1e-2), the large L2 penalty acts like a low-pass filter on the weights. It continuously penalizes small scattered weights, so the gradient updates that encode one-off example-specific patterns get suppressed. What survives is the large-scale structure that was reinforced consistently across many examples — the strokes, blobs, and orientation patterns that make up the primitive features the network actually relies on.

The tradeoff is a few percentage points of test accuracy: the strongly regularized networks are slightly less accurate because the weight decay prevents them from fitting some fine-grained structure that would help. For the purpose of this task the accuracy cost is acceptable — the goal here is to *see* the features, not to maximize performance.

It is worth being clear that using strong L2 specifically to make weights more interpretable is not a standard practice in the field. This is a pedagogical technique designed to help you see the underlying structure of what was learned. The wd=1e-4 figure is a more realistic picture of what first-layer weights look like under ordinary training conditions.


## Run the Experiment

Run the demo:

```
make demo-task-p
```

This trains six models on normalized MNIST — the same `Compose([ToArray(), Normalize()])` pipeline from task n — sweeping over H ∈ {4, 16, 64} at both weight decay values, then visualizes each model's first-layer weights. It writes two figures:

```
results/figures/learned_features_baseline.png
results/figures/learned_features_regularized.png
```

Each figure has three blocks, one per hidden layer width H ∈ {4, 16, 64}. Each block is a grid of that model's 28×28 weight filters, and its title is annotated with the value of H and that model's test accuracy. All filters use the diverging colormap described above, each scaled symmetrically to its own maximum absolute weight.

What to look for in the **baseline figure** (wd=1e-4):

- Filters at H=4 are already reasonably distinct — the small number of units forces broad coverage.
- Filters at H=64 are speckled and noisy. Structure is present but hard to read.
- This is what unregularized first-layer weights typically look like.

What to look for in the **regularized figure** (wd=1e-2):

- The same H values, but now the filters are smooth and readable.
- The **H=4** block has very few filters, and they are visually distinct from one another — broad, separate patterns.
- The larger-H blocks have many more filters, and you should see visible redundancy: groups of filters that look like near-copies of each other.
- None of the filters should look like pure random noise. Instead you should see strokes, blobs, or rough partial digit shapes.
- The test accuracy printed in each block title should rise as H increases.


## Deliverables

- Run `make demo-task-p` and open both figures.
- Study all three blocks in the regularized figure, then fill in the following keys in `answers.py`:

```python
p_observations: dict[str, bool | None]
# Keys (all start as None — mark True if you observed it, False if not):
#   first_layer_weight_columns_reshape_to_28x28
#   filters_are_not_pure_random_noise
#   some_filters_resemble_strokes_or_digit_templates
#   each_first_layer_unit_detects_exactly_one_digit_class
#   with_few_units_each_filter_is_more_distinct
#   increasing_units_makes_each_filter_a_sharper_digit
#   with_many_units_filters_look_more_redundant_or_distributed

p_filter_image_side: int
# What is the side length (in pixels) of each filter image?

p_more_units_higher_test_accuracy: bool
# Does test accuracy increase from H=4 to H=64?
```

- Run `make submit-p` to generate `submission.json` and upload it on the course webpage.
