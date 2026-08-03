# Task K — Decision Boundaries: Seeing What the Network Learned

This is the first of five observe-only tasks. You do not implement anything here. You run a provided demo, study the figure it produces, and record what you saw in `answers.py`.


## Background

A classifier takes a point and assigns it a class label. If the input is two-dimensional, then every point on the plane gets a label. The *decision boundary* is the set of points where that label changes — the line, curve, or surface that separates one labelled region from the next. On one side of the boundary the model predicts class 0; on the other side it predicts class 1. The boundary itself is the dividing edge between those regions.

Because the input here is two-dimensional, we can draw the boundary directly. The recipe is simple:

1. Lay down a dense grid of points covering the region where the data lives.
2. Run every grid point through the model and record its predicted class.
3. Colour each grid cell by that prediction. This paints the plane into coloured regions.
4. Scatter the actual training points on top, coloured by their true label.

Where two colours of background meet, you are looking at the decision boundary. Where the scattered points sit on top of the matching background colour, the model is correct.

### What shape can the boundary take?

The shape of the boundary is controlled entirely by the structure of the model.

**No hidden layer.** A model with no hidden layer is just `Linear -> softmax`. The output score for each class is a linear function of the input — a weighted sum of the two coordinates plus a bias. The place where one class overtakes another is therefore where one linear function equals another, and that is always a straight line (in higher dimensions, a flat hyperplane). A straight line can perfectly separate data that falls cleanly into two half-planes, which we call *linearly separable*. It cannot separate XOR, moons, or circles, because no single straight line splits those datasets correctly.

**One hidden layer with ReLU.** Adding a hidden layer with a ReLU activation changes the picture. Recall that ReLU outputs `max(0, x)`: it passes positive values through unchanged and clamps negatives to zero. Each ReLU unit therefore introduces a "fold" — a place where the unit switches from off (output 0) to on (output rising). When you compose these folded pieces with the linear layers around them, the overall boundary becomes *piecewise linear*: it is made of straight segments joined at kinks, like a folded sheet of paper rather than a single flat line. With enough segments, the joined-up boundary can wrap around curved or non-linearly-separable data.

**More units, more pieces.** Each hidden unit contributes one fold. More hidden units means more folds, which means more straight segments, which means the boundary can bend in more places and trace more intricate shapes. The width (units per layer) and depth (number of layers) of a network together set its *capacity* — how complex a function, and therefore how complex a boundary, it can represent.

### A subtlety: depth without non-linearity does nothing

It is tempting to think that simply stacking more `Linear` layers makes the boundary more powerful. It does not. Two `Linear` layers with no activation between them compute `W2 (W1 x + b1) + b2`, which multiplies out to `(W2 W1) x + (W2 b1 + b2)`. That is just another linear function with a combined weight matrix and a combined bias — exactly the same form as a single `Linear` layer. The boundary stays a straight line no matter how many bare `Linear` layers you stack.

The thing that buys you curved, piecewise-linear boundaries is the *non-linearity* between the layers, not the depth by itself. Remove the ReLU and a deep network collapses back to a single linear classifier.

### Width versus depth

Both of these are really routes to the same thing: more ReLU units means more folds means a more complex boundary. The question of whether it is more efficient to make one hidden layer wider or to add another hidden layer is subtler than it first appears.

Theory says that stacking layers is *exponentially* more parameter-efficient for certain function families — you can express with depth-2 and width `n` what would require width *exponential in n* with only depth-1, for the right choice of function. This is the *compositionality* argument: a second layer can fold the outputs of the first, so the overall function can combine fine-grained features into coarser structures.

But for simple 2D classification tasks like the ones in this demo, the boundary is only a 1D curve, and a wide-enough single hidden layer reaches it just fine. The compositionality advantage becomes visible in higher-dimensional settings where the data has hierarchical structure: a small image, for instance, has edges → textures → parts → objects, and each level naturally maps to a network layer. You will see this more directly in task p, where the first-layer weights of a network trained on MNIST reshape into small edge-and-stroke detectors — exactly the kind of low-level feature a second layer then combines into digit templates.

For now, the demo focuses on the two clearest points: what makes the boundary possible at all (non-linearity), and how raw fold count — regardless of whether it comes from width or depth — controls how intricate the boundary can be.

### Connection to A2

Four of the six toy datasets — `linear`, `moons`, `circles`, and `xor` — are the same ones you classified with decision trees in A2. The contrast in boundary shape is worth noticing. A decision tree splits on one feature at a time, so its boundaries are *axis-aligned rectangles* — staircases of horizontal and vertical cuts. A neural network has no such restriction: its piecewise-linear boundaries can sit at any angle and bend smoothly around the data. Same datasets, very different drawing tools.


## Run the Experiment

Run the demo:

```
make demo-task-k
```

This trains several models on six toy datasets and writes the figure:

```
results/figures/decision_boundaries.png
```

The figure is a grid. Each **column** is one of the six datasets, in this order: `linear`, `moons`, `circles`, `xor`, `checkerboard`, `spiral`. The two harder datasets at the right — `checkerboard` (4 × 4 alternating-class cells) and `spiral` (two interleaved curved arms) — expose capacity differences more clearly than the first four can on their own.

Each **row** is one model configuration, in this order:

1. **No hidden layer** (`hidden=[]`) — a plain linear classifier, `Linear -> softmax`.
2. **No activation** (`Linear -> Linear`, no ReLU between them) — same depth as row 3, but with the non-linearity removed.
3. **One hidden layer of 4 units + ReLU** (`hidden=[4]`).
4. **One hidden layer of 16 units + ReLU** (`hidden=[16]`).
5. **Two hidden layers of 64 units each + ReLU** (`hidden=[64, 64]`).

Each panel shows the coloured decision regions with the dataset scattered on top, and is titled with the configuration and that model's test accuracy.

What to look for:

- **Rows 1 & 2.** Both produce a single straight line in every panel — they look identical despite row 2 being deeper. Stacking two `Linear` layers without a ReLU in between collapses mathematically to one `Linear` layer. The non-linearity is what would have made the extra depth matter.
- **Rows 3–5 (capacity progression).** Watch the boundary become more intricate as total ReLU folds increase. The [4] row does not bend smoothly; [16] traces moons and circles cleanly but struggles on checkerboard; [64, 64] follows the spiral arms and fills in the checkerboard cells. More folds — whether from width or depth — gives the boundary more pieces to work with.
- **Rows 4 & 5 on the harder columns.** The gap between [16] and [64, 64] is widest on `checkerboard` and `spiral`. Compare the accuracy numbers: [16] will plateau before it can carve all the cells or follow the full spiral, while [64, 64] has enough total folds — and the extra layer lets those folds interact — to push further.


## Deliverables

- Run `make demo-task-k` and open `results/figures/decision_boundaries.png`.
- Study each row and column, then fill in the following keys in `answers.py`:

```python
k_observations: dict[str, bool | None]
# Keys (all start as None — mark True if you observed it, False if not):
#   linear_model_solves_linear_but_not_xor
#   linear_model_separates_moons
#   stacked_linear_no_activation_stays_straight
#   relu_boundary_is_piecewise_linear
#   relu_encloses_inner_circle
#   small_relu_underfits_checkerboard
#   all_configs_equal_accuracy_on_xor
#   large_relu_traces_spiral_arms

k_num_datasets_linear_solves: int
# How many of the six datasets does the no-hidden-layer model solve
# (test accuracy > 0.90)?

k_smallest_arch_solving_checkerboard: str
# Which is the smallest ReLU architecture that achieves accuracy > 0.75
# on the checkerboard dataset?
# One of: "[4]" / "[16]" / "[64,64]" / "none"
```

- Run `make submit-k` to generate `submission.json` and upload it on the course webpage.
