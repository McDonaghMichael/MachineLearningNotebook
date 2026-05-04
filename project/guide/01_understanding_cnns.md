# Guide 1: Understanding Convolutional Neural Networks (CNNs)

Before writing a single line of code, you need to understand *what* you're building and *why* each part exists. This guide covers the foundations.

---

## What Problem Are We Solving?

You have a dataset of food images (512×512 pixels, colour). You want a program that looks at an image and outputs a label like "pizza" or "sushi". This is called **image classification**.

The challenge: a pixel is just a number (0–255). A 512×512 RGB image is a grid of 512×512×3 = **786,432 numbers**. How do you go from those numbers to a meaningful label?

A traditional approach (passing all pixels into a regular neural network) doesn't work well because:
- It ignores *spatial structure* — nearby pixels are related, but a flat list loses that
- It would have hundreds of millions of parameters and overfit immediately
- It doesn't generalise — a cat in the top-left corner looks different to the model than a cat in the bottom-right

CNNs solve all three problems.

---

## The Core Idea: Convolution

A **convolutional layer** slides a small window (called a **filter** or **kernel**) across the image. The filter is a small grid of numbers (e.g., 3×3) that looks for a specific pattern — an edge, a curve, a colour gradient.

### How one convolution step works:

Imagine a 3×3 filter positioned over a 3×3 patch of the image:

```
Image patch:      Filter (learned):    Output value:
1  2  3           0.1  0.2  0.1        sum of element-wise
4  5  6    ×      0.2  0.4  0.2   =    products = one number
7  8  9           0.1  0.2  0.1
```

The filter slides across the entire image, producing one output number per position. This output grid is called a **feature map**.

**The key insight:** the filter weights are *learned* during training. The network figures out what patterns are useful for the task. Early layers tend to learn edges and colours; deeper layers learn textures and shapes; the deepest layers learn abstract concepts like "this looks like food".

---

## Anatomy of a CNN

A typical CNN has this structure:

```
Input Image
    ↓
[Conv Layer] → finds low-level features (edges, colours)
[Activation Function (ReLU)]
[Pooling Layer] → shrinks the feature map, keeps important info
    ↓
[Conv Layer] → finds mid-level features (textures, shapes)
[Activation Function (ReLU)]
[Pooling Layer]
    ↓
[Conv Layer] → finds high-level features (food-specific patterns)
[Activation Function (ReLU)]
[Pooling Layer]
    ↓
[Flatten] → convert 3D output to 1D list
[Dense (Fully Connected) Layer] → combine all features
[Dropout] → randomly zero out some neurons (prevents overfitting)
[Dense Output Layer] → one output per class, e.g. 10 food types
[Softmax] → convert outputs to probabilities that sum to 1
    ↓
Predicted class
```

---

## Each Layer Explained

### Convolutional Layer (`Conv2D`)

```python
keras.layers.Conv2D(
    filters=32,         # How many different patterns to look for
    kernel_size=(3, 3), # Size of the sliding window
    padding='same',     # Whether to keep output the same size as input
    activation='relu'   # Activation function applied after convolution
)
```

**`filters` (most important parameter):**
- Each filter looks for one type of pattern
- `filters=32` means 32 different patterns are detected in this layer
- Output shape: (height, width, 32) — one feature map per filter
- More filters = more expressive, but more parameters and slower to train
- Typical progression: 32 → 64 → 128 as you go deeper

**`kernel_size`:**
- `(3, 3)` is the standard — small enough to be efficient, large enough to capture local patterns
- `(5, 5)` captures slightly wider patterns but costs more computation
- Stick with `(3, 3)` throughout your project

**`padding='same'`:**
- Without padding, each conv shrinks the spatial size slightly
- `padding='same'` adds zeros around the border so output size = input size
- This means pooling layers control the size reduction, not conv layers

**`padding='valid'`:**
- No padding — output is slightly smaller than input
- Less common in simple architectures

---

### Activation Function: ReLU

After the convolution, you apply an **activation function**. The most common is **ReLU** (Rectified Linear Unit):

```
ReLU(x) = max(0, x)
```

If the output of a convolution is negative, ReLU sets it to zero. If it's positive, it stays as-is.

**Why do we need it?** Without activation functions, stacking multiple layers is mathematically equivalent to just one layer — you can only model linear relationships. ReLU introduces *non-linearity*, letting the network learn complex patterns.

You'll often see it written as part of the Conv2D call:
```python
Conv2D(32, (3,3), activation='relu', padding='same')
```

Or as a separate layer:
```python
Conv2D(32, (3,3), padding='same'),
keras.layers.Activation('relu'),
```

They're equivalent — use whichever is clearer.

---

### Pooling Layer (`MaxPooling2D`)

```python
keras.layers.MaxPooling2D(pool_size=(2, 2))
```

Pooling **shrinks** the feature map. MaxPooling takes the maximum value in each 2×2 window and discards the rest.

**Before pooling:** feature map is 64×64
**After MaxPooling(2,2):** feature map is 32×32

**Why do this?**
1. Reduces computation (fewer numbers to process in the next layer)
2. Builds in *spatial invariance* — the model becomes less sensitive to exactly where a feature appears
3. Prevents the network from just memorising pixel positions

**Alternative:** `AveragePooling2D` takes the average instead of the max. MaxPooling is generally preferred for classification.

---

### Flatten Layer

```python
keras.layers.Flatten()
```

After several conv+pool blocks, you have a 3D tensor (height × width × channels). `Flatten` reshapes it into a 1D vector so it can be fed into Dense layers.

Example: if the last feature map is 8×8×128, Flatten gives you a vector of 8×8×128 = **8192 values**.

---

### Dense (Fully Connected) Layer

```python
keras.layers.Dense(128, activation='relu')
```

Every neuron in this layer connects to every value in the previous layer. This layer combines all the spatial features detected by the conv layers into a final decision.

**`units=128`:** how many neurons in this layer. Common values: 64, 128, 256, 512.

---

### Dropout

```python
keras.layers.Dropout(0.5)
```

During training, Dropout **randomly sets 50% of the layer's outputs to zero** on each forward pass.

**Why?** It forces the network to not rely on any single neuron, making it learn more robust, distributed representations. This is a powerful technique to fight overfitting.

- `0.5` means 50% dropout (very common for dense layers)
- `0.25` or `0.3` is common for convolutional layers
- Dropout is **disabled during evaluation/prediction** — all neurons are active

---

### Output Layer + Softmax

```python
keras.layers.Dense(num_classes, activation='softmax')
```

The final layer has one neuron per class (e.g., 10 food categories = 10 neurons).

**Softmax** converts the raw scores into probabilities:
- All outputs are between 0 and 1
- They all sum to exactly 1
- The highest value is the model's prediction

Example: `[0.02, 0.05, 0.85, 0.03, 0.05]` → the model is 85% confident it's class 3.

---

## How the Network Learns: Backpropagation

Training a CNN means finding the right filter weights. Here's the process:

1. **Forward pass:** image goes through the network, producing a prediction
2. **Loss calculation:** compare prediction to the true label using a **loss function** (more below)
3. **Backward pass (backpropagation):** calculate how much each weight contributed to the error
4. **Weight update:** adjust each weight slightly in the direction that reduces the error

This repeats for every batch of images, for many **epochs** (passes through the full dataset).

---

## Loss Function

For multi-class classification:

```python
model.compile(
    loss='sparse_categorical_crossentropy',  # when labels are integers (0, 1, 2...)
    # OR
    loss='categorical_crossentropy',          # when labels are one-hot encoded
)
```

**Categorical cross-entropy** measures how far the predicted probability distribution is from the true one. If the true class is "pizza" (class 3) and the model gives it 85% probability, the loss is low. If it gives it 5% probability, the loss is high.

---

## Optimiser

The optimiser decides *how* to update the weights based on the gradients from backpropagation.

```python
model.compile(
    optimizer='adam',  # or keras.optimizers.Adam(learning_rate=0.001)
)
```

**Adam** is the standard choice — it adapts the learning rate for each parameter automatically and works well in most cases.

**Learning rate:** how big each weight update step is
- Too high → training is unstable, loss oscillates or diverges
- Too low → training is very slow
- `0.001` (Adam's default) is a safe starting point

---

## Summary: What You Need to Know for This Project

| Concept | What it does | Key parameter |
|---|---|---|
| Conv2D | Detect patterns in images | `filters`, `kernel_size` |
| ReLU | Add non-linearity | none |
| MaxPooling2D | Shrink feature maps | `pool_size` |
| Flatten | Convert 3D to 1D | none |
| Dense | Combine features into a decision | `units` |
| Dropout | Prevent overfitting | rate (0.0–0.5) |
| Softmax | Convert scores to probabilities | none |
| Adam | Update weights during training | `learning_rate` |
| Cross-entropy loss | Measure prediction error | none |

---

## Next Step

→ [Guide 2: Data Loading and Preprocessing](02_data_loading_and_preprocessing.md)
