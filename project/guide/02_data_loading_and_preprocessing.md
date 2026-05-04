# Guide 2: Data Loading and Preprocessing

This guide covers everything you need to do *before* building a model: loading the Food dataset, creating proper train/validation/test splits, understanding class imbalance, and normalising the data.

---

## The Food Dataset

Your dataset is the Food-101 subset, combined from `food1.zip` and `food2.zip`. Images are:
- **512×512 pixels**
- **Colour (RGB)** — 3 channels
- Organised into folders by class (e.g., `pizza/`, `sushi/`, `burger/`)

The folder structure looks like this:
```
food/
├── burger/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── pizza/
│   ├── img001.jpg
│   └── ...
└── sushi/
    └── ...
```

Keras can read this structure directly — it infers the class names from the folder names.

---

## Train / Validation / Test Split

The assignment requires **three separate sets**:

| Set | Purpose | Suggested size |
|---|---|---|
| **Training** | The model learns from this | 70% |
| **Validation** | You use this to tune hyperparameters and pick the best model | 10% |
| **Test** | Used *once only* for the final reported result | 20% |

**Critical rule:** The test set must never influence any decisions. You do not look at it, tune based on it, or use it to select models. Only evaluate on it once, right at the very end, with your chosen model.

---

## Loading the Data with Keras

The assignment tells you to use `image_dataset_from_directory`. This is the correct approach for a dataset of this size — it loads images in batches from disk rather than loading everything into memory at once.

```python
import tensorflow as tf
import keras

# Your student ID number becomes the random seed
SEED = 123456  # replace with YOUR student ID digits (no G, no leading zeros)
IMAGE_SIZE = (128, 128)  # we'll discuss image size choice in Guide 3
BATCH_SIZE = 32

# Load training set (70%)
train_ds = keras.utils.image_dataset_from_directory(
    'path/to/food/',          # path to your combined food folder
    labels='inferred',        # infer class labels from folder names
    label_mode='int',         # labels as integers (0, 1, 2, ...)
    image_size=IMAGE_SIZE,    # resize all images to this size
    batch_size=BATCH_SIZE,
    shuffle=True,             # shuffle the training data
    seed=SEED,                # MUST be your student ID
    validation_split=0.3,     # hold out 30% for val+test
    subset='training'         # this 70% is training
)

# Load the remaining 30% — we'll split this into val and test
remaining_ds = keras.utils.image_dataset_from_directory(
    'path/to/food/',
    labels='inferred',
    label_mode='int',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.3,
    subset='validation'       # this is the held-out 30%
)
```

Because `image_dataset_from_directory` only gives you training/validation splits (not three-way), you need to manually split the remaining 30% into validation and test:

```python
# Count how many batches are in the remaining 30%
remaining_batches = tf.data.experimental.cardinality(remaining_ds).numpy()
val_batches = int(remaining_batches * 0.33)  # ~10% of total
test_batches = remaining_batches - val_batches  # ~20% of total

val_ds = remaining_ds.take(val_batches)
test_ds = remaining_ds.skip(val_batches)
```

### What `image_dataset_from_directory` gives you

Each item in the dataset is a tuple: `(image_batch, label_batch)`.
- `image_batch` shape: `(32, 128, 128, 3)` — 32 images, 128×128, 3 colour channels
- `label_batch` shape: `(32,)` — 32 integer labels

You can inspect the class names:
```python
class_names = train_ds.class_names
print(class_names)  # ['burger', 'hot_dog', 'pizza', ...]
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
```

---

## The Seed Requirement — Critical

> **"Any random seeds must use your student ID number (without the G and leading zeros)."**

This applies to **every** place a random seed is used:
- `image_dataset_from_directory(..., seed=SEED)`
- `tf.random.set_seed(SEED)` at the top of your notebook
- `numpy` random seeds if used

If your ID is G00432121, your seed is `432121`.

Define it once at the top of your notebook:
```python
SEED = 432121  # your student ID

import tensorflow as tf
import numpy as np
tf.random.set_seed(SEED)
np.random.seed(SEED)
```

---

## Normalisation

Images come as pixel values from 0 to 255. Neural networks train much better with small values close to 0. You should normalise to the range [0, 1].

```python
# Add this as a layer inside your model, OR as a preprocessing step on the dataset

# Option 1: As a Keras layer inside the model (recommended — runs on GPU if available)
normalization_layer = keras.layers.Rescaling(1./255)

# Option 2: As a dataset map operation
train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
```

Option 1 (as a layer) is cleaner and ensures the same normalisation always happens regardless of how the model is called.

---

## Performance: Prefetching

Large datasets can become a bottleneck where the GPU is idle waiting for data to load. Fix this with `prefetch`:

```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
```

`AUTOTUNE` lets TensorFlow automatically decide how many batches to prefetch. This makes training noticeably faster.

---

## Class Imbalance

**Class imbalance** happens when some categories have far more images than others. For example, if "pizza" has 2000 images but "sushi" has 200, the model can achieve 90% accuracy by just always predicting "pizza" — even though it's completely useless for other classes.

### Step 1: Check the class distribution

Before doing anything else, count images per class:

```python
import os
import matplotlib.pyplot as plt

data_dir = 'path/to/food/'
class_counts = {}

for class_name in sorted(os.listdir(data_dir)):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        class_counts[class_name] = count

# Plot it
plt.figure(figsize=(12, 5))
plt.bar(class_counts.keys(), class_counts.values())
plt.xticks(rotation=45, ha='right')
plt.title('Images per Class')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

for name, count in class_counts.items():
    print(f"{name}: {count}")
```

Include this plot in your paper — it's your "exploratory data analysis".

### Step 2: Compute class weights

If the classes are imbalanced, you need to tell Keras to penalise mistakes on minority classes more heavily. You do this with **class weights**.

```python
import numpy as np

total_images = sum(class_counts.values())
num_classes = len(class_counts)

class_weights = {}
for i, class_name in enumerate(sorted(class_counts.keys())):
    count = class_counts[class_name]
    # Classes with fewer images get higher weight
    weight = total_images / (num_classes * count)
    class_weights[i] = weight

print(class_weights)
```

Then pass it to `model.fit`:
```python
model.fit(train_ds, ..., class_weight=class_weights)
```

### Why class weights matter

Without class weights: the model learns to favour majority classes. Accuracy looks high but minority class performance is terrible.

With class weights: mistakes on rare classes are penalised proportionally more, forcing the model to learn all classes equally well.

**In your paper:** report whether the dataset is balanced or imbalanced, what you did about it, and whether it affected per-class performance (look at the per-class F1 scores).

---

## Exploratory Data Analysis

Before training, always look at your data. This goes in your paper under "Data and Preprocessing".

```python
import matplotlib.pyplot as plt

# Show sample images from the training set
plt.figure(figsize=(12, 8))
for images, labels in train_ds.take(1):  # take one batch
    for i in range(min(12, len(images))):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
plt.suptitle('Sample Training Images')
plt.tight_layout()
plt.show()
```

Also report:
- Number of classes
- Total number of images
- Image size and colour mode
- Class distribution (balanced or imbalanced?)
- Train/val/test split sizes

---

## Image Size Choice

The assignment says to **compare at least two image sizes** (or justify choosing one). This matters because:
- **Larger images** (e.g., 224×224): more detail, more accurate, but slower to train and needs more memory
- **Smaller images** (e.g., 64×64): faster, uses less memory, but loses fine detail

Typical choices for this project:
- `64×64` — very fast, but may lose too much detail for food
- `128×128` — good balance for scratch models
- `224×224` — required for VGG16 (transfer learning); also good for scratch

**Important for transfer learning:** VGG16 expects input of at least 32×32, and was trained on 224×224. Using 224×224 for transfer learning is strongly recommended.

To compare image sizes, train the same model architecture twice with different `IMAGE_SIZE` values and compare validation accuracy. Document the trade-off.

---

## Summary: Preprocessing Checklist

- [ X ] Combine food1 and food2 folders
- [ X ] Set `SEED` to your student ID at the top of the notebook
- [ X ] Load dataset with `image_dataset_from_directory`
- [ X ] Split into train (70%), val (10%), test (20%)
- [ ] Add `.prefetch(AUTOTUNE)` to all three datasets
- [ X ] Normalise pixel values to [0, 1]
- [ X ] Count images per class and plot the distribution
- [ X ] Compute and apply class weights if imbalanced
- [ X ] Display sample images from the training set
- [ ] Document image sizes you'll compare

---

## Next Step

→ [Guide 3: Building a CNN from Scratch](03_building_cnn_from_scratch.md)
