# Guide 3: Building a CNN from Scratch

This guide walks you through building your own CNN architecture from the ground up. Every parameter is explained. By the end, you'll have a working RGB model that you can then tune and iterate on.

---

## The Workflow

The assignment says:
> "Start with one model, diagnose whether overfitting or underfitting is occurring, adjust hyperparameters accordingly, and iterate until you are satisfied."

So the process is:
1. Build a simple model
2. Train it
3. Look at training vs validation curves
4. Diagnose the problem (overfitting? underfitting?)
5. Make one change
6. Repeat

Don't try to build the perfect model from scratch. Build something simple that works, then improve it.

---

## Your First Model: Simple Baseline

Start with this minimal architecture. It's intentionally simple so you can see what's happening.

```python
import keras
from keras import layers

IMAGE_SIZE = (128, 128)
NUM_CLASSES = 10  # adjust to your actual number of food classes

def build_baseline_cnn(input_shape, num_classes):
    model = keras.Sequential([
        # ---- Preprocessing ----
        layers.Rescaling(1./255, input_shape=input_shape),  # normalise [0,255] → [0,1]
        
        # ---- Block 1 ----
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # ---- Block 2 ----
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # ---- Classifier ----
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_baseline_cnn(input_shape=(128, 128, 3), num_classes=NUM_CLASSES)
model.summary()
```

### Why start simple?

- Fewer parameters = faster training = faster iteration
- Easier to diagnose: if this small model underfits, you know to add complexity
- If this small model overfits, you know regularisation is your main problem

---

## Understanding `model.summary()`

Always run `model.summary()` after building. It shows:

```
Model: "sequential"
_________________________________________________________________
 Layer (type)             Output Shape          Param #   
=================================================================
 rescaling (Rescaling)    (None, 128, 128, 3)   0         
 conv2d (Conv2D)          (None, 128, 128, 32)  896       
 max_pooling2d            (None, 64, 64, 32)    0         
 conv2d_1 (Conv2D)        (None, 64, 64, 64)    18496     
 max_pooling2d_1          (None, 32, 32, 64)    0         
 flatten (Flatten)        (None, 65536)          0         
 dense (Dense)            (None, 128)           8388736   
 dense_1 (Dense)          (None, 10)            1290      
=================================================================
Total params: 8,409,418
```

**Reading the output shapes:**
- `(None, 128, 128, 32)` — None is the batch size (flexible), then height × width × channels
- After `MaxPooling2D(2,2)`: 128→64 (halved), 64→32, etc.
- After `Flatten`: 32×32×64 = 65,536 values become a 1D vector
- **Total params** is what you report in your paper — record this for every model

**Where do the parameter counts come from?**

For `Conv2D(32, (3,3))` applied to a 3-channel input:
- Each filter: 3×3×3 weights + 1 bias = 28 values
- 32 filters: 32 × 28 = **896 parameters**

For `Dense(128)` applied to 65,536 inputs:
- 65,536 × 128 weights + 128 biases = **8,388,736 parameters**

Notice: the dense layer dominates. This is why people use Global Average Pooling (covered later) to reduce parameters before the dense layer.

---

## Compiling the Model

Before training, you must `compile` the model — this sets the loss function, optimiser, and metrics:

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

**`optimizer`:** Adam with learning rate 0.001 is a reliable default. Don't change this yet.

**`loss='sparse_categorical_crossentropy'`:** Use this when your labels are integers (0, 1, 2...). If you one-hot encode labels, use `'categorical_crossentropy'` instead. `image_dataset_from_directory` with `label_mode='int'` gives integer labels, so use `sparse_categorical_crossentropy`.

**`metrics=['accuracy']`:** Report accuracy during training. You'll add more detailed metrics (precision, recall, F1) separately after training.

---

## Training the Model

```python
EPOCHS = 20  # start with 20, increase if the model is still improving

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights  # from Guide 2 — omit if classes are balanced
)
```

`history` contains a record of loss and accuracy for every epoch — you'll use this to plot training curves.

### What happens during one epoch?

1. The model sees every image in `train_ds` once (in random batches)
2. For each batch: forward pass → loss calculation → backpropagation → weight update
3. At the end of the epoch: evaluate on `val_ds` (no weight updates, just measurement)
4. Report training loss/accuracy and validation loss/accuracy

---

## Plotting Training Curves — Your Most Important Diagnostic Tool

After training, always plot this:

```python
import matplotlib.pyplot as plt

def plot_history(history, title='Training History'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_history(history, title='Baseline CNN')
```

---

## Diagnosing What the Curves Tell You

### Underfitting

```
Train accuracy:      55%
Validation accuracy: 53%
```

Both are low, and they're close together. The model isn't learning enough.

**Causes & fixes:**
- Model is too simple → add more Conv2D layers or more filters
- Not enough epochs → train longer
- Learning rate too low → increase slightly (try 0.003)
- Images too small → use larger images

### Overfitting

```
Train accuracy:      95%
Validation accuracy: 62%
```

Training accuracy is much higher than validation. The model has memorised the training data but doesn't generalise.

**Causes & fixes:**
- Model too complex for the data → reduce filters, remove a layer
- Not enough data → use data augmentation (Guide 4)
- No regularisation → add Dropout (Guide 4)
- Too many epochs → use early stopping (Guide 4)

### Good fit

```
Train accuracy:      80%
Validation accuracy: 77%
```

Both are high and close together. A small gap is normal and expected. This is what you're aiming for.

---

## Building Up: A More Capable Model

Once you've confirmed the baseline works (even if it overfits), build a deeper version:

```python
def build_deeper_cnn(input_shape, num_classes):
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),  # second conv in same block
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

**What changed and why:**
- Two conv layers per block instead of one → network can learn more complex features at each resolution
- Three blocks instead of two → more levels of abstraction (edges → textures → food shapes)
- 32 → 64 → 128 filters → progressively more patterns as images get smaller
- Dropout(0.5) → fight overfitting in the dense layer

### Network depth — does it help?

The assignment specifically asks about this. Compare:
- Shallow model (2 conv blocks) vs deep model (3+ conv blocks)
- Look at validation accuracy and validation loss
- Check if the deeper model overfits more (it likely will — that's okay, that's information)
- Discuss in your paper: did depth improve accuracy? At what cost?

---

## Using Global Average Pooling Instead of Flatten

An alternative to Flatten + Dense is **Global Average Pooling (GAP)**:

```python
layers.GlobalAveragePooling2D(),  # replaces Flatten()
layers.Dense(num_classes, activation='softmax')
```

Instead of flattening all values, GAP averages each feature map into a single number. If you have 128 filters, GAP gives you a vector of 128 values regardless of the input image size.

**Advantages:**
- Massively reduces parameters (no giant Dense layer)
- Less overfitting
- Works with any input image size

**Trade-off:** slightly less expressive than Flatten + Dense, but usually performs comparably or better.

This is worth trying and mentioning in your paper.

---

## Recording Results

For every model you train, record these numbers in a table. You'll need this for your paper.

| Model | Image Size | Params | Train Acc | Val Acc | Train Loss | Val Loss | Training Time |
|---|---|---|---|---|---|---|---|
| Baseline CNN (RGB) | 128×128 | 8.4M | 92% | 61% | 0.25 | 1.2 | 4min |
| Deeper CNN (RGB) | 128×128 | 15M | 89% | 70% | 0.3 | 0.9 | 7min |
| ... | | | | | | | |

Training time: wrap your `model.fit()` call with:
```python
import time
start = time.time()
history = model.fit(...)
training_time = time.time() - start
print(f"Training time: {training_time:.1f} seconds")
```

---

## Building the Greyscale Version

The assignment requires a greyscale model **with the same hyperparameters** as your best RGB model (for a fair comparison).

```python
# Load greyscale data
train_ds_grey = keras.utils.image_dataset_from_directory(
    'path/to/food/',
    color_mode='grayscale',   # <-- this is the only change in data loading
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.3,
    subset='training'
)

# Model changes: input shape is (128, 128, 1) instead of (128, 128, 3)
model_grey = build_deeper_cnn(input_shape=(128, 128, 1), num_classes=NUM_CLASSES)
```

Keep everything else identical: same number of layers, same filters, same dropout rate, same learning rate, same number of epochs. The only difference should be the input channels (1 vs 3).

**What to expect and discuss:**
- RGB has 3× more input information; does it actually help?
- Food classification is colour-dependent (strawberries are red, avocados are green) — so RGB should help here
- If greyscale performs similarly, that's an interesting finding worth discussing

---

## Checklist for This Guide

- [ ] Build and train a simple baseline CNN
- [ ] Run `model.summary()` and record the parameter count
- [ ] Plot training and validation curves
- [ ] Diagnose: is it overfitting or underfitting?
- [ ] Build a deeper CNN and compare
- [ ] Compare at least two image sizes
- [ ] Record all results in a table
- [ ] Build the greyscale version with the same hyperparameters
- [ ] Compare RGB vs greyscale results

---

## Next Step

→ [Guide 4: Overfitting and Regularisation](04_overfitting_and_regularisation.md)
