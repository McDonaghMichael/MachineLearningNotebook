# Guide 4: Overfitting and Regularisation

Overfitting is almost guaranteed to happen when you train a CNN from scratch. This guide explains what it is, why it happens, and gives you four tools to fight it.

---

## What Is Overfitting?

During training, the model sees the same images many times. If the model is complex enough, it can simply **memorise** the training images rather than learning the general patterns that make a pizza look like a pizza.

**Signs of overfitting in your training curves:**
```
Epoch 1:   train_loss=1.8,  val_loss=1.9  (normal — both high)
Epoch 10:  train_loss=0.5,  val_loss=0.8  (both improving — good)
Epoch 20:  train_loss=0.1,  val_loss=1.2  (train keeps dropping, val gets worse — overfitting)
```

The model has memorised the training set. It does great on training data, but performs much worse on images it hasn't seen (validation set).

**The gap between train and val accuracy is your overfitting indicator.**

---

## Tool 1: Data Augmentation

**The core idea:** if you only have 1000 photos of burgers, create new training images by applying random transformations. A horizontally flipped burger is still a burger. This gives the model more variety to learn from.

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),        # flip left-right with 50% probability
    layers.RandomRotation(0.1),             # rotate by up to ±10% of 360° = ±36°
    layers.RandomZoom(0.1),                 # zoom in/out by up to 10%
    layers.RandomTranslation(0.1, 0.1),     # shift up/down/left/right by up to 10%
    layers.RandomContrast(0.1),             # randomly adjust contrast
], name='data_augmentation')
```

**Add it to your model as the first layers:**

```python
def build_cnn_with_augmentation(input_shape, num_classes):
    model = keras.Sequential([
        # Augmentation (only active during training, automatically disabled during evaluation)
        layers.RandomFlip('horizontal', input_shape=input_shape),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        # Normalisation
        layers.Rescaling(1./255),
        
        # Conv blocks
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
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

### Each augmentation parameter explained

**`RandomFlip('horizontal')`**
- Flips images left-to-right with 50% probability
- A burger looks the same flipped horizontally — valid augmentation
- Don't use `'vertical'` for food — an upside-down pizza is unusual and could confuse the model

**`RandomRotation(factor)`**
- `factor=0.1` means rotate by up to 10% of a full rotation = ±36°
- `factor=0.2` = ±72°
- Keep it moderate for food images (slight tilt is realistic; 90° rotation is not)

**`RandomZoom(factor)`**
- `factor=0.1` zooms in or out by up to 10%
- Simulates the food being closer or further from the camera

**`RandomTranslation(height_factor, width_factor)`**
- Shifts the image up/down and left/right
- Good for training the model not to rely on the subject being centred

**`RandomContrast(factor)`**
- Simulates different lighting conditions
- Food photos often vary a lot in lighting — this helps

### Important: augmentation is only active during training

Keras augmentation layers are automatically in "training mode" during `model.fit()` and disabled during `model.predict()` and `model.evaluate()`. You don't need to do anything special — it's handled automatically.

---

## Tool 2: Dropout

Covered in Guide 1, but more detail here on where and how much.

```python
layers.Dropout(rate)
```

**The rate parameter:**
- `0.0` — no dropout (off)
- `0.25` — drop 25% of neurons (mild)
- `0.5` — drop 50% (standard for Dense layers)
- `0.3` — common for Conv layers

**Where to add dropout:**

```python
# After conv blocks (lighter dropout — don't destroy spatial features too much)
layers.Conv2D(128, (3,3), padding='same', activation='relu'),
layers.MaxPooling2D((2,2)),
layers.Dropout(0.25),  # ← after pooling, before next block

# After Dense layers (heavier dropout)
layers.Dense(256, activation='relu'),
layers.Dropout(0.5),  # ← standard here
layers.Dense(num_classes, activation='softmax')
```

**Why not use 0.9?** If you drop 90% of neurons, the network effectively has almost no capacity left — it will underfit. There's a balance. Start with 0.5 on Dense layers and 0.25 on Conv blocks.

**What to document:** train a model without dropout, then with dropout. Show the training curves. Does the gap between train and val accuracy narrow? It should.

---

## Tool 3: L2 Regularisation (Weight Decay)

Dropout randomly drops neurons. L2 regularisation takes a different approach: it adds a **penalty** to the loss function for having large weights.

```python
from keras import regularizers

layers.Conv2D(64, (3,3), padding='same', activation='relu',
              kernel_regularizer=regularizers.l2(0.001))

layers.Dense(256, activation='relu',
             kernel_regularizer=regularizers.l2(0.001))
```

**How it works:** the loss function becomes:
```
total_loss = prediction_loss + λ × sum(weights²)
```

Where λ (lambda) = 0.001 in the example above. The model is penalised for having large weights, which discourages it from "memorising" specific training examples.

**`l2(0.001)`** — the lambda value
- Too small (0.0001): barely any effect
- Too large (0.1): weights are forced very small, model underfits
- 0.001 or 0.0001 are common starting points

**L2 vs Dropout:** they're both regularisation but work differently. In practice, Dropout is more commonly used in CNNs. You don't need to use both — pick the one that helps more. But it's fine to try both and compare.

---

## Tool 4: Early Stopping

Early stopping monitors the validation loss during training and **stops training automatically** when it stops improving.

```python
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',    # watch validation loss
    patience=5,            # stop if no improvement for 5 consecutive epochs
    restore_best_weights=True  # revert to the weights from the best epoch
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,            # set a high upper bound — early stopping will stop it
    callbacks=[early_stopping],
    class_weight=class_weights
)

print(f"Training stopped at epoch {len(history.history['val_loss'])}")
```

### Parameters explained

**`monitor='val_loss'`**
- Watch validation loss (not training loss, not accuracy)
- Validation loss is the most reliable signal of real performance
- You could also use `'val_accuracy'` but loss is more sensitive to small changes

**`patience=5`**
- If val_loss doesn't improve for 5 epochs in a row, stop training
- Low patience (3) → might stop too early if there's a temporary plateau
- High patience (10) → waits longer, might still overfit
- 5 is a reasonable default for this project

**`restore_best_weights=True`**
- Without this: when training stops, you have the weights from the last epoch (which may be overfit)
- With this: you get the weights from the epoch where val_loss was lowest — always use this

### ModelCheckpoint: save the best model

Early stopping with `restore_best_weights=True` is usually enough. But you can also save the model to disk:

```python
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True  # only saves when val_loss improves
)

history = model.fit(..., callbacks=[early_stopping, checkpoint])

# Later, load the best model:
best_model = keras.models.load_model('best_model.keras')
```

---

## Combining All Four Tools

A production-ready CNN with all regularisation techniques:

```python
def build_regularised_cnn(input_shape, num_classes):
    model = keras.Sequential([
        # Augmentation
        layers.RandomFlip('horizontal', input_shape=input_shape),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        # Normalisation
        layers.Rescaling(1./255),
        
        # Block 1
        layers.Conv2D(32, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

---

## How to Document Overfitting Work in Your Paper

The assignment expects you to show evidence of iterative improvement. A good structure:

1. **Show the first model's curves** — identify overfitting ("Training accuracy reached 95% while validation accuracy plateaued at 62%, indicating overfitting")
2. **Describe what you added** — "Data augmentation and dropout (0.5) were applied..."
3. **Show the new curves** — "After regularisation, the gap between training and validation accuracy reduced from 33% to 12%..."
4. **Interpret** — "While the training accuracy decreased slightly, validation accuracy improved, indicating better generalisation"

Don't just report numbers — interpret them. Show you understand what happened and why.

---

## What to Expect

After applying regularisation, you'll typically see:
- Training accuracy drops (model can no longer memorise)
- Validation accuracy improves (model generalises better)
- The gap between train and val narrows
- Training curves are smoother and less erratic

This is success, even if training accuracy went down.

---

## Checklist for This Guide

- [ ] Identify overfitting in your baseline model's curves
- [ ] Add data augmentation and retrain — compare curves
- [ ] Add dropout and retrain — compare curves
- [ ] Try early stopping on your best model
- [ ] Record all results in your comparison table
- [ ] Explain in paper: what changed, why, what effect it had

---

## Next Step

→ [Guide 5: Transfer Learning](05_transfer_learning.md)
