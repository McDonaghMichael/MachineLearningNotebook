# Guide 5: Transfer Learning and Fine-Tuning

Transfer learning is often the most powerful technique in this project — a pre-trained model brings in knowledge learned from millions of images that your scratch CNN never had. This guide explains how it works, how to implement it, and how to fine-tune it.

---

## What Is Transfer Learning?

Training a CNN from scratch teaches it to detect features from random initial weights. This takes a lot of data and many epochs.

**Transfer learning** takes a model that was already trained on a large dataset (ImageNet — 1.2 million images, 1000 categories) and reuses the features it learned for your specific task.

The intuition: low-level features (edges, textures, colour gradients) are universal. A model trained on dogs and cars has already learned to detect edges and textures. Those are exactly what you need to detect food too. You're just adding a new "head" on top that maps those features to your food categories.

### The two phases

**Phase 1 — Feature extraction (frozen base):**
- Load VGG16/ResNet50/MobileNetV2 with ImageNet weights
- Freeze all the base model layers (don't update them)
- Add your own Dense classifier layers on top
- Train only the new classifier layers
- The base model acts as a fixed feature extractor

**Phase 2 — Fine-tuning (unfreeze top layers):**
- Unfreeze the top few layers of the base model
- Use a very small learning rate (e.g., 1e-5)
- Train for at least one epoch
- The model's high-level features adapt to your specific dataset

---

## Choosing Your Base Model

You must use one of: **VGG16**, **ResNet50**, or **MobileNetV2**.

| Model | Parameters | Speed | Accuracy | Notes |
|---|---|---|---|---|
| VGG16 | ~138M | Slow | Good | Simple architecture, easy to understand |
| ResNet50 | ~25M | Medium | Very good | Uses "skip connections" — skip connections allow gradients to flow through deeply |
| MobileNetV2 | ~3.4M | Fast | Good | Designed for mobile/embedded — very efficient |

**Recommendation for this project:** MobileNetV2 or ResNet50. VGG16 is enormous (138M parameters) and slow. MobileNetV2 trains quickly and performs well. ResNet50 is a good middle ground.

All three expect **RGB images** (3 channels) — this is why the assignment says there's no point doing greyscale with transfer learning.

**Minimum input size for all three:** 32×32, but they were trained on 224×224. Using 224×224 gives best results.

---

## Phase 1: Building the Transfer Learning Model

```python
import keras
from keras import layers
from keras.applications import MobileNetV2  # or VGG16, ResNet50

IMAGE_SIZE = (224, 224)  # match what the base model was trained on

# Step 1: Load the pre-trained base model
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,     # don't include the original ImageNet classifier
    weights='imagenet'     # use weights pre-trained on ImageNet
)

# Step 2: Freeze the base model — don't update its weights
base_model.trainable = False

print(f"Base model parameters: {base_model.count_params():,}")
print(f"Trainable: {sum([tf.size(v).numpy() for v in base_model.trainable_variables])}")
```

**`include_top=False`:** The pre-trained model has a final Dense layer for 1000 ImageNet classes. We remove that ("top") and replace it with our own classifier for food categories.

**`weights='imagenet'`:** Download and use the weights learned from ImageNet. The first time this runs, Keras downloads the weights from the internet (a few hundred MB). After that, they're cached locally.

**`base_model.trainable = False`:** Freeze all layers. Setting this to False on the model object freezes all child layers. Their weights will not be updated during training.

```python
# Step 3: Add preprocessing specific to the base model
# Each model has its own expected input range — use the built-in preprocessor
from keras.applications.mobilenet_v2 import preprocess_input  # use the right one for your model
# For VGG16: from keras.applications.vgg16 import preprocess_input
# For ResNet50: from keras.applications.resnet50 import preprocess_input

# Step 4: Build the full model
def build_transfer_model(base_model, num_classes):
    inputs = keras.Input(shape=(224, 224, 3))
    
    # Preprocessing: scale to the expected range for this model
    x = preprocess_input(inputs)   # MobileNetV2 expects [-1, 1], ResNet/VGG expect different ranges
    
    # Base model (frozen)
    x = base_model(x, training=False)  # training=False keeps BatchNorm in inference mode
    
    # Our new classifier head
    x = layers.GlobalAveragePooling2D()(x)   # convert spatial features to a vector
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

model_tl = build_transfer_model(base_model, NUM_CLASSES)
model_tl.summary()
```

**`GlobalAveragePooling2D()`:** The base model's output is a 3D tensor (e.g., 7×7×1280 for MobileNetV2). GAP averages each feature map to produce a 1D vector of 1280 values. This is the standard approach for adding a custom head to a pre-trained model.

**`training=False` in `base_model(x, training=False)`:** Some layers (like BatchNormalisation, which is used in MobileNetV2 and ResNet50) behave differently during training vs inference. Setting `training=False` ensures they use their pre-trained statistics and don't try to update — which is what you want when the base model is frozen.

### Compile and train Phase 1

```python
model_tl.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # normal learning rate for the new head
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

import time
start = time.time()
history_tl = model_tl.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stopping],
    class_weight=class_weights
)
training_time_tl = time.time() - start
```

Because most of the model is frozen, Phase 1 trains much faster than a full scratch model. Only the small Dense head needs updating.

---

## Phase 2: Fine-Tuning

After Phase 1, the classifier head is trained. Now you can unfreeze the top layers of the base model and fine-tune them.

**Why only the top layers?** The early layers of the base model learned generic features (edges, textures) that don't need to change. The top layers learned more abstract ImageNet-specific patterns that could benefit from adjusting to food images.

```python
# Step 1: Unfreeze the top layers of the base model
base_model.trainable = True  # unfreeze everything

# Find out how many layers the base model has
print(f"Total layers in base model: {len(base_model.layers)}")

# Freeze all layers except the last N
fine_tune_from = len(base_model.layers) - 20  # unfreeze the top 20 layers

for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False  # keep early layers frozen

# Check how many layers are now trainable
trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"Trainable layers: {trainable_count} / {len(base_model.layers)}")
```

**Why freeze early layers during fine-tuning?**
- Early layers learned very generic features — no need to change them
- Unfreezing everything risks "catastrophic forgetting" — the model forgets the useful ImageNet features if the learning rate is too high
- Unfreezing only the top layers is a safer, more targeted approach

```python
# Step 2: Recompile with a VERY low learning rate
# This is critical — a high learning rate will destroy the pre-trained weights
model_tl.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # 100x smaller than before
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 3: Train for at least 1 epoch (assignment requirement)
start = time.time()
history_ft = model_tl.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,  # fine-tuning usually needs fewer epochs
    callbacks=[early_stopping],
    class_weight=class_weights
)
ft_time = time.time() - start
```

**`learning_rate=1e-5`:** This is 0.00001. The assignment specifically mentions this value. It's tiny because:
- The pre-trained weights are already good — you want to nudge them, not overwrite them
- A large learning rate would destroy the ImageNet features before the model can adapt them

### Comparing before and after fine-tuning

```python
# Evaluate Phase 1 model (before fine-tuning)
# (restore Phase 1 weights first if you saved them, or track val_loss from Phase 1 history)
print("Phase 1 - Best val accuracy:", max(history_tl.history['val_accuracy']))
print("Phase 2 - Best val accuracy:", max(history_ft.history['val_accuracy']))

# Plot both histories
plot_history(history_tl, "Transfer Learning - Phase 1 (Frozen Base)")
plot_history(history_ft, "Transfer Learning - Phase 2 (Fine-Tuning)")
```

**What to report in your paper:** "After one epoch of fine-tuning with a learning rate of 1e-5, the validation accuracy changed from X% to Y%. / Fine-tuning [did / did not] improve performance." Report the numbers and interpret why.

---

## Practical Preprocessing: The Right Function Per Model

Each model expects a specific input range. Always use the matching `preprocess_input` function:

```python
# MobileNetV2: input scaled to [-1, 1]
from keras.applications.mobilenet_v2 import preprocess_input

# VGG16: input is mean-subtracted (not scaled to [-1,1])
from keras.applications.vgg16 import preprocess_input

# ResNet50: similar to VGG16
from keras.applications.resnet50 import preprocess_input
```

If you use the wrong preprocessing (e.g., rescale to [0,1] when the model expects [-1,1]), performance will be poor for no obvious reason. Always match the preprocessing to the model.

A cleaner approach is to do preprocessing inside the model:

```python
inputs = keras.Input(shape=(224, 224, 3))
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
...
```

This way the preprocessing is baked into the model and you don't need to remember to apply it separately.

---

## Comparing Transfer Learning to Scratch

Record everything in your comparison table:

| Model | Params | Train Acc | Val Acc | Training Time | Notes |
|---|---|---|---|---|---|
| Scratch CNN (RGB) | 8.4M | 89% | 72% | 15min | Best scratch model |
| Transfer (Phase 1) | 3.4M base + 170K head | 95% | 84% | 5min | Frozen base |
| Fine-tuned | same | 96% | 86% | 7min | +1 epoch fine-tune |

In your paper, discuss:
- Did transfer learning outperform your scratch model? (It almost certainly will)
- How does the parameter count compare? (Transfer model "uses" 3.4M base but only *trains* a small head)
- Did fine-tuning improve results? By how much?
- Why does transfer learning work so well even with fewer training steps?

---

## Using VGG16 (if you choose it)

VGG16 is older and simpler but very well understood:

```python
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

base_model = VGG16(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# VGG16 output is (7, 7, 512) — use GlobalAveragePooling or Flatten
inputs = keras.Input(shape=(224, 224, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model_vgg = keras.Model(inputs, outputs)
```

VGG16 is 138M parameters — the base model alone is much larger than any scratch CNN you'd build. Mention this in your paper when comparing parameter counts.

---

## Checklist for This Guide

- [ ] Choose your base model (MobileNetV2, ResNet50, or VGG16)
- [ ] Load with `include_top=False, weights='imagenet'`
- [ ] Freeze the base model and add a custom classification head
- [ ] Train Phase 1 (frozen base)
- [ ] Record val accuracy and training time
- [ ] Unfreeze top layers with `learning_rate=1e-5`
- [ ] Train Phase 2 (fine-tuning) for at least 1 epoch
- [ ] Compare val accuracy before and after fine-tuning
- [ ] Record all results in comparison table
- [ ] Compare to your best scratch CNN

---

## Next Step

→ [Guide 6: Metrics and Evaluation](06_metrics_and_evaluation.md)
