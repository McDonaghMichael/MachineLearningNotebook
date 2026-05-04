# Guide 6: Metrics and Evaluation

The assignment requires more than just accuracy. This guide covers every metric you need, how to calculate it, what it means, and how to interpret it.

---

## Why Accuracy Alone Isn't Enough

Imagine your dataset has 900 pizza images and 100 sushi images. A model that always predicts "pizza" gets **90% accuracy**. But it never identifies sushi at all — it's completely useless. This is the class imbalance problem.

You need metrics that account for per-class performance:
- **Precision** — when the model says "pizza", how often is it right?
- **Recall** — of all actual pizza images, how many does the model catch?
- **F1-score** — a single number combining both

---

## The Confusion Matrix

The confusion matrix is the foundation of all classification metrics. It's a table showing what the model predicted vs. what the true label was.

```python
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Get predictions from the test set
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))  # class with highest probability
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Compute and plot the confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,         # show numbers in each cell
    fmt='d',            # format as integers
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

### Reading the confusion matrix

Each row represents the **true class**. Each column represents the **predicted class**.

```
            Predicted:
            Pizza  Sushi  Burger
True Pizza:  [90]    5      5        ← 90 pizzas correctly identified
True Sushi:   8    [85]    7        ← 85 sushis correctly identified  
True Burger:  3      2    [95]      ← 95 burgers correctly identified
```

**Diagonal values** (top-left to bottom-right) = correct predictions.
**Off-diagonal values** = errors — what class was confused with what.

**What to look for:**
- Which classes have the most errors?
- Is there a pattern? (e.g., burger is often confused with hot dog)
- Which class is hardest to classify?
- Is there asymmetric confusion? (A is confused for B more than B for A)

Include this in your paper and **discuss the patterns you see**.

---

## Precision, Recall, and F1-Score

For each class:

**Precision:** Of everything the model labelled as class X, what fraction was actually class X?
```
Precision = True Positives / (True Positives + False Positives)
```
High precision = few false alarms.

**Recall:** Of all actual instances of class X, what fraction did the model find?
```
Recall = True Positives / (True Positives + False Negatives)
```
High recall = few misses.

**F1-Score:** The harmonic mean of precision and recall. Balanced metric.
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

All three range from 0 to 1. Higher is better.

### Example

You have 100 pizza images in the test set. The model:
- Correctly identifies 80 pizzas (True Positives = 80)
- Misses 20 pizzas, calling them burger (False Negatives = 20)
- Calls 10 burgers "pizza" when they're not (False Positives = 10)

```
Precision = 80 / (80 + 10) = 0.89  (89% of things it called pizza are pizza)
Recall    = 80 / (80 + 20) = 0.80  (it found 80% of all pizzas)
F1        = 2 × (0.89 × 0.80) / (0.89 + 0.80) = 0.84
```

---

## Getting All Metrics at Once

```python
from sklearn.metrics import classification_report

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4  # 4 decimal places
)
print(report)
```

Output looks like:
```
              precision    recall  f1-score   support

      burger     0.8923    0.8800    0.8861       100
       pizza     0.9100    0.9200    0.9150       200
       sushi     0.8500    0.8700    0.8598       150

    accuracy                         0.8933       450
   macro avg     0.8841    0.8900    0.8870       450
weighted avg     0.8952    0.8933    0.8942       450
```

**`support`:** how many images of each class are in the test set.

**`macro avg`:** average across all classes, treating each class equally regardless of size. This is the metric the assignment says to use as your primary comparison metric — it treats all classes equally.

**`weighted avg`:** average weighted by class size. This inflates the score for majority classes and is misleading when classes are imbalanced. Don't use this as your primary metric.

**What to report:** macro-averaged F1-score for every model, in a comparison table.

---

## Evaluating on the Test Set — Only Once

```python
# Only do this for your FINAL chosen model
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Then get per-class metrics
y_true = []
y_pred = []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

print(classification_report(y_true, y_pred, target_names=class_names))
```

**Important:** the test set is evaluated once, right at the end. This gives you the honest, unbiased measure of how well your model generalises. Report these numbers in your paper.

---

## Training Time

The assignment wants training time included in comparisons.

```python
import time

start = time.time()
history = model.fit(train_ds, validation_data=val_ds, epochs=20)
training_time = time.time() - start

print(f"Training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
```

Report this for every model. Transfer learning should train much faster than scratch models (for Phase 1). Larger image sizes train slower.

---

## Inference Time — Use `timeit`

The assignment specifically says to use `timeit` for inference measurements on the test set. Inference time measures how fast the model can make predictions.

```python
import timeit

# Get one batch of test images
for images, labels in test_ds.take(1):
    test_batch = images

# Warm up (first run is often slower due to compilation)
_ = model.predict(test_batch, verbose=0)

# Time inference on one batch
n_repeats = 10
total_time = timeit.timeit(
    lambda: model.predict(test_batch, verbose=0),
    number=n_repeats
)
avg_time_ms = (total_time / n_repeats) * 1000

print(f"Average inference time per batch of {len(test_batch)}: {avg_time_ms:.2f} ms")
print(f"Per image: {avg_time_ms / len(test_batch):.2f} ms")
```

**Do NOT use `timeit` during training** — it's only for measuring prediction speed on the test set.

**Inference time comparison** is interesting because:
- Transfer learning models (especially VGG16) may be slower at inference despite training faster (frozen Phase 1 is fast to train, but the large model is slower to predict)
- MobileNetV2 is designed for fast inference
- Smaller image sizes = faster inference

---

## Parameter Count

Run `model.summary()` and read the "Total params" line. Include this in your comparison table.

```python
model.summary()

# Get the number programmatically:
total_params = model.count_params()
print(f"Total parameters: {total_params:,}")
```

For transfer learning, clarify in your paper:
- Total params in the model
- How many are trainable (Phase 1: only the head; Phase 2: head + top layers)

```python
trainable = sum(v.numpy().size for v in model.trainable_variables)
non_trainable = total_params - trainable
print(f"Trainable: {trainable:,}")
print(f"Non-trainable (frozen): {non_trainable:,}")
```

**Discussion point for paper:** VGG16 has 138M parameters. Your scratch CNN might have 8M. Despite the scratch model being smaller, the transfer model will likely perform better — because the 138M parameters were already trained on ImageNet. What does this tell you about the relationship between model size and performance?

---

## Building a Full Comparison Table

For your paper, build this table for every model you trained:

| Model | Image Size | Colour | Params | Val F1 (macro) | Test Acc | Test F1 | Training Time | Inference (ms/batch) |
|---|---|---|---|---|---|---|---|---|
| Baseline CNN | 128×128 | RGB | 8.4M | 0.61 | — | — | 4min | 12ms |
| Deeper CNN | 128×128 | RGB | 15M | 0.70 | — | — | 7min | 18ms |
| Deeper + Aug + Dropout | 128×128 | RGB | 15M | 0.76 | — | — | 9min | 18ms |
| Same + Greyscale | 128×128 | Grey | 15M | 0.73 | — | — | 8min | 17ms |
| MobileNetV2 (Phase 1) | 224×224 | RGB | 3.4M | 0.84 | — | — | 5min | 25ms |
| MobileNetV2 (Fine-tuned) | 224×224 | RGB | 3.4M | 0.86 | — | — | 7min | 25ms |
| **Best model** | | | | | **X%** | **X.XX** | | |

Fill in Test Acc and Test F1 only for your final chosen model (the one with the highest val F1).

---

## Interpreting the Confusion Matrix for Your Paper

Don't just show the heatmap. Explain it. For example:

> "The model performed well on most classes but frequently confused 'hamburger' with 'hot dog' (18 misclassifications). This is expected — both are elongated, brown, meat-based foods photographed from similar angles. The model struggled most with 'sushi' (recall=0.72), potentially due to the variety of sushi types within the class, as nigiri, rolls, and sashimi differ significantly in appearance."

That kind of interpretation is what earns marks. The numbers are just evidence for your interpretation.

---

## Checklist for This Guide

- [ ] Generate and plot confusion matrix for your final model (test set)
- [ ] Run `classification_report` — record macro avg precision, recall, F1
- [ ] Measure training time for every model
- [ ] Measure inference time using `timeit` on the test set
- [ ] Record parameter count from `model.summary()` for every model
- [ ] Build a full comparison table
- [ ] Interpret the confusion matrix in your paper — identify patterns

---

## Next Step

→ [Guide 7: Real-World Test and Writing the Paper](07_real_world_test_and_paper.md)
