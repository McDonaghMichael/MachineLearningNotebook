# ML Assignment 2026 — Complete Guide Index

**Due: 11th May 2026 by 11pm | Worth: 30% of overall grade**

These guides walk you through the entire project from zero knowledge to a complete, explainable submission.

---

## The Assignment in One Sentence

Build multiple CNN models to classify food images, compare them fairly, pick the best one, and write a 6-page academic paper explaining what you did and what you found.

---

## What You Must Produce

| Deliverable | Details |
|---|---|
| Jupyter notebook | All code, all outputs visible, submitted for verification |
| Academic paper | Max 6 pages, PDF only, third person, no code |
| 4 photographs | Your own photos of food, handwritten name+ID visible |

---

## Critical Reminders

- **Your random seed = your student ID** (no G, no leading zeros). Use it everywhere.
- **Test set is evaluated ONCE** at the very end with your best model only.
- **Use `image_dataset_from_directory`** to load data — not manual loading.
- **Use `timeit`** for inference time on the test set.
- **Include `model.summary()`** for every model — parameter count is required.
- **Primary metric:** macro-averaged F1-score (not accuracy, not weighted F1).
- **Class imbalance:** check it, address it with class weights, discuss it.

---

## Guide Sequence

Read and follow these in order:

| # | Guide | What it covers |
|---|---|---|
| 1 | [Understanding CNNs](01_understanding_cnns.md) | What CNNs are, every layer explained, how training works |
| 2 | [Data Loading and Preprocessing](02_data_loading_and_preprocessing.md) | Loading food dataset, train/val/test split, class imbalance, normalisation |
| 3 | [Building a CNN from Scratch](03_building_cnn_from_scratch.md) | Building your first model, all parameters, training curves, RGB vs greyscale |
| 4 | [Overfitting and Regularisation](04_overfitting_and_regularisation.md) | Data augmentation, dropout, L2 regularisation, early stopping |
| 5 | [Transfer Learning](05_transfer_learning.md) | VGG16/ResNet50/MobileNetV2, fine-tuning, what to report |
| 6 | [Metrics and Evaluation](06_metrics_and_evaluation.md) | Confusion matrix, precision, recall, F1, timing, parameter count |
| 7 | [Real-World Test and Paper](07_real_world_test_and_paper.md) | Testing your 4 photos, writing the academic paper section by section |

---

## Recommended Workflow

1. Read Guide 1 (understand the concepts)
2. Read Guide 2 → load your data and explore it
3. Read Guide 3 → build and train baseline CNN, get it working
4. Read Guide 4 → fix overfitting, iterate
5. Read Guide 5 → build transfer learning model, fine-tune
6. Read Guide 6 → evaluate every model properly, build comparison table
7. Take your 4 photos and test them
8. Read Guide 7 → write the paper

---

## Models You Must Build

| Model | Required? | Notes |
|---|---|---|
| CNN from scratch (RGB) | Yes | Start here, iterate until good |
| CNN from scratch (Greyscale) | Yes | Same architecture as best RGB |
| Transfer learning (RGB) | Yes | VGG16, ResNet50, or MobileNetV2 |
| Fine-tuned transfer model | Yes | At least 1 epoch, LR = 1e-5 |
| Image size comparison | Yes | At least 2 sizes, or justify 1 |
