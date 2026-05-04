# Guide 7: Real-World Test and Writing the Paper

This is the final guide. It covers the real-world photo test (which is required) and exactly how to write the 6-page academic paper that your marks actually depend on.

---

## Part 1: The Real-World Test (Section 3.5)

### What you must do

Take **4 photographs yourself** of food items. Test whether your final chosen model correctly classifies them.

**Requirements:**
- Each photo must include **a handwritten note with your name and student ID visible** somewhere in the image — this proves the photos are yours
- The photos must be of food items from the categories your model was trained on
- Submit the 4 photos alongside your notebook and paper

### Taking good photos

- Use your phone camera
- Photograph food items on a clear background if possible
- Take photos that are realistic — similar to what the training data looks like
- But also try at least one "tricky" case (unusual angle, bad lighting, ambiguous food) — this makes for more interesting analysis

### Code: loading and testing your photos

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # or use keras.utils.load_img

def predict_photo(image_path, model, class_names, image_size=(224, 224)):
    # Load and resize the image
    img = keras.utils.load_img(image_path, target_size=image_size)
    img_array = keras.utils.img_to_array(img)          # shape: (H, W, 3)
    img_array = np.expand_dims(img_array, axis=0)      # shape: (1, H, W, 3) — add batch dimension
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    # Show the image and result
    plt.figure(figsize=(6, 5))
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({confidence:.1%} confidence)")
    plt.axis('off')
    plt.show()
    
    # Show all class probabilities
    print("All predictions:")
    sorted_indices = np.argsort(predictions[0])[::-1]  # highest first
    for i in sorted_indices:
        print(f"  {class_names[i]}: {predictions[0][i]:.2%}")
    
    return predicted_class, confidence

# Test each of your 4 photos
photo_paths = [
    'my_photos/photo1.jpg',
    'my_photos/photo2.jpg',
    'my_photos/photo3.jpg',
    'my_photos/photo4.jpg'
]

for path in photo_paths:
    predicted, confidence = predict_photo(path, model, class_names)
    print(f"\n{path}: Predicted '{predicted}' with {confidence:.1%} confidence\n")
```

**Important:** make sure you apply the same preprocessing to your photos as to the training data. If you use a transfer learning model with `preprocess_input`, that's handled inside the model. If you use a scratch CNN with `Rescaling(1./255)` as a layer, that's also handled automatically. But if you normalised with a dataset `.map()` instead, you need to apply it manually here.

### What to discuss in your paper

For each photo, report:
- The true food category
- What the model predicted
- The confidence level
- Whether it was correct

For wrong predictions, **explain why** — was the image:
- Ambiguous? (e.g., a wrap that could look like a burrito)
- A class the model consistently struggles with? (check your confusion matrix)
- An unusual example? (e.g., the food was in unusual lighting or from an odd angle)
- Close to the wrong class visually?

This connects your real-world test to your earlier analysis of model weaknesses.

---

## Part 2: Writing the Academic Paper

### Key rules from the assignment

- **At most 6 pages** — 10% deducted per page over the limit
- **Academic style: third person** ("The model was trained..." not "I trained...")
- **No code in the paper** — describe what you did, not function calls
- **Submitted as PDF only** — 10% penalty for other formats
- **All code goes in the Jupyter notebook**

---

## Suggested Paper Structure

You don't have to follow this exactly, but it maps cleanly to what the marker expects.

---

### Section 1: Introduction (0.5 pages)

A brief overview of the task. What is image classification? What dataset did you use? What is the goal of the study?

Write this last — once you know what you found.

**Example text style:**
> "This study investigates the classification of food images using convolutional neural networks (CNNs). The dataset used is a curated subset of Food-101, containing [N] images across [K] categories. Two approaches are compared: a CNN trained from scratch, and a model employing transfer learning from a pre-trained base network. The objective is to identify which model achieves the highest classification performance while managing overfitting."

---

### Section 2: Data and Preprocessing (1 page)

#### 2a — Dataset Description
- How many images total
- How many classes, what are they
- Image size (512×512 originally, resized to what)
- Colour: RGB
- Source: Food-101 subset

#### 2b — Exploratory Analysis
- Class distribution plot — are classes balanced or imbalanced?
- Sample images (a small grid showing one image per class)
- Any notable observations (lighting variation, image diversity)

#### 2c — Preprocessing and Splits
- Train/validation/test split (70/10/20) — state the sizes in actual image counts
- Random seed (your student ID) — mention this was fixed for reproducibility
- Normalisation: pixel values scaled to [0,1] / per-model preprocessing
- Class weights: were they applied? What were the values?
- Data augmentation: what transformations were applied to training data only

**How to describe augmentation without code:**
> "Training images were augmented through random horizontal flipping, rotations of up to 36 degrees, and zoom adjustments of up to 10%, applied stochastically during training to artificially increase dataset variety and reduce overfitting."

---

### Section 3: Experiments (2 pages)

#### 3a — CNN from Scratch

**What to cover:**
- Your architecture: number of layers, filters per layer, activation functions, pooling, dropout
- Image size comparison (e.g., 64×64 vs 128×128) — show a small table of results
- RGB vs greyscale comparison — same architecture, different input
- Overfitting diagnosis and what you did to address it
- How you iterated: "The initial model showed clear overfitting (training accuracy 94%, validation 61%). Dropout of 0.5 and data augmentation were introduced, improving validation accuracy to..."

**Describe architecture without code:**
> "The CNN architecture consists of three convolutional blocks, each containing two consecutive 3×3 convolution layers with ReLU activation, followed by 2×2 max-pooling. The number of filters increases from 32 in the first block to 128 in the third, allowing the network to learn progressively more abstract features. A dropout layer with rate 0.5 is applied before the final classification layer."

#### 3b — Transfer Learning

**What to cover:**
- Which base model (VGG16 / ResNet50 / MobileNetV2) and why
- How it was configured (frozen base + custom head)
- What the custom head looks like (GAP, dropout, Dense layers)
- Phase 1 results
- Fine-tuning: what you unfroze, learning rate used (1e-5), and for how many epochs
- Before and after fine-tuning results

---

### Section 4: Results and Comparison (1.5 pages)

This is where you present your full comparison table and discuss it.

**Include:**
- Table of all models: image size, colour mode, parameter count, validation F1, training time, inference time
- Final test set results for your chosen model only
- Confusion matrix for the final model (with discussion)

**How to interpret and write about results:**

Do NOT write: "Model 3 had F1 of 0.82."

DO write:
> "The transfer learning model significantly outperformed all scratch-trained models, achieving a macro-averaged F1-score of 0.86 on the validation set compared to 0.76 for the best scratch model. This improvement is attributed to the rich feature representations learned during ImageNet pre-training, which provide a useful initialisation even when the target domain (food images) differs from the source domain (general objects)."

> "Fine-tuning the top 20 layers of the base model with a learning rate of 1×10⁻⁵ improved the validation F1-score from 0.84 to 0.86, suggesting that the high-level features of the pre-trained model benefit from slight adaptation to the food domain."

> "The confusion matrix reveals that the model most frequently confused 'sushi' with 'sashimi' (14 misclassifications), which is visually reasonable given the similar presentation of raw fish in both dishes. In contrast, 'pizza' and 'salad' were almost never confused, reflecting their distinct visual properties."

---

### Section 5: Real-World Test (0.5 pages)

Report the results of your 4 personal photos. Include the photos (small thumbnails), model predictions, and confidence scores.

Discuss:
- Which were correct? Which were wrong?
- For wrong predictions: refer back to confusion matrix — is this a class the model struggles with?
- Is the model's confidence calibrated? (High confidence wrong is more concerning than low confidence wrong)

---

### Section 6: Conclusion (0.5 pages)

Summarise:
- What was the best model and why?
- What were the main factors that improved performance?
- What were the model's main weaknesses?
- What would you do differently or next?

> "The fine-tuned MobileNetV2 model achieved the best performance with a test accuracy of 86% and a macro-averaged F1-score of 0.84. Transfer learning proved significantly more effective than training from scratch, requiring less training time and fewer custom hyperparameter choices while delivering superior generalisation. The primary weakness of all models was confusion among visually similar food categories, particularly sushi-type dishes. Future work could explore larger image sizes and more aggressive augmentation for these problematic classes."

---

### Section 7: References

Cite your sources. The assignment already gives you two references. Also cite:

- The Keras / TensorFlow documentation
- The original VGG16/ResNet50/MobileNetV2 papers if you use transfer learning
- Any other papers you reference

Use numbered references in IEEE or a consistent format. The assignment shows Harvard-style with [1] and [2].

---

## AI Tools Appendix

If you used any AI tool (including me), the assignment requires an appendix stating:
- What you asked it
- What it produced
- What you changed, rejected, or couldn't get to work — and why

This is not penalised but **its absence when AI use is evident is treated as misrepresentation**. Be honest. The appendix doesn't count toward the 6-page limit.

---

## Final Pre-Submission Checklist

### Notebook
- [ ] All cells run top-to-bottom with no errors
- [ ] All outputs visible (graphs, tables, model summaries)
- [ ] `SEED = YOUR_STUDENT_ID` at the top and used everywhere
- [ ] Training time measured for every model
- [ ] Inference time measured with `timeit` on the test set
- [ ] `model.summary()` shown for every model
- [ ] Confusion matrix and `classification_report` for final model
- [ ] 4 real-world photos tested and shown

### Paper
- [ ] At most 6 pages
- [ ] Third person throughout
- [ ] No code — describe processes, not function calls
- [ ] Class distribution plot included
- [ ] Sample images shown
- [ ] Architecture described in words
- [ ] All models in a comparison table (val F1, training time, inference time, params)
- [ ] Test set results reported only for the final chosen model
- [ ] Confusion matrix included and discussed
- [ ] Fine-tuning before/after comparison discussed
- [ ] Real-world test results with discussion
- [ ] References formatted correctly
- [ ] AI appendix if applicable
- [ ] Submitted as PDF

### Files to submit
- [ ] Paper as PDF
- [ ] Jupyter notebook with all outputs
- [ ] Your 4 photographs (with handwritten name and student ID visible)
