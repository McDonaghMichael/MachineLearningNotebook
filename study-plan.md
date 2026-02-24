# 1-Week Machine Learning Catch-Up Plan
## Every day except Tuesday = 6 study days (~4–6 hrs each)

---

## Course Material Reference

| Material | File |
|----------|------|
| Lecture 1 | `Lecture1 - Intro Regression.pdf` |
| Lecture 2 | `Lecture2 - Generalisation.pdf` |
| Lecture 3 | `Lecture3 - Classification Logistic Regression.pdf` |
| Lecture 4 | `Lecture4 - Metrics kNearest Neighbours.pdf` |
| Lecture 5 | `Lecture5 - Decision Trees-RandomForests.pdf` |
| Lecture 6 | `Lecture6 - Linear Models - SVM.pdf` |
| Lab 1 | `Week 1 Files-20260221/ML_Lab1_Data_Fundamentals.ipynb` |
| Lab 2 | `Week 2 Files-20260221/LinearRegressionLab.ipynb` |
| Lab 3 | `Week 3 Files-20260221/logistic_regression_lab.ipynb` |
| Lab 4 | `Week 4 Files-20260221/knn_lab.ipynb` |
| Lab 5 | `Week 5 Files-20260221/Decision Trees Lab.ipynb` |

---

## Day 1 — Monday: Foundations
**~4 hours**

### Morning — Lab 1: Data Fundamentals
**File:** `Week 1 Files-20260221/ML_Lab1_Data_Fundamentals.ipynb`
- Run every cell: NumPy, Pandas, CSV/JSON loading, matplotlib, scikit-learn intro, k-means
- Don't just read — execute and tweak the code

### Afternoon — Lecture 1: Intro to Regression
**File:** `Lecture1 - Intro Regression.pdf`
- What is a model, feature, target?
- Loss function = MSE (mean squared error)
- Gradient descent: minimise loss by adjusting weights

---
## TUESDAY — OFF
---

## Day 2 — Wednesday: Regression + Generalisation
**~5 hours**

### Morning — Lecture 2: Generalisation
**File:** `Lecture2 - Generalisation.pdf`
- **Overfitting** = high train accuracy, low test accuracy → model too complex
- **Underfitting** = low on both → model too simple
- Fix: train/validation/test split — never evaluate on training data
- Bias-variance tradeoff
> This is the most important theory topic — understand it deeply.

### Afternoon — Lab 2: Linear Regression
**File:** `Week 2 Files-20260221/LinearRegressionLab.ipynb`
- Dataset: `mtcars.csv`
- Simple → multiple linear regression
- Skills: `LinearRegression().fit()`, `.coef_`, `.predict()`, MSE, R², adjusted R²
- Try adding/removing features and watch how R² changes

---

## Day 3 — Thursday: Classification + Logistic Regression
**~5 hours**

### Morning — Lecture 3: Classification & Logistic Regression
**File:** `Lecture3 - Classification Logistic Regression.pdf`
- Classification outputs a category, not a number
- Sigmoid: squashes output into 0–1 probability
- Decision boundary: line separating classes
- C parameter: high C = less regularisation (fits training data more closely)

### Afternoon — Lab 3: Logistic Regression
**File:** `Week 3 Files-20260221/logistic_regression_lab.ipynb`
- Datasets: `heart_disease.csv` (binary), `wine_quality.csv` (multi-class)
- Skills: `LogisticRegression()`, `StandardScaler()`, `confusion_matrix`, `classification_report`
- Always scale features before logistic regression

**Memorise these metrics:**
- Precision = TP / (TP + FP) → of what I predicted positive, how many were right?
- Recall = TP / (TP + FN) → of all actual positives, how many did I catch?
- F1 = harmonic mean of precision & recall → use when classes are imbalanced
- Accuracy = correct / total → only use when classes are balanced

---

## Day 4 — Friday: Metrics + kNN
**~5 hours**

### Morning — Lecture 4: Metrics & kNN
**File:** `Lecture4 - Metrics kNearest Neighbours.pdf`
- Deep dive on precision, recall, F1 — make sure you can calculate them by hand
- kNN: classify by finding k nearest data points and voting
- Small k = complex/noisy, large k = smooth/simple
- k is a hyperparameter — tune it with cross-validation

### Afternoon — Lab 4: kNN
**File:** `Week 4 Files-20260221/knn_lab.ipynb`
- Dataset: `penguins.csv`
- kNN **requires feature scaling** (distance-based algorithm)
- Skills: `KNeighborsClassifier`, `GridSearchCV`, `cross_val_score`
- Pipeline: `Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])`
- Also look at: `BreastCancerExample.ipynb`

**Cross-validation key concepts:**
- k-fold CV: split data k times, train on k-1 folds, test on 1 → average score
- More reliable than a single train/test split
- `GridSearchCV`: tries all hyperparameter combinations with CV built in

---

## Day 5 — Saturday: Decision Trees + SVM
**~5 hours**

### Morning — Lecture 5 + Lab 5: Decision Trees
**Files:** `Lecture5 - Decision Trees-RandomForests.pdf` + `Week 5 Files-20260221/Decision Trees Lab.ipynb`

**Lecture 5 key concepts:**
- Trees split on features to reduce impurity (Gini or entropy)
- Deep trees overfit → control with `max_depth`, `min_samples_split`, `min_samples_leaf`
- Feature importance: trees rank features automatically
- Random Forests: many trees + randomness = better generalisation (ensemble)
- Decision trees do **NOT** need feature scaling

**Lab 5:**
- Dataset: `creditcard.csv` (fraud — severely imbalanced!)
- Accuracy fails here → use F1/recall
- Skills: `DecisionTreeClassifier`, `feature_importances_`, `GridSearchCV(scoring='f1')`

### Afternoon — Lecture 6: SVM
**File:** `Lecture6 - Linear Models - SVM.pdf`
- SVM finds the maximum margin hyperplane between classes
- Support vectors = data points closest to the boundary
- Kernel trick: RBF kernel allows non-linear boundaries
- C parameter: controls regularisation (same intuition as logistic regression)
- SVM **requires feature scaling**

---

## Day 6 — Sunday: Mock + Final Review
**~4–5 hours**

### Morning — Algorithm Comparison (write this by hand)

| Algorithm | Needs Scaling? | Key Hyperparameters | When to use |
|-----------|---------------|---------------------|-------------|
| Linear Regression | No | — | Continuous output |
| Logistic Regression | Yes | C | Binary/multi-class |
| kNN | **Yes** | k | Simple, no assumptions |
| Decision Tree | No | max_depth, min_samples | Interpretable |
| Random Forest | No | n_estimators, max_depth | General purpose |
| SVM | **Yes** | C, kernel | Small datasets, high-dim |

### Midday — Timed Mock Practical (~90 mins)
Pick one dataset. Without looking at any lab:
1. Load and explore (`df.head()`, `df.info()`, `df.describe()`)
2. Split into train/test (80/20)
3. Scale if needed
4. Fit 2 different models
5. Evaluate with appropriate metrics — justify your choice
6. Pick the better model and explain why

### Afternoon — Final Self-Test
Can you answer these without notes?
- What is overfitting? How do you detect and fix it?
- When do you use F1 instead of accuracy?
- Why does kNN need scaling but Decision Trees don't?
- What is cross-validation and why is it better than a single split?
- What does the C parameter control?
- What is a support vector?

---

## Essential Code Template (same sklearn pattern every time)
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)   # NEVER fit on test data — only transform

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Overfitting Quick Reference
- Train accuracy high, test accuracy low → **overfitting** → reduce complexity
- Both accuracies low → **underfitting** → increase complexity
- Decision Tree: lower `max_depth`
- Logistic Regression / SVM: lower C (more regularisation)
- kNN: increase k
