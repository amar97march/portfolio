---
title: "Algorithm Spotlight: Logistic Regression — Binary Classification Explained"
date: 2026-05-30T10:00:00+05:30
draft: false
description: "A comprehensive guide to logistic regression — the go-to algorithm for binary classification that uses the sigmoid function to predict probabilities."
tags: ["Logistic Regression", "Classification", "Machine Learning", "Algorithms", "Sigmoid"]
categories: ["Machine Learning"]
image: "/images/blogs/pool-ml/1.jpg"
keywords: ["logistic regression", "binary classification", "sigmoid function", "classification algorithm", "ML algorithms"]
---

Despite its name, logistic regression is not a regression algorithm — it is a **classification** algorithm. Specifically, it is the go-to algorithm for **binary classification**: predicting one of two outcomes. Will the email be spam or not? Will the customer churn or stay? Will the patient test positive or negative?

Logistic regression is one of the most important algorithms in machine learning. It is simple, fast, interpretable, and surprisingly effective. It also serves as the conceptual bridge between linear regression and neural networks — understanding logistic regression makes both more intuitive.

## From Linear Regression to Logistic Regression

Linear regression predicts continuous values: house prices, temperatures, stock returns. But what if we want to predict a binary outcome — yes or no, 0 or 1?

The naive approach would be to use linear regression and round the output. But this fails for several reasons:

1. Linear regression outputs can be any real number (negative, greater than 1), which makes no sense as a probability
2. The linear decision boundary is too sensitive to outliers
3. Linear regression minimizes squared error, which is not the right objective for classification

Logistic regression solves this by wrapping the linear model in a **sigmoid function**:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Linear model: z = w*x + b
# Logistic model: p = sigmoid(w*x + b)

z = np.linspace(-10, 10, 100)
p = sigmoid(z)

# The output is always between 0 and 1 — a valid probability!
print(f"sigmoid(-10) = {sigmoid(-10):.6f}")  # ≈ 0
print(f"sigmoid(0)   = {sigmoid(0):.6f}")    # = 0.5
print(f"sigmoid(10)  = {sigmoid(10):.6f}")   # ≈ 1
```

## How Logistic Regression Works

### Step 1: Compute the Linear Combination

Just like linear regression, compute a weighted sum of features:

```
z = w1*x1 + w2*x2 + ... + wn*xn + b
```

### Step 2: Apply the Sigmoid Function

Pass the result through the sigmoid to get a probability:

```
p = 1 / (1 + e^(-z))
```

### Step 3: Make a Decision

Apply a threshold (typically 0.5) to convert the probability to a class:

```
prediction = 1 if p >= 0.5 else 0
```

```python
def logistic_regression_predict(X, weights, bias, threshold=0.5):
    z = np.dot(X, weights) + bias
    probabilities = sigmoid(z)
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities
```


![Machine learning algorithm visualization](/images/blogs/pool-ml/6.jpg)

## The Loss Function: Binary Cross-Entropy

Linear regression uses mean squared error. Logistic regression uses **binary cross-entropy** (also called log loss):

```
Loss = -[y * log(p) + (1 - y) * log(1 - p)]
```

This loss function has beautiful properties:
- When `y = 1` and `p` is close to 1: loss is near 0 (correct and confident)
- When `y = 1` and `p` is close to 0: loss is very large (confident but wrong)
- When `y = 0` and `p` is close to 0: loss is near 0 (correct and confident)
- When `y = 0` and `p` is close to 1: loss is very large (confident but wrong)

```python
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # Avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

## Implementation from Scratch

```python
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iter):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(z)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Track loss
            loss = binary_cross_entropy(y, predictions)
            self.losses.append(loss)

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Usage
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.4f}")
```

## Using Scikit-Learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Comprehensive evaluation
print(classification_report(y_test, y_pred))
print(f"\nAUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

# Interpret coefficients
for feature_idx, coef in enumerate(model.coef_[0]):
    print(f"Feature {feature_idx}: coef={coef:.4f}, odds ratio={np.exp(coef):.4f}")
```


![Statistical modeling and classification concepts](/images/blogs/pool-ml/7.jpg)

## Interpreting Coefficients

One of logistic regression's greatest strengths is interpretability. The coefficients have a clear meaning in terms of **odds ratios**:

- A coefficient of 0.5 means a one-unit increase in that feature multiplies the odds by e^0.5 = 1.65 (65% increase in odds)
- A coefficient of -0.3 means a one-unit increase multiplies the odds by e^-0.3 = 0.74 (26% decrease in odds)
- A coefficient of 0 means the feature has no effect on the odds

```python
# Example: Predicting customer churn
feature_names = ['tenure', 'monthly_charges', 'contract_length',
                 'tech_support', 'num_complaints']

print("\nFeature Interpretation:")
for name, coef in zip(feature_names, model.coef_[0]):
    odds_ratio = np.exp(coef)
    direction = "increases" if coef > 0 else "decreases"
    print(f"  {name}: {direction} churn odds by {abs(odds_ratio - 1)*100:.1f}% per unit")
```

## The Decision Boundary

Logistic regression creates a linear decision boundary in feature space. In 2D, this is a line; in 3D, a plane; in higher dimensions, a hyperplane.

```python
import matplotlib.pyplot as plt

# 2D example for visualization
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=2.0)
model = LogisticRegression()
model.fit(X, y)

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                      np.arange(y_min, y_max, 0.01))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
plt.title("Logistic Regression Decision Boundary")
plt.savefig('decision_boundary.png')
```

## Choosing the Threshold

The default threshold of 0.5 is not always optimal. Adjusting it trades off precision and recall:

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Find threshold for desired recall (e.g., 90% recall)
target_recall = 0.90
idx = np.argmin(np.abs(recalls - target_recall))
optimal_threshold = thresholds[idx]
print(f"Threshold for {target_recall:.0%} recall: {optimal_threshold:.3f}")

# Apply custom threshold
y_pred_custom = (y_prob >= optimal_threshold).astype(int)
print(f"Precision at this threshold: {precision_score(y_test, y_pred_custom):.3f}")
print(f"Recall at this threshold: {recall_score(y_test, y_pred_custom):.3f}")
```


![Data science workflow and model evaluation](/images/blogs/pool-ml/8.jpg)

## Multi-Class Logistic Regression

Logistic regression can handle more than two classes using **One-vs-Rest** or **Softmax (Multinomial)**:

```python
from sklearn.datasets import load_iris

iris = load_iris()
model = LogisticRegression(multi_class='multinomial', max_iter=1000)
model.fit(iris.data, iris.target)

# Probabilities for each class
probs = model.predict_proba(iris.data[:3])
for i, prob in enumerate(probs):
    print(f"Sample {i}: {dict(zip(iris.target_names, prob.round(3)))}")
```

## When to Use Logistic Regression

**Use it when:**
- You need a fast, interpretable baseline
- The relationship between features and log-odds is approximately linear
- You need probability outputs, not just class labels
- You need to understand which features drive predictions
- You have a binary or multi-class classification problem

**Avoid it when:**
- The decision boundary is highly nonlinear (use tree-based models or neural networks)
- Feature interactions are important (logistic regression does not capture them without manual feature engineering)
- You have very high-dimensional sparse data (consider regularized variants like L1)

## Key Takeaways

1. Logistic regression uses the sigmoid function to map linear outputs to probabilities
2. It minimizes binary cross-entropy loss, not squared error
3. Coefficients are interpretable as log-odds ratios
4. The decision boundary is always linear (or hyperplanar in higher dimensions)
5. Threshold tuning allows trading off precision and recall
6. Despite its simplicity, it remains one of the most widely used classification algorithms in industry

---

*Next: Applying linear regression to a real problem — predicting house prices.*
