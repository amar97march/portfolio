---
title: "Code Tutorial: Building a Decision Tree in Scikit-Learn"
date: 2026-07-02T10:00:00+05:30
draft: false
description: "A hands-on tutorial for building, visualizing, and evaluating a Decision Tree classifier using Scikit-Learn. Includes code for training, pruning, cross-validation, and feature importance analysis."
tags: ["Machine Learning", "Decision Trees", "Scikit-Learn", "Tutorial", "Python"]
categories: ["Machine Learning"]
image: "/images/blogs/pool-ml/1.jpg"
keywords: ["decision tree tutorial", "scikit-learn decision tree", "python machine learning tutorial", "decision tree visualization", "cross validation"]
---

Theory is important, but nothing cements understanding like writing code. In this tutorial, we will build a Decision Tree classifier from scratch using Scikit-Learn, visualize it, tune it, and evaluate it properly.

By the end, you will have a complete, reproducible pipeline that you can adapt to your own datasets.

---

### Step 1: Setup and Data Loading

We will use the Wine dataset — a classic multiclass classification problem with 13 features describing chemical properties of wines from three different cultivars.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Load the dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='target')

print(f"Dataset shape: {X.shape}")
print(f"Classes: {wine.target_names}")
print(f"\nClass distribution:")
print(y.value_counts().sort_index())
print(f"\nFeature summary:")
print(X.describe().round(2))
```

The Wine dataset has 178 samples, 13 features, and 3 classes. It is small enough to train quickly but complex enough to demonstrate real concepts.

```python
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
```

We use `stratify=y` to ensure both splits have the same class proportions. This is important for smaller datasets.

---

### Step 2: Training a Basic Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

# Start with an unconstrained tree
tree_full = DecisionTreeClassifier(random_state=42)
tree_full.fit(X_train, y_train)

print(f"Tree depth: {tree_full.get_depth()}")
print(f"Number of leaves: {tree_full.get_n_leaves()}")
print(f"Training accuracy: {tree_full.score(X_train, y_train):.4f}")
print(f"Test accuracy: {tree_full.score(X_test, y_test):.4f}")
```

You will likely see that the training accuracy is 1.0 (perfect) while the test accuracy is lower. This is the hallmark of overfitting — the unconstrained tree has memorized the training data.

---

### Step 3: Visualizing the Tree

One of the greatest advantages of Decision Trees is that you can literally see the model's logic.

```python
from sklearn.tree import plot_tree

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    tree_full,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax
)
plt.title("Full Decision Tree (Unconstrained)")
plt.tight_layout()
plt.savefig("decision_tree_full.png", dpi=150, bbox_inches='tight')
plt.show()
```

You can also print a text representation:

```python
from sklearn.tree import export_text

tree_rules = export_text(
    tree_full,
    feature_names=list(wine.feature_names)
)
print(tree_rules[:1000])  # Print first 1000 characters
```

---

![Visualizing decision tree structure and splits](/images/blogs/pool-ml/6.jpg)

### Step 4: Pruning — Finding the Right Complexity

We need to constrain the tree to prevent overfitting. Let us try different values of `max_depth`:

```python
depths = range(1, 15)
train_scores = []
test_scores = []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'b-o', label='Training Accuracy')
plt.plot(depths, test_scores, 'r-o', label='Test Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Training vs Test Accuracy by Depth')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(depths)
plt.tight_layout()
plt.savefig("depth_analysis.png", dpi=150)
plt.show()
```

You will see the classic bias-variance tradeoff: at low depths, both training and test accuracy are low (underfitting). As depth increases, training accuracy rises. At some point, test accuracy plateaus or starts dropping (overfitting).

#### Cost-Complexity Pruning (Post-Pruning)

Scikit-Learn also supports cost-complexity pruning via the `ccp_alpha` parameter:

```python
# Find the effective alpha values
path = tree_full.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

# Train trees for each alpha value
trees = []
for alpha in ccp_alphas:
    tree = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    tree.fit(X_train, y_train)
    trees.append(tree)

# Plot accuracy vs alpha
train_scores = [t.score(X_train, y_train) for t in trees]
test_scores = [t.score(X_test, y_test) for t in trees]

plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, 'b-', label='Training', marker='o', markersize=3)
plt.plot(ccp_alphas, test_scores, 'r-', label='Test', marker='o', markersize=3)
plt.xlabel('Cost-Complexity Alpha')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Alpha for Cost-Complexity Pruning')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ccp_pruning.png", dpi=150)
plt.show()
```

---

### Step 5: Cross-Validation — Reliable Evaluation

A single train-test split can be misleading. Cross-validation gives a more robust estimate of model performance.

```python
from sklearn.model_selection import cross_val_score

tree_pruned = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)

# 5-fold cross-validation
cv_scores = cross_val_score(tree_pruned, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"Individual folds: {cv_scores.round(4)}")
```

For a more thorough evaluation, use `cross_validate` to get multiple metrics:

```python
from sklearn.model_selection import cross_validate

scores = cross_validate(
    tree_pruned, X, y, cv=5,
    scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'],
    return_train_score=True
)

for metric in ['accuracy', 'f1_weighted']:
    train_key = f'train_{metric}'
    test_key = f'test_{metric}'
    print(f"{metric}:")
    print(f"  Train: {scores[train_key].mean():.4f} (+/- {scores[train_key].std():.4f})")
    print(f"  Test:  {scores[test_key].mean():.4f} (+/- {scores[test_key].std():.4f})")
```

---

![Cross-validation and model evaluation workflow](/images/blogs/pool-ml/7.jpg)

### Step 6: Hyperparameter Tuning with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [2, 3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 3, 5],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
print(f"Test accuracy: {grid_search.score(X_test, y_test):.4f}")
```

---

### Step 7: Feature Importance Analysis

```python
best_tree = grid_search.best_estimator_

importances = pd.Series(
    best_tree.feature_importances_,
    index=wine.feature_names
).sort_values(ascending=True)

plt.figure(figsize=(10, 8))
importances.plot(kind='barh', color='steelblue')
plt.xlabel('Feature Importance (Gini)')
plt.title('Decision Tree Feature Importance')
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
```

Note that Decision Tree feature importance can be misleading — correlated features may show lower importance than they deserve. For a more robust assessment, use permutation importance:

```python
from sklearn.inspection import permutation_importance

perm_result = permutation_importance(
    best_tree, X_test, y_test, n_repeats=30, random_state=42
)

perm_importances = pd.Series(
    perm_result.importances_mean,
    index=wine.feature_names
).sort_values(ascending=True)

plt.figure(figsize=(10, 8))
perm_importances.plot(kind='barh', color='coral')
plt.xlabel('Permutation Importance')
plt.title('Permutation-Based Feature Importance')
plt.tight_layout()
plt.savefig("permutation_importance.png", dpi=150)
plt.show()
```

---

![Feature importance analysis and model interpretation](/images/blogs/pool-ml/8.jpg)

### Step 8: Final Model and Predictions

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Final predictions
y_pred = best_tree.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=wine.target_names,
            yticklabels=wine.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
```

---

### Complete Pipeline Summary

Here is the full pipeline in a concise form:

```python
# 1. Load and split data
# 2. Explore the dataset
# 3. Train a baseline (unconstrained) tree
# 4. Visualize and diagnose overfitting
# 5. Tune hyperparameters with GridSearchCV
# 6. Evaluate with cross-validation
# 7. Analyze feature importance
# 8. Generate final predictions and reports
```

This pipeline applies to virtually any classification problem. Swap out the dataset, adjust the hyperparameter grid, and you have a production-ready workflow.

---

### Final Thoughts

This tutorial covered the complete lifecycle of building a Decision Tree model — from data loading to evaluation. The key takeaways are:

1. Always start with an unconstrained tree to understand the data, then prune.
2. Use cross-validation, not a single train-test split, for reliable evaluation.
3. Tune hyperparameters systematically with GridSearchCV.
4. Visualize the tree — it is one of the few models where you can literally see the logic.
5. Use permutation importance for more reliable feature rankings.

In the next post, we will step back from algorithms and ask an important question: **When do you actually need deep learning, and when is a simpler model enough?**
