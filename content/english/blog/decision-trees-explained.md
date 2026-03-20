---
title: "Algorithm Spotlight: Decision Trees — A Flowchart of Questions"
date: 2026-06-20T10:00:00+05:30
draft: false
description: "Decision Trees make predictions by asking a series of yes-or-no questions. Learn how they work, how they split data, and why they are one of the most interpretable algorithms in machine learning."
tags: ["Machine Learning", "Decision Trees", "Classification", "Algorithms", "Python"]
categories: ["Machine Learning"]
image: "/images/blogs/pool-ml/1.jpg"
keywords: ["decision trees", "decision tree explained", "gini impurity", "information gain", "entropy", "machine learning classification"]
---

If you have ever played the game "20 Questions," you already understand how a Decision Tree works. You ask a series of yes-or-no questions, each one narrowing down the possibilities, until you arrive at an answer.

Decision Trees are one of the most intuitive and interpretable algorithms in machine learning. Unlike SVMs or neural networks, you can literally draw a Decision Tree on a whiteboard and explain every single prediction it makes. This transparency makes them invaluable in industries where explainability is not optional — healthcare, finance, and legal.

In this post, we will build your understanding from the ground up.

---

### Part 1: The Intuition — Asking Smart Questions

Imagine you work at a bank and need to decide whether to approve a loan. You might ask a series of questions:

1. **Is the applicant's credit score above 700?**
   - If yes, lean toward approval.
   - If no, ask more questions.
2. **Does the applicant have a stable income above $50,000?**
   - If yes, approve.
   - If no, check their debt-to-income ratio.
3. **Is their debt-to-income ratio below 40%?**
   - If yes, approve with conditions.
   - If no, reject.

This is exactly what a Decision Tree does. It learns the *best* sequence of questions from the training data to make the most accurate predictions.

Each question is called a **split**. Each endpoint is called a **leaf node**. The path from the root to a leaf is the decision rule for that prediction.

---

### Part 2: How Does the Tree Decide Which Question to Ask?

This is the key question. Not all splits are equally useful. Asking "Is the applicant's name alphabetically before M?" is a terrible split because it has no predictive power. Asking about credit score is brilliant because it strongly separates good borrowers from risky ones.

Decision Trees use mathematical criteria to find the **best split** at every node. The two most common criteria are:

#### Gini Impurity

Gini Impurity measures how "mixed" a node is. A node that contains only one class has a Gini of 0 (perfectly pure). A node that is evenly split has the maximum Gini.

$$Gini = 1 - \sum_{i=1}^{c} p_i^2$$

Where $p_i$ is the proportion of class $i$ in the node.

**Example:** A node with 90 positive and 10 negative samples:
$$Gini = 1 - (0.9^2 + 0.1^2) = 1 - (0.81 + 0.01) = 0.18$$

A node with 50 positive and 50 negative:
$$Gini = 1 - (0.5^2 + 0.5^2) = 1 - 0.5 = 0.5$$

The tree picks the split that results in the **lowest weighted Gini** across the child nodes.

#### Information Gain (Entropy-Based)

Entropy measures the disorder or uncertainty in a node:

$$Entropy = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

**Information Gain** is the reduction in entropy after a split:

$$IG = Entropy(parent) - \sum_{k} \frac{n_k}{n} Entropy(child_k)$$

The tree picks the split that maximizes Information Gain.

Both methods generally produce similar trees. Scikit-Learn uses Gini by default because it is slightly faster to compute.

---

![Decision tree splitting data into pure groups](/images/blogs/pool-ml/3.jpg)

### Part 3: Building a Decision Tree Step by Step

Let us trace through the algorithm:

1. **Start at the root.** Consider all possible features and all possible split points.
2. **Evaluate every possible split** using Gini or Entropy.
3. **Choose the best split** — the one that produces the purest child nodes.
4. **Create two child nodes** based on the split.
5. **Repeat recursively** for each child node.
6. **Stop** when a stopping criterion is met (maximum depth, minimum samples per leaf, or pure nodes).

This is called the **CART algorithm** (Classification and Regression Trees), developed by Leo Breiman in 1984.

---

### Part 4: Decision Trees for Regression

Decision Trees are not limited to classification. For regression problems, instead of predicting a class, each leaf predicts a continuous value — typically the **mean** of the target values in that leaf.

The split criterion changes too. Instead of Gini or Entropy, we minimize the **Mean Squared Error (MSE)** or **Mean Absolute Error (MAE)** at each split.

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2$$

The tree finds the split that minimizes the weighted MSE across child nodes.

---

### Part 5: Code Example — Building a Decision Tree

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Train the tree
tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    min_samples_split=5,
    random_state=42
)
tree.fit(X_train, y_train)

# Evaluate
y_pred = tree.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Print the tree as text
print(export_text(tree, feature_names=iris.feature_names))
```

The `export_text` function prints a human-readable version of the tree, showing exactly which features and thresholds the model uses at each split. This is the interpretability advantage that makes Decision Trees so valuable.

---

![Machine learning algorithm flowchart and logic](/images/blogs/pool-ml/5.jpg)

### Part 6: The Overfitting Problem

Decision Trees have a notorious weakness: they **overfit** like crazy if you do not constrain them.

An unconstrained tree will keep splitting until every leaf contains exactly one sample. It will memorize the training data perfectly and fail miserably on new data. This is because the tree has learned the noise, not the signal.

**How to prevent overfitting:**

| Technique | What It Does |
|-----------|-------------|
| `max_depth` | Limits how deep the tree can grow |
| `min_samples_split` | Requires a minimum number of samples to split a node |
| `min_samples_leaf` | Requires a minimum number of samples in each leaf |
| `max_features` | Limits the number of features considered at each split |
| **Pruning** | Grows a full tree first, then removes branches that do not improve validation performance |

**Pre-pruning** (setting constraints before training) is the most common approach in practice. **Post-pruning** (growing the full tree and trimming it) is more computationally expensive but can sometimes find better structures.

---

### Part 7: Strengths and Weaknesses

**Strengths:**
- **Interpretable:** You can explain every prediction to a non-technical stakeholder.
- **No feature scaling required:** Unlike SVMs or neural networks, trees do not care about the scale of features.
- **Handles both numerical and categorical data** (with proper encoding).
- **Fast training and prediction.**
- **Captures non-linear relationships** naturally through hierarchical splits.

**Weaknesses:**
- **Overfitting:** Without constraints, they memorize training data.
- **High variance:** Small changes in data can produce completely different trees.
- **Greedy algorithm:** CART makes locally optimal splits, which may not be globally optimal.
- **Biased toward features with many levels:** Features with more unique values get an unfair advantage.
- **Unstable:** A single outlier can dramatically change the tree structure.

---

![Ensemble methods building on individual decision trees](/images/blogs/pool-ml/7.jpg)

### Part 8: The Foundation for Something Greater

Here is the thing about Decision Trees: their weaknesses are well-known, and those weaknesses have inspired some of the most powerful algorithms in machine learning.

The high variance problem? That led to **Random Forests** — an ensemble of hundreds of trees that vote together.

The greedy, locally-optimal splitting? That led to **Gradient Boosting** — where each new tree corrects the errors of the previous one.

A single Decision Tree is a good model. An *ensemble* of Decision Trees is often the best model you can build for structured data.

---

### Final Thoughts

Decision Trees are the Swiss Army knife of machine learning. They are simple enough to explain to a business executive, powerful enough to capture complex patterns, and foundational enough to serve as the building block for Random Forests, XGBoost, and LightGBM.

If you understand how a Decision Tree works — how it chooses splits, how it overfits, and how to constrain it — you have the foundation to understand almost every tree-based algorithm that dominates data science competitions today.

In the next post, we will take the natural next step: combining hundreds of Decision Trees into a **Random Forest**.
