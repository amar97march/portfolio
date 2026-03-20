---
title: "Data Preprocessing: Normalization vs. Standardization Explained"
date: 2026-05-15T10:00:00+05:30
draft: false
description: "Understand the difference between normalization and standardization — when to use each, why they matter, and how they affect model performance."
tags: ["Normalization", "Standardization", "Preprocessing", "Feature Scaling", "Data Science"]
categories: ["Data Science"]
image: "/images/blogs/pool-data/1.jpg"
keywords: ["normalization vs standardization", "feature scaling", "min max scaling", "standard scaler", "data preprocessing"]
---

One of the most common preprocessing steps in machine learning is **feature scaling** — transforming your numerical features so they are on a similar scale. Without scaling, features with large values (like salary in dollars) can dominate features with small values (like age in years), leading to poor model performance.

The two most common scaling techniques are **normalization** (Min-Max scaling) and **standardization** (Z-score scaling). They sound similar, and they are often confused, but they work differently and are suited to different situations.

## Why Feature Scaling Matters

Consider a dataset with two features: **age** (range: 18-70) and **salary** (range: 30,000-200,000). If you use a distance-based algorithm like K-Nearest Neighbors, the salary feature will completely dominate the distance calculation simply because its numbers are larger, not because it is more important.

```python
import numpy as np

# Distance without scaling
person_a = np.array([25, 50000])   # Age 25, Salary $50,000
person_b = np.array([60, 52000])   # Age 60, Salary $52,000
person_c = np.array([26, 120000])  # Age 26, Salary $120,000

dist_ab = np.sqrt(np.sum((person_a - person_b) ** 2))
dist_ac = np.sqrt(np.sum((person_a - person_c) ** 2))

print(f"Distance A to B: {dist_ab:.0f}")   # ~2,000 (dominated by salary)
print(f"Distance A to C: {dist_ac:.0f}")   # ~70,000 (dominated by salary)

# Person B is "closer" to A despite a 35-year age difference,
# because the $2,000 salary difference is tiny compared to scales
```

Age is effectively invisible in this calculation. Scaling fixes this.

## Normalization (Min-Max Scaling)

Normalization scales each feature to a fixed range, typically [0, 1].

**Formula:**
```
X_normalized = (X - X_min) / (X_max - X_min)
```

The minimum value becomes 0, the maximum becomes 1, and everything else is proportionally scaled in between.

```python
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Sample data
data = pd.DataFrame({
    'age': [25, 30, 35, 45, 60],
    'salary': [30000, 55000, 72000, 95000, 150000],
    'experience': [2, 5, 8, 15, 30]
})

# Apply Min-Max scaling
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(
    scaler.fit_transform(data),
    columns=data.columns
)

print("Original:")
print(data)
print("\nNormalized (0-1):")
print(data_normalized.round(3))

# Custom range [0, 10]
scaler = MinMaxScaler(feature_range=(0, 10))
data_custom = scaler.fit_transform(data)
```

### Properties of Normalization

- Output is bounded to [0, 1] (or your custom range)
- Preserves the shape of the original distribution
- Does NOT center the data (mean is not 0)
- **Sensitive to outliers**: A single extreme value compresses all other values into a narrow range

```python
# Outlier sensitivity demonstration
values = np.array([10, 12, 14, 15, 13, 11, 100])  # 100 is an outlier

# Min-Max
normalized = (values - values.min()) / (values.max() - values.min())
print(normalized)
# [0.0, 0.022, 0.044, 0.056, 0.033, 0.011, 1.0]
# All non-outlier values are compressed between 0 and 0.056!
```

### When to Use Normalization

- When you need features in a bounded range (e.g., neural network inputs)
- When the data does NOT have significant outliers
- Image pixel values (already bounded 0-255, scale to 0-1)
- When using algorithms sensitive to magnitude: KNN, neural networks, SVMs with RBF kernel
- When the data distribution is NOT Gaussian


![Data preprocessing and transformation workflow](/images/blogs/pool-data/3.jpg)

## Standardization (Z-Score Scaling)

Standardization transforms features to have a **mean of 0** and a **standard deviation of 1**.

**Formula:**
```
X_standardized = (X - mean) / std
```

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_standardized = pd.DataFrame(
    scaler.fit_transform(data),
    columns=data.columns
)

print("Standardized:")
print(data_standardized.round(3))

# Verify: mean ≈ 0, std ≈ 1
print(f"\nMeans: {data_standardized.mean().round(6).values}")
print(f"Stds:  {data_standardized.std().round(3).values}")
```

### Properties of Standardization

- Output is unbounded (can be any real number)
- Centers the data (mean = 0)
- Scales by variance (std = 1)
- **Less sensitive to outliers** than normalization (outliers are not compressed)
- Does NOT change the shape of the distribution

### When to Use Standardization

- When your data has outliers
- When the features have different units or scales
- When using algorithms that assume normally distributed features
- Linear regression, logistic regression, SVM, PCA
- When you want zero-centered data

## Head-to-Head Comparison

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

# Generate sample data with different distributions
np.random.seed(42)
normal_data = np.random.normal(50, 15, 1000)
skewed_data = np.random.exponential(50, 1000)
outlier_data = np.concatenate([np.random.normal(50, 10, 990), [200, 250, 300, 350, 400, -100, -150, -200, -250, -300]])

datasets = {
    'Normal': normal_data.reshape(-1, 1),
    'Skewed': skewed_data.reshape(-1, 1),
    'With Outliers': outlier_data.reshape(-1, 1)
}

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for row, (name, data) in enumerate(datasets.items()):
    # Original
    axes[row, 0].hist(data, bins=50, alpha=0.7)
    axes[row, 0].set_title(f'{name} - Original')

    # Normalized
    normalized = MinMaxScaler().fit_transform(data)
    axes[row, 1].hist(normalized, bins=50, alpha=0.7, color='green')
    axes[row, 1].set_title(f'{name} - Normalized')

    # Standardized
    standardized = StandardScaler().fit_transform(data)
    axes[row, 2].hist(standardized, bins=50, alpha=0.7, color='orange')
    axes[row, 2].set_title(f'{name} - Standardized')

plt.tight_layout()
plt.savefig('scaling_comparison.png')
```


![Feature engineering and data normalization](/images/blogs/pool-data/4.jpg)

## Other Scaling Methods

### Robust Scaler

Uses the median and interquartile range instead of mean and standard deviation. Excellent for data with outliers:

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
data_robust = scaler.fit_transform(data)
# X_robust = (X - median) / IQR
```

### Max Abs Scaler

Scales by the maximum absolute value. Useful for sparse data:

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
data_maxabs = scaler.fit_transform(data)
# X_scaled = X / max(|X|)
```

### Log Transformation

For heavily right-skewed data:

```python
df['log_salary'] = np.log1p(df['salary'])  # log1p handles zero values
```

## Critical Rule: Fit on Training Data Only

One of the most common mistakes in ML preprocessing is **data leakage** through scaling. You must fit the scaler on training data only, then transform both training and test data using the same scaler.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()

# FIT on training data, TRANSFORM training data
X_train_scaled = scaler.fit_transform(X_train)

# Only TRANSFORM test data (using training data's mean and std)
X_test_scaled = scaler.transform(X_test)

# WRONG: Never do this
# X_test_scaled = scaler.fit_transform(X_test)  # Data leakage!
```


![Clean structured data ready for machine learning](/images/blogs/pool-data/5.jpg)

## Quick Reference Guide

| Scenario | Recommended Scaler |
|---|---|
| Neural networks | MinMaxScaler (0-1) |
| Data with outliers | RobustScaler |
| Gaussian-like data, linear models | StandardScaler |
| Sparse data (many zeros) | MaxAbsScaler |
| Heavily skewed data | Log transform + StandardScaler |
| KNN, SVM (RBF kernel) | StandardScaler or MinMaxScaler |
| Tree-based models (RF, XGBoost) | **No scaling needed** |
| PCA, LDA | StandardScaler |

That last row is important: **tree-based models are scale-invariant**. Decision trees split on thresholds, so the absolute scale of features does not matter. Random forests, gradient boosting (XGBoost, LightGBM), and decision trees do not require scaling.

## Key Takeaways

1. **Normalization** (Min-Max) scales to [0, 1] — use for bounded outputs and neural networks
2. **Standardization** (Z-score) scales to mean=0, std=1 — use for linear models and data with outliers
3. Always **fit on training data only** and transform both train and test
4. Tree-based models do not need scaling
5. Use RobustScaler when outliers are present
6. Consider log transforms for heavily skewed distributions
7. The choice of scaling can significantly impact model performance — experiment and compare

---

*Next: Data Augmentation — how to create synthetic data when you do not have enough.*
