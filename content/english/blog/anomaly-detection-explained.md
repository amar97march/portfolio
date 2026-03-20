---
title: "Anomaly Detection: Finding the Weird Data Point"
date: 2026-07-23T10:00:00+05:30
draft: false
description: "Anomaly detection identifies unusual patterns that do not conform to expected behavior. Learn the key techniques — statistical, distance-based, and model-based — and when to use each approach."
tags: ["Machine Learning", "Anomaly Detection", "Unsupervised Learning", "Outliers", "Python"]
categories: ["Machine Learning"]
image: "/images/blogs/pool-ml/1.jpg"
keywords: ["anomaly detection", "outlier detection", "isolation forest", "novelty detection", "unsupervised anomaly detection", "fraud detection"]
---

In most of machine learning, we focus on learning the *normal* patterns in data. But sometimes, the most valuable insights come from the data points that *do not* fit the pattern — the anomalies, outliers, and exceptions.

Anomaly detection is the art and science of finding these unusual data points. It is critical in cybersecurity (detecting intrusions), finance (detecting fraud), manufacturing (detecting defective products), and healthcare (detecting abnormal test results).

In this post, we will explore the major approaches to anomaly detection and when to use each one.

---

### Part 1: What Is an Anomaly?

An anomaly is a data point that deviates significantly from the expected pattern. There are three types:

**Point anomalies:** A single data point that is far from the rest. For example, a credit card transaction of $10,000 when the cardholder typically spends $50-$200.

**Contextual anomalies:** A data point that is normal in one context but abnormal in another. A temperature of 35 degrees Celsius is normal in July but anomalous in January (in the Northern Hemisphere).

**Collective anomalies:** A group of related data points that together form an anomalous pattern, even though each individual point might look normal. A series of small ATM withdrawals from different cities in rapid succession might individually look normal but collectively indicate fraud.

![Different types of anomalies detected in datasets including point and contextual outliers](/images/blogs/pool-ml/3.jpg)

---

### Part 2: Statistical Methods

The simplest approach is to define "normal" statistically and flag anything that deviates.

#### Z-Score Method
If the data is normally distributed, points more than 3 standard deviations from the mean are anomalies.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data with some outliers
np.random.seed(42)
normal_data = np.random.randn(1000)
outliers = np.array([4.5, -5.0, 5.5, -4.8, 6.0])
data = np.concatenate([normal_data, outliers])

# Z-score method
mean = np.mean(data)
std = np.std(data)
z_scores = np.abs((data - mean) / std)
threshold = 3

anomalies = data[z_scores > threshold]
print(f"Found {len(anomalies)} anomalies")
print(f"Anomalous values: {anomalies}")

# Visualize
plt.figure(figsize=(12, 5))
plt.hist(data, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
for a in anomalies:
    plt.axvline(x=a, color='red', linestyle='--', alpha=0.7)
plt.title('Z-Score Anomaly Detection')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
```

**Limitation:** Assumes normal distribution. Real-world data is often skewed or multimodal.

#### IQR Method
More robust to non-normal distributions. Uses the interquartile range:

```python
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

anomalies_iqr = data[(data < lower_bound) | (data > upper_bound)]
print(f"IQR anomalies: {len(anomalies_iqr)}")
```

---

### Part 3: Isolation Forest

The **Isolation Forest** is one of the most effective anomaly detection algorithms. Its key insight is elegant: **anomalies are easier to isolate than normal points.**

Imagine randomly splitting the data with binary partitions (like a Decision Tree). Normal points, which are dense and clustered, require many splits to isolate. Anomalies, which are sparse and distant, require very few splits.

The Isolation Forest builds an ensemble of random trees and measures how many splits each point needs to be isolated. Points that are isolated quickly (short path length) are anomalies.

```python
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

# Generate 2D data with outliers
X_normal, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=42)
X_outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X = np.vstack([X_normal, X_outliers])

# Fit Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # Expected proportion of anomalies
    random_state=42
)
predictions = iso_forest.fit_predict(X)
# predictions: 1 = normal, -1 = anomaly

scores = iso_forest.decision_function(X)

# Visualize
plt.figure(figsize=(10, 7))
plt.scatter(
    X[predictions == 1, 0], X[predictions == 1, 1],
    c='steelblue', label='Normal', s=30, alpha=0.7
)
plt.scatter(
    X[predictions == -1, 0], X[predictions == -1, 1],
    c='red', label='Anomaly', s=60, marker='x', linewidths=2
)
plt.title('Isolation Forest Anomaly Detection')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Detected {sum(predictions == -1)} anomalies out of {len(X)} points")
```

**Key parameters:**
- `contamination`: The expected proportion of anomalies. This is the most important parameter. If unknown, start with 0.01-0.05.
- `n_estimators`: Number of trees. More is generally better.

---

### Part 4: Local Outlier Factor (LOF)

LOF detects anomalies based on **local density**. A point is an anomaly if its local density is significantly lower than the density of its neighbors.

This is powerful because it can detect anomalies in clusters of varying density — something global methods like Z-scores miss.

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05
)
predictions = lof.fit_predict(X)
scores = lof.negative_outlier_factor_

plt.figure(figsize=(10, 7))
plt.scatter(
    X[predictions == 1, 0], X[predictions == 1, 1],
    c='steelblue', label='Normal', s=30, alpha=0.7
)
plt.scatter(
    X[predictions == -1, 0], X[predictions == -1, 1],
    c='red', label='Anomaly', s=60, marker='x', linewidths=2
)
plt.title('Local Outlier Factor (LOF)')
plt.legend()
plt.tight_layout()
plt.show()
```

![Local outlier factor detecting anomalies in clusters of varying density](/images/blogs/pool-ml/5.jpg)

---

### Part 5: One-Class SVM

The One-Class SVM learns a boundary around the "normal" data and classifies anything outside this boundary as an anomaly:

```python
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

oc_svm = OneClassSVM(
    kernel='rbf',
    gamma='scale',
    nu=0.05  # Similar to contamination
)
predictions = oc_svm.fit_predict(X_scaled)

print(f"One-Class SVM detected {sum(predictions == -1)} anomalies")
```

---

### Part 6: Autoencoder-Based Anomaly Detection

For high-dimensional data, autoencoders provide a deep learning approach. An autoencoder is trained to reconstruct normal data. When it encounters an anomaly, the reconstruction error is high.

```python
# Conceptual example (requires tensorflow/keras)
# from tensorflow import keras
#
# # Build autoencoder
# encoder = keras.Sequential([
#     keras.layers.Dense(32, activation='relu'),
#     keras.layers.Dense(16, activation='relu'),
#     keras.layers.Dense(8, activation='relu'),   # Bottleneck
# ])
#
# decoder = keras.Sequential([
#     keras.layers.Dense(16, activation='relu'),
#     keras.layers.Dense(32, activation='relu'),
#     keras.layers.Dense(n_features, activation='linear'),
# ])
#
# autoencoder = keras.Sequential([encoder, decoder])
# autoencoder.compile(optimizer='adam', loss='mse')
#
# # Train on normal data only
# autoencoder.fit(X_train_normal, X_train_normal, epochs=50)
#
# # Detect anomalies by reconstruction error
# reconstructions = autoencoder.predict(X_test)
# errors = np.mean((X_test - reconstructions) ** 2, axis=1)
# threshold = np.percentile(errors, 95)
# anomalies = errors > threshold
```

The idea is simple: train the model to compress and reconstruct normal patterns. Anything it cannot reconstruct well is unusual.

![Autoencoder architecture used for high-dimensional anomaly detection](/images/blogs/pool-ml/7.jpg)

---

### Part 7: Comparing Methods

| Method | Best For | Handles Multivariate | Scalability |
|--------|----------|---------------------|-------------|
| **Z-Score / IQR** | Univariate, quick checks | No (per-feature only) | Excellent |
| **Isolation Forest** | General purpose, tabular data | Yes | Excellent |
| **LOF** | Varying density clusters | Yes | Moderate |
| **One-Class SVM** | High-dimensional, clear boundary | Yes | Moderate |
| **Autoencoder** | Very high-dimensional, complex patterns | Yes | GPU-dependent |

---

### Part 8: Practical Considerations

**The contamination parameter is critical.** You are essentially telling the algorithm how many anomalies to expect. If you set it too high, you will get too many false positives. Too low, and you will miss real anomalies. When in doubt, err on the side of catching more anomalies and filter manually.

**Evaluation is hard.** In most real-world scenarios, you do not have labeled anomalies. You must rely on domain expertise, manual review, and feedback loops to evaluate your model.

**Feature engineering matters.** Anomaly detection on raw features often produces poor results. Engineered features — ratios, rolling averages, deviations from typical behavior — are often more effective.

**Combine methods.** No single method catches everything. In production, it is common to use an ensemble: if multiple methods flag a point, it is more likely a true anomaly.

---

### Final Thoughts

Anomaly detection is one of the most practically valuable applications of unsupervised learning. Whether you are detecting fraud, equipment failure, or security threats, the ability to find what does not belong is an essential skill.

The key is choosing the right method for your data and understanding the tradeoffs between sensitivity and specificity.

In the next post, we will take a broader view: **The Most Valuable Use Cases for Unsupervised Learning**.
