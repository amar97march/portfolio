---
title: "Use Case: Image Compression with PCA"
date: 2026-07-20T10:00:00+05:30
draft: false
description: "PCA can compress images by keeping only the most important components. This hands-on tutorial shows how to compress and reconstruct images using Principal Component Analysis in Python."
tags: ["Machine Learning", "PCA", "Image Processing", "Dimensionality Reduction", "Python"]
categories: ["Machine Learning"]
image: "https://images.unsplash.com/photo-1527066236128-2f274b14d05f?w=1200&h=630&fit=crop&auto=format"
keywords: ["PCA image compression", "image compression machine learning", "dimensionality reduction images", "principal component analysis tutorial", "image reconstruction"]
---

In the previous post, we learned how PCA finds the directions of maximum variance in data. Now, let us see this in action with something visual and tangible: **compressing images**.

An image is just a matrix of numbers. A grayscale image of size 512x512 has 262,144 pixel values. PCA can represent this image with far fewer numbers while still looking recognizably similar to the original.

This is not just an academic exercise. Understanding how PCA works on images builds intuition that transfers to all dimensionality reduction problems.

---

### Part 1: How Images Become Data

A grayscale image is a 2D matrix where each value represents a pixel's brightness (0=black, 255=white). For PCA, we treat each row of the image as a separate data point with as many features as there are columns.

For a 512x512 image:
- **512 data points** (rows)
- **512 features** (columns)

PCA finds the principal components across these rows and reconstructs the image using only the top $k$ components.

---

### Part 2: Step-by-Step Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces

# Load the Olivetti faces dataset (64x64 grayscale faces)
faces = fetch_olivetti_faces()
face = faces.images[0]  # Pick one face

print(f"Image shape: {face.shape}")
print(f"Total pixel values: {face.size}")

plt.figure(figsize=(6, 6))
plt.imshow(face, cmap='gray')
plt.title('Original Image (64x64)')
plt.axis('off')
plt.tight_layout()
plt.show()
```

---


![Visual representation of machine learning model training and optimization](https://picsum.photos/seed/image-compression-with-pca-1/800/450)

### Part 3: Applying PCA with Different Numbers of Components

```python
def compress_image(image, n_components):
    """Compress an image using PCA with n_components."""
    pca = PCA(n_components=n_components)

    # Fit and transform (compress)
    compressed = pca.fit_transform(image)

    # Inverse transform (reconstruct)
    reconstructed = pca.inverse_transform(compressed)

    # Calculate compression ratio
    original_size = image.shape[0] * image.shape[1]
    compressed_size = (
        compressed.shape[0] * compressed.shape[1]  # Scores
        + pca.components_.shape[0] * pca.components_.shape[1]  # Components
        + pca.mean_.shape[0]  # Mean
    )
    ratio = original_size / compressed_size

    variance_explained = pca.explained_variance_ratio_.sum()

    return reconstructed, ratio, variance_explained


# Test different numbers of components
n_components_list = [1, 5, 10, 20, 30, 50]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

# Original
axes[0].imshow(face, cmap='gray')
axes[0].set_title(f'Original\n{face.shape[0]*face.shape[1]} values')
axes[0].axis('off')

# Reconstructions
for i, n in enumerate(n_components_list):
    reconstructed, ratio, var_exp = compress_image(face, n)

    axes[i + 1].imshow(reconstructed, cmap='gray')
    axes[i + 1].set_title(f'{n} Components\nVariance: {var_exp:.1%}\nCompression: {ratio:.1f}x')
    axes[i + 1].axis('off')

# Hide unused subplot
axes[-1].axis('off')

plt.suptitle('Image Compression with PCA', fontsize=16)
plt.tight_layout()
plt.savefig("pca_compression.png", dpi=150)
plt.show()
```

You will see a clear progression:
- **1 component:** A blurry blob — just the average brightness pattern.
- **5 components:** Basic face shape is visible but details are lost.
- **10 components:** Clearly recognizable as a specific person.
- **20 components:** Most features are captured with minor artifacts.
- **50 components:** Nearly indistinguishable from the original.

---

### Part 4: The Compression-Quality Tradeoff

```python
components_range = range(1, face.shape[1] + 1)
variances = []
errors = []

for n in components_range:
    pca = PCA(n_components=n)
    compressed = pca.fit_transform(face)
    reconstructed = pca.inverse_transform(compressed)

    variances.append(pca.explained_variance_ratio_.sum())
    errors.append(np.mean((face - reconstructed) ** 2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(components_range, variances, 'b-')
ax1.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
ax1.set_xlabel('Number of Components')
ax1.set_ylabel('Cumulative Explained Variance')
ax1.set_title('Variance Retained vs. Components')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(components_range, errors, 'r-')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Mean Squared Error')
ax2.set_title('Reconstruction Error vs. Components')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("compression_tradeoff.png", dpi=150)
plt.show()

# Find the "sweet spot"
for threshold in [0.90, 0.95, 0.99]:
    n_needed = np.argmax(np.array(variances) >= threshold) + 1
    print(f"{threshold:.0%} variance retained with {n_needed} components "
          f"({n_needed/face.shape[1]:.1%} of original)")
```

---


![Data flowing through a machine learning pipeline illustration](https://picsum.photos/seed/image-compression-with-pca-2/800/450)

### Part 5: Compressing Color Images

Color images have three channels (Red, Green, Blue). We apply PCA to each channel independently:

```python
from PIL import Image
import requests
from io import BytesIO

def compress_color_image(image_array, n_components):
    """Compress a color image by applying PCA to each channel."""
    reconstructed_channels = []

    for channel in range(3):
        channel_data = image_array[:, :, channel].astype(float)
        pca = PCA(n_components=n_components)
        compressed = pca.fit_transform(channel_data)
        reconstructed = pca.inverse_transform(compressed)
        reconstructed = np.clip(reconstructed, 0, 255)
        reconstructed_channels.append(reconstructed)

    return np.stack(reconstructed_channels, axis=2).astype(np.uint8)


# Create a sample color image (gradient + pattern)
np.random.seed(42)
height, width = 128, 128
r = np.tile(np.linspace(0, 255, width), (height, 1))
g = np.tile(np.linspace(0, 255, height), (width, 1)).T
b = np.random.randint(50, 200, (height, width))
sample_image = np.stack([r, g, b], axis=2).astype(np.uint8)

# Compress at different levels
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(sample_image)
axes[0].set_title('Original')
axes[0].axis('off')

for i, n in enumerate([5, 20, 50]):
    compressed = compress_color_image(sample_image, n)
    axes[i + 1].imshow(compressed)
    axes[i + 1].set_title(f'{n} Components')
    axes[i + 1].axis('off')

plt.suptitle('Color Image Compression with PCA', fontsize=16)
plt.tight_layout()
plt.savefig("color_compression.png", dpi=150)
plt.show()
```

---

### Part 6: Understanding What PCA Captures

Each principal component captures a specific pattern in the image. The first few components capture large-scale structures (overall brightness, broad shapes). Later components capture finer details (edges, textures, noise).

```python
# Visualize the principal components themselves
pca = PCA(n_components=10)
pca.fit(face)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i in range(10):
    axes[i].imshow(pca.components_[i].reshape(1, -1).repeat(20, axis=0), cmap='RdBu')
    axes[i].set_title(f'PC{i+1}\n({pca.explained_variance_ratio_[i]:.1%})')
    axes[i].axis('off')

plt.suptitle('Top 10 Principal Components (Patterns)', fontsize=14)
plt.tight_layout()
plt.savefig("principal_components_patterns.png", dpi=150)
plt.show()
```

---


![Visualization of algorithm performance and evaluation metrics](https://picsum.photos/seed/image-compression-with-pca-3/800/450)

### Part 7: PCA vs. Modern Compression

PCA-based compression is educational but not competitive with modern image compression algorithms. JPEG, PNG, and WebP are specifically designed for images and achieve much better compression ratios.

However, PCA teaches fundamental concepts that appear throughout machine learning:

- **Low-rank approximation:** Many real-world data matrices are approximately low-rank, meaning a few components capture most of the information.
- **The bias-variance tradeoff:** More components = more detail (less bias) but also more noise (more variance).
- **Lossy compression:** You trade some information for a more compact representation.
- **Eigenfaces:** PCA applied to face images is the basis of the classic Eigenfaces algorithm for face recognition.

---

### Part 8: The Eigenfaces Connection

When you apply PCA to a dataset of faces, the principal components are called **Eigenfaces**. These are the most common face patterns — the "building blocks" of facial variation.

```python
# Apply PCA to the entire Olivetti faces dataset
all_faces = faces.images.reshape(400, -1)  # 400 faces, each 64x64 = 4096 pixels

pca_faces = PCA(n_components=50)
pca_faces.fit(all_faces)

# Visualize the top eigenfaces
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i in range(10):
    axes[i].imshow(pca_faces.components_[i].reshape(64, 64), cmap='gray')
    axes[i].set_title(f'Eigenface {i+1}\n({pca_faces.explained_variance_ratio_[i]:.1%})')
    axes[i].axis('off')

plt.suptitle('Top 10 Eigenfaces', fontsize=14)
plt.tight_layout()
plt.savefig("eigenfaces.png", dpi=150)
plt.show()
```

Any face in the dataset can be reconstructed as a weighted sum of these eigenfaces. This is the same principle that powers face recognition systems.

---

### Final Thoughts

Image compression with PCA is a powerful demonstration of dimensionality reduction in action. It makes abstract concepts concrete: you can literally *see* information being retained and lost as you vary the number of components.

The principles at work here — low-rank approximation, compression-quality tradeoffs, and the idea that most data lies on a lower-dimensional manifold — are foundational to modern machine learning.

In the next post, we explore another unsupervised technique: **Anomaly Detection — Finding the Weird Data Point**.
