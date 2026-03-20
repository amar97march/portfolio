---
title: "What is an Activation Function? ReLU, Sigmoid, and Beyond"
date: 2026-03-31T10:00:00+05:30
draft: false
description: "A deep dive into activation functions — the mathematical gatekeepers that give neural networks the power to learn complex, nonlinear patterns."
tags: ["Activation Functions", "ReLU", "Sigmoid", "Deep Learning", "Neural Networks"]
categories: ["Deep Learning"]
image: "/images/blogs/pool-dl/1.jpg"
keywords: ["activation function", "ReLU", "sigmoid", "tanh", "neural network activation", "deep learning"]
---

In my previous posts on neural networks, I mentioned activation functions several times without fully explaining them. Now it is time to give them the spotlight they deserve, because activation functions are arguably the single most important ingredient that makes neural networks work.

Without activation functions, a neural network — no matter how many layers it has — would be nothing more than a fancy linear equation. And linear equations, powerful as they are, cannot learn the complex patterns that make deep learning so revolutionary.

## Why Do We Need Activation Functions?

Let me demonstrate the problem with a concrete example. Consider a two-layer network without any activation function:

```
Layer 1: output1 = W1 * input + b1
Layer 2: output2 = W2 * output1 + b2
```

If we substitute:

```
output2 = W2 * (W1 * input + b1) + b2
output2 = (W2 * W1) * input + (W2 * b1 + b2)
output2 = W_combined * input + b_combined
```

The entire two-layer network collapses into a single linear transformation. Adding more layers does not help — the composition of linear functions is always linear. You could stack a thousand layers, and the result would still be equivalent to a single layer.

This is the fundamental problem. Real-world data is **nonlinear**. The relationship between an image's pixels and whether it contains a cat is not a straight line. The relationship between a patient's symptoms and their diagnosis is not a simple linear equation.

Activation functions introduce **nonlinearity** into the network, breaking the linear chain and allowing each layer to learn something genuinely new.

![Neural network layers processing data through nonlinear activation functions](/images/blogs/pool-dl/3.jpg)

## The Classic: Sigmoid

The sigmoid function was one of the first activation functions used in neural networks, and it remains important today for specific use cases.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

**Properties:**
- Maps any input to the range (0, 1)
- S-shaped curve
- Output can be interpreted as a probability
- Smooth and differentiable everywhere

**When to use sigmoid:**
- In the **output layer** for binary classification (probability of yes/no)
- In **gates** within LSTM and GRU networks
- When you need outputs bounded between 0 and 1

**The problem with sigmoid:**
- **Vanishing gradient**: For very large or very small inputs, the gradient approaches zero. This means neurons in early layers learn extremely slowly because the error signal "vanishes" as it propagates backward.
- **Not zero-centered**: Outputs are always positive, which can cause zigzagging during gradient descent.
- **Computationally expensive**: The exponential function is relatively costly to compute.

```python
# Demonstrating the vanishing gradient problem
x_values = np.array([-10, -5, 0, 5, 10])
sigmoid_values = sigmoid(x_values)
gradients = sigmoid_values * (1 - sigmoid_values)

for x, s, g in zip(x_values, sigmoid_values, gradients):
    print(f"x={x:3d} | sigmoid={s:.6f} | gradient={g:.10f}")

# x=-10 | sigmoid=0.000045 | gradient=0.0000453979
# x= -5 | sigmoid=0.006693 | gradient=0.0066480567
# x=  0 | sigmoid=0.500000 | gradient=0.2500000000
# x=  5 | sigmoid=0.993307 | gradient=0.0066480567
# x= 10 | sigmoid=0.999955 | gradient=0.0000453979
```

Notice how the gradient at x=10 is 0.0000454 — essentially zero. If this is in an early layer of a deep network, virtually no learning signal gets through.

## Tanh: The Zero-Centered Sigmoid

The hyperbolic tangent (tanh) function is mathematically related to sigmoid but addresses one of its key weaknesses.

```python
def tanh(x):
    return np.tanh(x)
    # Equivalent to: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
```

**Properties:**
- Maps inputs to the range (-1, 1)
- S-shaped, like sigmoid
- **Zero-centered**, meaning its average output is around 0
- Stronger gradients than sigmoid (gradient ranges up to 1.0 vs. sigmoid's 0.25)

**When to use tanh:**
- In hidden layers where you want bounded, zero-centered outputs
- In RNN architectures
- When features can be naturally negative or positive

**The problem with tanh:**
- Still suffers from the vanishing gradient problem for large inputs (just less severely than sigmoid)
- Still computationally expensive due to exponentials

## ReLU: The Game Changer

The **Rectified Linear Unit** (ReLU) is deceptively simple, and it revolutionized deep learning when it became popular around 2012.

```python
def relu(x):
    return np.maximum(0, x)
```

That is it. If the input is positive, output the input unchanged. If the input is negative, output zero.

**Properties:**
- Maps inputs to the range [0, infinity)
- Not bounded above
- **Sparse activation**: many neurons output exactly zero, which creates a natural form of regularization
- Computationally trivial (just a comparison with zero)
- Gradient is either 0 (for negative inputs) or 1 (for positive inputs) — no vanishing gradient for positive values

**Why ReLU changed everything:**
- **Training speed**: Networks with ReLU converge 6x faster than networks with sigmoid, according to the original AlexNet paper
- **Simplicity**: No exponentials, no division — just `max(0, x)`
- **Gradient flow**: For positive values, the gradient is always 1, allowing error signals to flow freely through many layers

```python
import torch
import torch.nn as nn

# Modern deep network with ReLU
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),              # ReLU after each hidden layer
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
```

**The problem with ReLU — "Dying ReLU":**
- If a neuron's input is always negative, its output is always zero, and its gradient is always zero. The neuron is effectively "dead" and will never learn again.
- This can happen when learning rates are too high, causing weights to update so aggressively that a neuron gets pushed into the permanently negative zone.

![Comparing activation function curves and their gradient behavior](/images/blogs/pool-dl/5.jpg)

## Leaky ReLU: Fixing the Dying Neuron Problem

Leaky ReLU addresses the dying ReLU problem by allowing a small, non-zero gradient for negative inputs.

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
```

Instead of outputting 0 for negative inputs, it outputs a small fraction (typically 0.01) of the input. This means even "negative" neurons still have a small gradient and can potentially recover.

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.01),
    nn.Linear(256, 128),
    nn.LeakyReLU(0.01),
    nn.Linear(128, 10),
)
```

### Parametric ReLU (PReLU)

A variant where the slope for negative inputs is **learned** during training rather than being fixed:

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.PReLU(),  # The negative slope is a learnable parameter
    nn.Linear(256, 128),
    nn.PReLU(),
    nn.Linear(128, 10),
)
```

## ELU: Exponential Linear Unit

ELU uses an exponential curve for negative inputs, smoothly approaching a negative saturation value.

```python
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
```

ELU has the advantage of pushing mean activations closer to zero, which can speed up learning. However, it is slower to compute than ReLU due to the exponential operation.

## Softmax: The Multi-Class Output Function

Softmax is special — it is not typically used in hidden layers but rather in the **output layer** for multi-class classification. It converts a vector of raw scores into a probability distribution.

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)

scores = np.array([2.0, 1.0, 0.1])
probabilities = softmax(scores)
print(probabilities)
# [0.659, 0.242, 0.099]
print(f"Sum: {np.sum(probabilities):.3f}")
# Sum: 1.000
```

The key property of softmax is that all outputs are positive and they sum to exactly 1, making them interpretable as probabilities. The class with the highest probability is the prediction.

## Swish and GELU: Modern Alternatives

Recent research has produced activation functions that can outperform ReLU in certain scenarios.

**Swish** (discovered by Google in 2017):
```python
def swish(x):
    return x * sigmoid(x)
```

Swish is smooth, non-monotonic (it dips slightly below zero for negative inputs), and has been shown to outperform ReLU on deep networks.

**GELU** (Gaussian Error Linear Unit):
```python
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
```

GELU is used in modern transformer architectures like BERT and GPT. It is similar to ReLU but with a smooth curve that allows small negative values to pass through.

```python
# In PyTorch, these are readily available
model = nn.Sequential(
    nn.Linear(768, 3072),
    nn.GELU(),              # Used in BERT/GPT
    nn.Linear(3072, 768),
)
```

![Modern deep learning architectures using advanced activation functions](/images/blogs/pool-dl/7.jpg)

## Choosing the Right Activation Function

Here is my practical guide based on experience:

| Layer Type | Recommended | Why |
|---|---|---|
| Hidden layers (default) | ReLU | Fast, effective, proven |
| Hidden layers (deep nets) | Leaky ReLU or GELU | Avoids dying neurons |
| Binary output | Sigmoid | Outputs probability [0, 1] |
| Multi-class output | Softmax | Outputs probability distribution |
| Regression output | None (Linear) | Unconstrained output |
| Transformer hidden | GELU | Empirically best for NLP |
| RNN/LSTM gates | Sigmoid | Natural gate mechanism [0, 1] |
| RNN hidden state | Tanh | Zero-centered, bounded |

**My default advice**: Start with ReLU. If you encounter dying neuron problems, switch to Leaky ReLU. If you are building transformers, use GELU. Only reach for exotic activation functions if you have a specific reason.

## Key Takeaways

1. Activation functions introduce **nonlinearity**, which is essential for learning complex patterns
2. Without activation functions, any deep network collapses to a single linear transformation
3. **ReLU** is the default choice for hidden layers — it is fast, simple, and effective
4. **Sigmoid** is used for binary classification outputs and gates
5. **Softmax** is used for multi-class classification outputs
6. The vanishing gradient problem (sigmoid/tanh) and dying ReLU problem are the two main pitfalls to watch for
7. Modern alternatives like GELU and Swish offer incremental improvements in specific architectures

In the next post, I will cover **backpropagation** — the algorithm that allows neural networks to learn by flowing error signals backward through the network and adjusting weights accordingly.

---

*This post is part of my Deep Learning series. If activation functions feel abstract, do not worry — they will become much more concrete when we implement them in code tutorials later in this series.*
