---
title: "What is a Neural Network? A Simple Model of the Brain"
date: 2026-03-25T10:00:00+05:30
draft: false
description: "A beginner-friendly explanation of neural networks, drawing parallels with the human brain and breaking down the math behind artificial neurons."
tags: ["Neural Networks", "Deep Learning", "AI Basics", "Perceptron"]
categories: ["Deep Learning"]
image: "https://picsum.photos/seed/what-is-a-neural-network-cover/1200/630"
keywords: ["what is a neural network", "neural network explained", "artificial neurons", "perceptron", "deep learning basics"]
---

If you have spent any time reading about artificial intelligence, you have almost certainly encountered the term "neural network." It is one of the most powerful ideas in modern computing, and it sits at the heart of everything from voice assistants to self-driving cars. But what actually is a neural network?

The name gives us a clue. A neural network is, at its core, a computational system **inspired by the human brain**. Not a replica of the brain — that is an important distinction — but a mathematical model that borrows some of the brain's architectural principles and applies them to data processing.

Let me break this down in a way that makes the concept genuinely intuitive.

## The Biological Inspiration

Your brain contains roughly 86 billion neurons. Each neuron is a cell that receives electrical signals from other neurons, processes those signals, and either fires its own signal or stays quiet. The decision to fire depends on whether the combined input signals exceed a certain threshold.

Neurons are connected to each other through **synapses** — tiny gaps where chemical signals pass from one neuron to the next. Some connections are strong (meaning the signal passes easily), and some are weak. **Learning**, at a biological level, is largely about strengthening and weakening these connections.

An artificial neural network mirrors this structure with three key components:

- **Artificial neurons** (also called nodes or units) that receive inputs and produce outputs
- **Weights** on the connections between neurons (analogous to synapse strength)
- **An activation function** that determines whether a neuron "fires"

## The Simplest Neural Network: A Single Neuron

The simplest possible neural network is a single artificial neuron, historically called a **Perceptron**. It was invented by Frank Rosenblatt in 1958, and despite its simplicity, it contains all the fundamental ideas that scale up to modern deep learning.

Here is how a single neuron works:

1. It receives one or more **inputs** (let us call them x1, x2, x3, ...)
2. Each input is multiplied by a **weight** (w1, w2, w3, ...)
3. The weighted inputs are **summed** together
4. A **bias** term is added
5. The result passes through an **activation function**
6. The output is produced

Mathematically:

```
output = activation(w1*x1 + w2*x2 + w3*x3 + ... + bias)
```

Let me show this in Python:

```python
import numpy as np

def simple_neuron(inputs, weights, bias):
    # Step 1: Weighted sum
    weighted_sum = np.dot(inputs, weights) + bias

    # Step 2: Activation function (using sigmoid)
    output = 1 / (1 + np.exp(-weighted_sum))

    return output

# Example: Should I go for a run?
# Inputs: [weather_is_nice, I_have_energy, I_have_time]
inputs = np.array([1.0, 0.7, 0.9])

# Weights: how important each factor is
weights = np.array([0.5, 0.8, 0.3])

# Bias: general tendency
bias = -0.5

result = simple_neuron(inputs, weights, bias)
print(f"Probability of going for a run: {result:.2f}")
# Output: Probability of going for a run: 0.75
```

In this toy example, the neuron takes three inputs (how nice the weather is, how much energy you have, how much time you have), weights them by importance, and produces a probability. The weight on "energy" is highest (0.8), meaning the neuron considers energy level the most important factor.

## What Are Weights, Really?

Weights are the **knowledge** of a neural network. They encode everything the network has learned. When we say a neural network is "learning," what we really mean is that it is **adjusting its weights** to produce better outputs.

Think of it like this. Imagine you are trying to predict whether a student will pass an exam based on three factors: hours studied, hours slept, and number of practice tests taken. Initially, you might weight all three factors equally. But after seeing data from hundreds of students, you discover that hours studied matters most, sleep matters moderately, and practice tests matter least. Your brain has effectively "learned" the right weights.

A neural network does the same thing, but automatically and at massive scale.

## What Is the Bias?

The bias term is a constant that gives the neuron flexibility. Without it, the neuron can only learn patterns that pass through the origin point (zero). With it, the neuron can shift its decision boundary.

A simple analogy: imagine a thermostat. The bias is like the baseline temperature setting. The inputs and weights adjust the temperature up or down from that baseline. Without the baseline, you would always be adjusting from zero degrees, which is not very useful.

![How a single artificial neuron processes inputs and produces output](https://picsum.photos/seed/what-is-a-neural-network-1/800/450)

## From One Neuron to Many: The Network

A single neuron can only learn simple, linear patterns. The real power emerges when you connect many neurons together into a **network**. This is where the "network" in "neural network" comes from.

A typical neural network has three types of layers:

1. **Input Layer**: Receives the raw data (pixel values, word embeddings, sensor readings)
2. **Hidden Layer(s)**: Processes the data through multiple neurons. Each neuron learns different features
3. **Output Layer**: Produces the final prediction

When a network has multiple hidden layers, we call it a **deep neural network**, and training such networks is called **deep learning**.

```python
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self):
        # Random initial weights for a 3-input, 4-hidden, 1-output network
        self.weights_hidden = np.random.randn(3, 4)
        self.bias_hidden = np.zeros(4)
        self.weights_output = np.random.randn(4, 1)
        self.bias_output = np.zeros(1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        # Hidden layer
        hidden = self.sigmoid(
            np.dot(inputs, self.weights_hidden) + self.bias_hidden
        )

        # Output layer
        output = self.sigmoid(
            np.dot(hidden, self.weights_output) + self.bias_output
        )

        return output

nn = SimpleNeuralNetwork()
sample_input = np.array([0.5, 0.8, 0.2])
result = nn.forward(sample_input)
print(f"Network output: {result[0]:.4f}")
```

This code creates a neural network with 3 inputs, 4 hidden neurons, and 1 output. The `forward` method shows how data flows through the network — from input to hidden layer to output. This forward flow of data is called **forward propagation**.

## Why Multiple Layers Matter

A single neuron can only draw a straight line to separate data. But real-world problems are rarely that simple. Consider trying to classify images of cats versus dogs. No single straight line in pixel space can separate all cats from all dogs.

Multiple layers solve this by learning **hierarchical features**:

- The first hidden layer might learn to detect edges and simple textures
- The second hidden layer might combine edges into shapes (ears, noses, tails)
- The third hidden layer might combine shapes into high-level concepts (cat face, dog snout)

Each layer builds on the abstractions learned by the previous layer. This hierarchical feature learning is what makes neural networks so powerful and so different from traditional algorithms.

![Layers of neurons learning hierarchical features](https://picsum.photos/seed/what-is-a-neural-network-2/800/450)

## How Does a Neural Network Learn?

The learning process works like this:

1. **Forward pass**: Data flows through the network, producing a prediction
2. **Loss calculation**: The prediction is compared to the correct answer using a **loss function** (e.g., mean squared error)
3. **Backward pass**: The error is propagated backward through the network using **backpropagation** (which I will cover in a dedicated post)
4. **Weight update**: Each weight is adjusted slightly to reduce the error
5. **Repeat**: Steps 1-4 are repeated thousands or millions of times

This process is called **gradient descent**, and it is the engine that drives all neural network training. The network starts with random weights (knowing nothing) and gradually adjusts them until it can make accurate predictions.

```python
# Conceptual training loop
for epoch in range(1000):
    # Forward pass
    predictions = network.forward(training_data)

    # Calculate loss
    loss = calculate_loss(predictions, true_labels)

    # Backward pass (compute gradients)
    gradients = compute_gradients(loss)

    # Update weights
    network.update_weights(gradients, learning_rate=0.01)
```

![Neural network training through gradient descent and backpropagation](https://picsum.photos/seed/what-is-a-neural-network-3/800/450)

## The Power and the Limitations

Neural networks are extraordinarily powerful. They can:

- Recognize images with superhuman accuracy
- Translate between languages in real time
- Generate realistic text, images, and audio
- Play complex games better than any human
- Predict protein structures that took scientists decades to determine

But they also have significant limitations:

- **They require massive amounts of data.** A child can learn to recognize a cat from a few examples. A neural network typically needs thousands.
- **They are computationally expensive.** Training large models requires specialized hardware (GPUs, TPUs) and significant energy.
- **They are black boxes.** It is often very difficult to understand why a neural network made a particular decision.
- **They can be brittle.** Small, carefully crafted changes to an input (called adversarial examples) can fool a network completely.

## A Mental Model to Carry Forward

Here is the mental model I want you to carry with you:

A neural network is a **function approximator**. Given enough neurons and enough data, it can learn to approximate any mathematical function — mapping inputs to outputs. When you feed it images and labels, it learns the function that maps pixels to categories. When you feed it English and French text, it learns the function that maps one language to another.

The neurons are the building blocks. The weights are the knowledge. The training process is how the knowledge is acquired. And the layers allow the network to learn increasingly abstract representations of the data.

That is the essence of a neural network. Everything else — CNNs, RNNs, Transformers, GANs — are architectural variations on this fundamental idea.

## What is Next

In the upcoming posts, I will dive deeper into the building blocks of neural networks. We will explore the different types of layers, understand activation functions in detail, and walk through backpropagation step by step. Each piece builds on what we have covered here.

---

*This post is part of my series on AI and Machine Learning. If you are new to the series, I recommend starting with my posts on supervised learning and how we measure AI performance before diving into neural networks.*
