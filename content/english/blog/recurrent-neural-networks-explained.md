---
title: "What is a Recurrent Neural Network (RNN)? The Original Memory Network"
date: 2026-04-09T10:00:00+05:30
draft: false
description: "An in-depth look at Recurrent Neural Networks — the architecture that gave neural networks memory, enabling them to process sequences like text, speech, and time series."
tags: ["RNN", "LSTM", "Deep Learning", "Sequence Modeling", "NLP"]
categories: ["Deep Learning"]
image: "https://picsum.photos/seed/recurrent-neural-networks-explained-cover/1200/630"
keywords: ["recurrent neural network", "RNN explained", "LSTM", "GRU", "sequence modeling", "NLP deep learning"]
---

In the previous post, we explored Convolutional Neural Networks, which excel at processing spatial data like images. But what about data that unfolds over **time**? Text, speech, music, stock prices, sensor readings — these are all **sequences**, where the order of the data matters enormously.

The sentence "the dog chased the cat" means something entirely different from "the cat chased the dog," even though they contain the same words. A standard feedforward network processes each input independently and has no notion of order. It cannot understand sequences.

This is the problem that **Recurrent Neural Networks (RNNs)** were designed to solve.

## The Core Idea: Memory Through Loops

The fundamental innovation of an RNN is a **feedback loop**. Unlike feedforward networks where data flows in one direction (input to output), an RNN passes information from one time step to the next through a **hidden state** — a form of memory.

At each time step, the RNN:
1. Takes the current input
2. Combines it with the previous hidden state (memory)
3. Produces an output and a new hidden state
4. Passes the new hidden state to the next time step

```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(hidden_size, output_size) * 0.01
        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(output_size)

    def forward(self, inputs, h_prev):
        """Process a sequence of inputs"""
        outputs = []
        h = h_prev

        for x in inputs:
            # Combine current input with previous hidden state
            h = np.tanh(np.dot(x, self.Wxh) + np.dot(h, self.Whh) + self.bh)
            y = np.dot(h, self.Why) + self.by
            outputs.append(y)

        return outputs, h

# Example: processing a sequence of 5 time steps
rnn = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
sequence = [np.random.randn(10) for _ in range(5)]
h_init = np.zeros(20)

outputs, final_hidden = rnn.forward(sequence, h_init)
print(f"Processed {len(outputs)} time steps")
print(f"Hidden state carries memory forward: shape = {final_hidden.shape}")
```

The key equation at each time step is:

```
h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
```

This is elegant in its simplicity: the new hidden state `h_t` depends on both the current input `x_t` and the previous hidden state `h_{t-1}`. The hidden state is the network's **memory**, carrying information from all previous time steps.


![Illustration of gradient flow and model training dynamics](https://picsum.photos/seed/recurrent-neural-networks-explained-1/800/450)

## Unrolling Through Time

When we visualize an RNN, we often "unroll" it through time. Imagine making a copy of the RNN for each time step and connecting them in a chain:

```
Input: x1    x2    x3    x4    x5
        |     |     |     |     |
       [RNN]-[RNN]-[RNN]-[RNN]-[RNN]
        |     |     |     |     |
Output: y1    y2    y3    y4    y5
```

Each box is the **same** RNN cell with the **same** weights, applied at different time steps. The horizontal arrows represent the hidden state flowing from one step to the next.

## RNN Applications

### Many-to-One: Sentiment Analysis
Process a sequence of words, produce a single output (positive/negative).

```python
import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        # Use only the last hidden state for classification
        last_hidden = hidden.squeeze(0)
        return torch.sigmoid(self.fc(last_hidden))
```

### One-to-Many: Text Generation
Take a single input (a starting word) and generate a sequence of outputs.

### Many-to-Many: Machine Translation
Process an input sequence (English sentence) and produce an output sequence (French sentence). This is the famous **encoder-decoder** architecture.

### Sequence-to-Sequence with Attention
The precursor to transformers, where the decoder can "attend" to different parts of the input sequence at each step.

## The Vanishing Gradient Problem

Standard RNNs have a critical flaw: they struggle to learn **long-range dependencies**. If the relevant context is far back in the sequence — say, 50 or 100 steps ago — the gradient signal becomes vanishingly small by the time it propagates back through all those time steps.

This is the vanishing gradient problem again, but made worse by the sequential nature of RNNs. The gradient is multiplied by the same weight matrix at each step, and if that matrix has eigenvalues less than 1, the gradient shrinks exponentially.

In practice, this means a standard RNN can effectively "remember" only about 10-20 time steps back. For many real-world tasks, that is not enough.

## LSTM: Long Short-Term Memory

The **Long Short-Term Memory** (LSTM) network, introduced by Hochreiter and Schmidhuber in 1997, is the most successful solution to the vanishing gradient problem in RNNs.


![Diagram showing neural network layers and data transformation](https://picsum.photos/seed/recurrent-neural-networks-explained-2/800/450)

The key innovation is a **cell state** — a separate memory pathway that runs through the entire sequence with minimal modification. Information can be added to or removed from the cell state through **gates**.

An LSTM has three gates:

1. **Forget Gate**: Decides what information to discard from the cell state
2. **Input Gate**: Decides what new information to add to the cell state
3. **Output Gate**: Decides what information from the cell state to output

```python
class LSTMFromScratch:
    """Simplified LSTM to show the gate mechanism"""

    def step(self, x, h_prev, c_prev):
        # Concatenate input and previous hidden state
        combined = np.concatenate([x, h_prev])

        # Forget gate: what to remove from memory
        f = sigmoid(np.dot(self.Wf, combined) + self.bf)

        # Input gate: what new info to store
        i = sigmoid(np.dot(self.Wi, combined) + self.bi)
        c_candidate = np.tanh(np.dot(self.Wc, combined) + self.bc)

        # Update cell state
        c = f * c_prev + i * c_candidate

        # Output gate: what to output
        o = sigmoid(np.dot(self.Wo, combined) + self.bo)
        h = o * np.tanh(c)

        return h, c
```

Think of the LSTM like a conveyor belt. The cell state is the belt, running along mostly undisturbed. The gates are workers along the belt who can:
- Remove items (forget gate)
- Add new items (input gate)
- Show items to the outside world (output gate)

Because the cell state has a direct path through the entire sequence (with only element-wise operations), gradients can flow much more easily, allowing LSTMs to learn dependencies spanning hundreds of time steps.

```python
# Using LSTM in PyTorch
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True  # Process sequence in both directions
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Concatenate forward and backward final hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.fc(hidden)
        return output
```

## GRU: Gated Recurrent Unit


![Visual representation of deep learning network architecture](https://picsum.photos/seed/recurrent-neural-networks-explained-3/800/450)

The **GRU**, introduced in 2014 by Cho et al., is a simplified version of the LSTM. It merges the cell state and hidden state into a single state and uses only two gates instead of three:

1. **Reset Gate**: Controls how much of the previous state to forget
2. **Update Gate**: Controls the balance between old state and new candidate

```python
# GRU in PyTorch — same interface as LSTM but simpler
gru = nn.GRU(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.2
)
```

GRUs often perform comparably to LSTMs but are faster to train because they have fewer parameters. In practice, the choice between LSTM and GRU often comes down to experimentation.

## Bidirectional RNNs

A standard RNN processes the sequence left to right. But for many tasks, the future context is just as important as the past. A **bidirectional RNN** processes the sequence in both directions and concatenates the hidden states.

For the sentence "The ___ was delicious," you need the word "delicious" (which comes later) to predict the blank (probably "food" or "cake"). A bidirectional model can capture this.

## The Rise and Decline of RNNs

RNNs (particularly LSTMs) dominated natural language processing from roughly 2014 to 2018. They powered:
- Machine translation (Google Translate)
- Speech recognition (Siri, Google Assistant)
- Text generation
- Sentiment analysis
- Named entity recognition

However, RNNs have since been largely supplanted by **Transformers** (the architecture behind BERT and GPT) for most NLP tasks. Transformers process all tokens in parallel rather than sequentially, making them much faster to train and better at capturing long-range dependencies.

That said, RNNs remain relevant for:
- Real-time sequence processing (streaming data)
- Edge devices with limited memory
- Tasks where the sequential nature is a feature, not a bug
- Understanding the fundamentals of sequence modeling

## Key Takeaways

1. RNNs process sequences by maintaining a **hidden state** (memory) that flows from step to step
2. Standard RNNs suffer from the **vanishing gradient problem**, limiting their memory
3. **LSTMs** solve this with a cell state and three gates (forget, input, output)
4. **GRUs** are a simpler alternative to LSTMs with similar performance
5. **Bidirectional** RNNs process sequences in both directions for richer representations
6. While Transformers have largely replaced RNNs in NLP, understanding RNNs is essential for grasping the evolution of deep learning

---

*This concludes the core Deep Learning architecture posts. Next, I will take a step back and tackle the most confusing concepts in AI — the terms and ideas that trip up beginners and even experienced practitioners.*
