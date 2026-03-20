---
title: "Before Transformers: Why RNNs and LSTMs Were Limited"
date: 2026-08-25T10:00:00+05:30
draft: false
description: "Before Transformers dominated AI, RNNs and LSTMs were the standard for sequence modeling. Learn how they work, why they were groundbreaking, and the fundamental limitations that led to their replacement."
tags: ["Deep Learning", "RNN", "LSTM", "NLP", "Neural Networks"]
categories: ["Deep Learning"]
image: "/images/blogs/pool-dl/1.jpg"
keywords: ["RNN", "LSTM", "recurrent neural network", "vanishing gradient", "sequence modeling", "GRU", "limitations of RNNs"]
---

To fully appreciate why Transformers were revolutionary, you need to understand what came before them. For nearly a decade, **Recurrent Neural Networks (RNNs)** and their variants — **LSTMs** and **GRUs** — were the dominant architecture for any task involving sequences: language modeling, translation, speech recognition, and time-series prediction.

They were clever, effective, and fundamentally limited. Understanding their limitations reveals exactly why the Transformer was such a breakthrough.

---

### Part 1: The Core Idea of RNNs

A standard feedforward neural network processes each input independently. But language is sequential — the meaning of a word depends on what came before it.

An RNN solves this by maintaining a **hidden state** that acts as memory. At each time step, the network takes the current input and the previous hidden state, combines them, and produces a new hidden state and an output.

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

Where:
- $h_t$ is the hidden state at time $t$
- $x_t$ is the input at time $t$
- $W_h$ and $W_x$ are learned weight matrices
- $b$ is a bias term

The hidden state $h_t$ is a compressed representation of everything the network has seen so far. It is updated at every step, incorporating new information while (theoretically) retaining old information.

```python
import numpy as np

def simple_rnn_step(x, h_prev, W_h, W_x, b):
    """One step of a simple RNN."""
    h = np.tanh(W_h @ h_prev + W_x @ x + b)
    return h

# Example: processing a sequence of 5 words (each 10-dimensional)
hidden_size = 20
input_size = 10
seq_length = 5

W_h = np.random.randn(hidden_size, hidden_size) * 0.1
W_x = np.random.randn(hidden_size, input_size) * 0.1
b = np.zeros(hidden_size)

# Initialize hidden state
h = np.zeros(hidden_size)

# Process sequence one step at a time
for t in range(seq_length):
    x = np.random.randn(input_size)  # Input at time t
    h = simple_rnn_step(x, h, W_h, W_x, b)
    print(f"Step {t}: hidden state norm = {np.linalg.norm(h):.4f}")
```

---

### Part 2: The Vanishing Gradient Problem

Here is the fundamental issue. During backpropagation through time (BPTT), gradients must flow backward through every time step. At each step, the gradient is multiplied by the weight matrix $W_h$.

If the largest eigenvalue of $W_h$ is less than 1, the gradients **shrink exponentially** — they vanish. After 20-30 steps, the gradient is essentially zero, and the network cannot learn long-range dependencies.

If the largest eigenvalue is greater than 1, the gradients **grow exponentially** — they explode. This causes training instability.

```python
# Demonstrating gradient vanishing
W = np.random.randn(3, 3) * 0.5  # Small weights

gradient = np.eye(3)
for step in range(50):
    gradient = W @ gradient
    if step % 10 == 0:
        print(f"Step {step}: gradient norm = {np.linalg.norm(gradient):.2e}")
```

Run this code and you will see the gradient norm drop from ~1 to effectively zero within 30-50 steps. This means an RNN cannot learn relationships between words that are more than 20-30 positions apart.

In practice, this meant RNNs struggled with sentences like:
"The author, who grew up in a small village near the coast and later moved to the capital where he studied literature for several years, eventually **published** his first **novel**."

The relationship between "author" and "published/novel" spans too many words for a vanilla RNN to capture.

---

![RNN sequential processing versus Transformer parallel processing](/images/blogs/pool-dl/6.jpg)


### Part 3: LSTMs — The Gating Solution

**Long Short-Term Memory (LSTM)** networks, introduced by Hochreiter and Schmidhuber in 1997, addressed the vanishing gradient problem with a clever mechanism: **gates**.

An LSTM maintains two state vectors:
- **Hidden state** ($h_t$): The short-term memory, like a vanilla RNN.
- **Cell state** ($c_t$): The long-term memory — a highway that information can flow along with minimal modification.

Three gates control the flow of information:

#### Forget Gate
Decides what to discard from the cell state.
$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

#### Input Gate
Decides what new information to add to the cell state.
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$

#### Output Gate
Decides what to output from the cell state.
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

The cell state update:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

The hidden state output:
$$h_t = o_t \odot \tanh(c_t)$$

The key innovation is the **cell state pathway** ($c_t$). Because the cell state is updated through addition (not multiplication by a weight matrix), gradients can flow along it without vanishing. The forget gate can learn to keep the cell state intact (gate value close to 1), allowing information to persist over long distances.

---

### Part 4: GRUs — A Simpler Alternative

**Gated Recurrent Units (GRUs)**, introduced by Cho et al. in 2014, simplify the LSTM by combining the forget and input gates into a single **update gate** and merging the cell state and hidden state:

$$z_t = \sigma(W_z [h_{t-1}, x_t])$$
$$r_t = \sigma(W_r [h_{t-1}, x_t])$$
$$\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

GRUs have fewer parameters than LSTMs and are often faster to train, with comparable performance on many tasks.

---

![LSTM gate mechanism controlling information flow](/images/blogs/pool-dl/7.jpg)


### Part 5: Sequence-to-Sequence Models

For tasks like translation, where the input and output sequences have different lengths, the **encoder-decoder** architecture was used:

1. **Encoder:** An LSTM reads the input sequence and compresses it into a single fixed-length vector (the final hidden state).
2. **Decoder:** Another LSTM generates the output sequence, one token at a time, conditioned on this vector.

The problem: compressing an entire sentence into a single vector creates a **bottleneck**. For long sentences, critical information is inevitably lost. This is what motivated the introduction of attention as an add-on to RNNs (Bahdanau et al., 2014).

---

### Part 6: The Five Fundamental Limitations

Despite gating mechanisms and attention add-ons, RNNs and LSTMs had five fundamental limitations that the Transformer overcame:

#### 1. Sequential Processing — No Parallelization
RNNs must process tokens one by one. On a GPU with 10,000 CUDA cores, this means only a tiny fraction of the hardware is utilized at each step. Training on long sequences is inherently slow.

**Transformer solution:** Self-attention processes all positions simultaneously. All the GPU cores are fully utilized.

#### 2. Long-Range Dependencies Still Suffer
Even LSTMs struggle with very long sequences (hundreds or thousands of tokens). While they are better than vanilla RNNs, the information still degrades over very long distances.

**Transformer solution:** Self-attention connects every position to every other position directly, regardless of distance. Word 1 and word 1,000 are equally accessible.

#### 3. Fixed-Length Bottleneck
In encoder-decoder RNNs, the entire input is compressed into a single vector. This creates a severe information bottleneck.

**Transformer solution:** The encoder produces a representation for *every* input position. The decoder attends to all of them.

#### 4. Difficulty with Bidirectional Context
A standard RNN reads left-to-right. To capture right-to-left context, you need a separate backward RNN (bidirectional LSTM). This doubles the model size and still processes each direction independently.

**Transformer solution:** Self-attention naturally captures bidirectional context. Every word attends to every other word — both left and right — in a single pass.

#### 5. Gradient Path Length
In an RNN, the gradient must flow through $T$ time steps to connect position 1 to position $T$. Even with LSTM gates, this is a long path.

**Transformer solution:** In self-attention, every position is directly connected to every other position. The maximum path length is 1, making gradient flow efficient.

---

![Vanishing gradient problem across long sequences](/images/blogs/pool-dl/8.jpg)


### Part 7: The Legacy of RNNs

Despite their limitations, RNNs and LSTMs were enormously important:

- They proved that neural networks could process sequential data effectively.
- They enabled the first high-quality machine translation systems.
- They introduced key concepts (gating, encoder-decoder, attention) that the Transformer built upon.
- They inspired the architectural innovations that led to modern AI.

LSTMs also remain useful in specific scenarios:
- **Real-time streaming:** When you need to process data one element at a time as it arrives.
- **Very low compute budgets:** LSTMs can be smaller and cheaper than Transformers.
- **Simple sequence tasks:** For short sequences and simple patterns, LSTMs can be more parameter-efficient.

---

### Part 8: The Transition

The transition from RNNs to Transformers was not instant. Key milestones:

- **2014:** Attention introduced as an RNN add-on (Bahdanau).
- **2015:** Attention becomes standard in sequence-to-sequence models.
- **2017:** "Attention Is All You Need" removes recurrence entirely.
- **2018:** BERT and GPT demonstrate the power of pure Transformer models.
- **2019-present:** RNNs are largely replaced by Transformers for most tasks.

---

### Final Thoughts

RNNs and LSTMs were the workhorses of sequence modeling for a decade. Their limitations — sequential processing, vanishing gradients, and fixed-length bottlenecks — were well-known and drove the research that ultimately produced the Transformer.

Understanding these limitations is not just historical knowledge. It explains *why* the Transformer was designed the way it was, and why it was such a dramatic improvement.

In the next post, we will dissect the Transformer's internal structure: **Encoder vs. Decoder: The Two Halves of a Transformer**.
