---
title: "How Are LLMs Trained? Pre-training vs. Fine-tuning Explained"
date: 2026-09-18T10:00:00+05:30
draft: false
description: "A deep dive into the two-phase training process of Large Language Models — pre-training on massive datasets and fine-tuning for specific tasks."
tags: ["LLM", "Training", "Pre-training", "Fine-tuning", "Deep Learning", "Generative AI"]
categories: ["Generative AI"]
image: "https://images.unsplash.com/photo-1542435503-956c469947f6?w=1200&h=630&fit=crop&auto=format"
keywords: ["LLM training", "pre-training", "fine-tuning", "how GPT is trained", "RLHF", "instruction tuning", "model training pipeline"]
---

When someone tells you that GPT-4 "knows" how to write code, explain quantum physics, and draft legal contracts, the natural question is: how did it learn all of that? Nobody sat down and programmed these capabilities one by one. The answer lies in a two-phase training process that is both elegant and computationally brutal.

In this post, we will walk through the complete training pipeline of a modern LLM — from raw internet text to the polished, helpful assistant you interact with through ChatGPT or Claude.

### The Two-Phase Training Pipeline

Training an LLM is not a single step. It happens in distinct phases, each serving a different purpose:

1. **Phase 1: Pre-training** — Learn language, facts, and reasoning from massive text data
2. **Phase 2: Alignment** — Learn to be helpful, honest, and safe through fine-tuning and RLHF

Think of it like education. Pre-training is like going through school and university — absorbing a vast amount of general knowledge. Alignment is like professional training — learning how to apply that knowledge helpfully and responsibly.


![Creative AI generating novel content from learned patterns](https://picsum.photos/seed/how-llms-are-trained-1/800/450)

### Phase 1: Pre-training

Pre-training is where the model learns "everything." This phase consumes the vast majority of the compute budget — often 99% or more.

**The Data:**

Modern LLMs are pre-trained on datasets that are almost incomprehensibly large:

- **Common Crawl**: Petabytes of web pages scraped from the internet
- **Books**: Digitized books covering every genre and topic
- **Wikipedia**: The entire encyclopedia in multiple languages
- **Code repositories**: GitHub, Stack Overflow, and other code sources
- **Academic papers**: ArXiv, PubMed, and other research databases
- **Social media and forums**: Reddit, discussion boards, Q&A sites

This data goes through extensive preprocessing:

```python
# Simplified data pipeline
raw_data = crawl_internet()                    # Petabytes of raw text
filtered_data = remove_duplicates(raw_data)     # Deduplication
clean_data = remove_toxic_content(filtered_data) # Content filtering
quality_data = filter_by_quality(clean_data)     # Quality scoring
tokenized_data = tokenize(quality_data)          # Convert to tokens
```

**The Objective: Next-Token Prediction**

The pre-training objective is simple: given a sequence of tokens, predict the next one. This is called **causal language modeling**.

The model sees a passage like: "The Eiffel Tower is located in ___" and must predict "Paris." It does this trillions of times across the entire dataset.

```python
# Conceptual pre-training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch[:, :-1]   # All tokens except last
        labels = batch[:, 1:]       # All tokens except first (shifted)

        logits = model(input_ids)   # Forward pass
        loss = cross_entropy(logits, labels)

        loss.backward()             # Compute gradients
        optimizer.step()            # Update weights
        optimizer.zero_grad()       # Reset gradients
```

**The Compute:**

Pre-training GPT-3 required approximately:
- 355 GPU-years of compute on V100 GPUs
- Several months of wall-clock time on thousands of GPUs
- An estimated $4.6 million in compute costs (at 2020 prices)

GPT-4 is estimated to have cost over $100 million in compute alone.

**What the Model Learns:**

Through next-token prediction on internet-scale data, the model implicitly learns:
- Grammar and syntax of multiple languages
- Factual knowledge (history, science, geography)
- Reasoning patterns (if A then B, because X therefore Y)
- Code syntax and programming patterns
- Common sense knowledge
- Mathematical relationships

None of these are explicitly taught. They emerge from the statistical patterns in the training data.

### The Gap: Pre-trained Models Are Not Helpful

Here is the important insight: a pre-trained model is powerful but not useful as an assistant. If you prompt a raw pre-trained model with "What is the capital of France?", it might respond with:

> "What is the capital of Germany? What is the capital of Italy? What is the capital of Spain?"

Why? Because in the training data, questions are often followed by more questions (think: quiz sheets, FAQ pages). The model has learned to predict what text comes next, but it has not learned to answer questions helpfully.

This is the gap that Phase 2 fills.


![Illustration of the generative process from noise to coherent output](https://picsum.photos/seed/how-llms-are-trained-2/800/450)

### Phase 2: Alignment (Fine-tuning + RLHF)

Alignment is the process of taking a pre-trained model and teaching it to be a helpful, honest, and harmless assistant. This happens in two sub-phases:

#### Step 2a: Supervised Fine-Tuning (SFT)

In this step, humans write thousands of example conversations demonstrating ideal assistant behavior:

```
Human: What is the capital of France?
Assistant: The capital of France is Paris. It is the largest city
in France and serves as the country's political, economic, and
cultural center.
```

The model is then fine-tuned on these demonstrations. This teaches it the format and style of helpful responses.

The SFT dataset is much smaller than the pre-training data — typically tens of thousands to hundreds of thousands of examples rather than trillions of tokens. But its impact is dramatic.

#### Step 2b: Reinforcement Learning from Human Feedback (RLHF)

SFT gets the model to respond in the right format, but it does not teach it to choose the best response. RLHF addresses this.

The process works as follows:

1. **Generate multiple responses**: For a given prompt, the model generates several candidate responses
2. **Human ranking**: Human evaluators rank the responses from best to worst
3. **Train a reward model**: A separate neural network learns to predict which responses humans prefer
4. **Optimize with RL**: The LLM is fine-tuned using reinforcement learning (PPO algorithm) to maximize the reward model's score

```
Prompt: "Explain photosynthesis simply"

Response A: "Photosynthesis is the process by which plants
convert sunlight into food using water and CO2..."
→ Human rank: 1 (Best)

Response B: "The biochemical process of photosynthesis involves
the reduction of CO2 through a series of electron..."
→ Human rank: 2 (Too technical)

Response C: "Plants eat sunlight lol"
→ Human rank: 3 (Too casual, inaccurate)
```

The reward model learns from thousands of these comparisons. Then the LLM is optimized to produce responses that the reward model would rate highly.

### The Complete Pipeline

Putting it all together, the complete training pipeline looks like this:

```
Raw Internet Data (petabytes)
    ↓ Cleaning, filtering, deduplication
Curated Pre-training Dataset (trillions of tokens)
    ↓ Next-token prediction (months, millions of $)
Pre-trained Base Model
    ↓ Supervised Fine-Tuning (thousands of demonstrations)
SFT Model (responds in assistant format)
    ↓ RLHF (human preference rankings)
Aligned Model (helpful, honest, harmless)
    ↓ Safety fine-tuning, red-teaming
Production Model (GPT-4, Claude, etc.)
```

Each step builds on the previous one. You cannot skip Phase 1 — the model needs the broad knowledge from pre-training. And you cannot skip Phase 2 — without alignment, the model is a powerful but uncontrolled text generator.


![Visual representation of text-to-image generation pipeline](https://picsum.photos/seed/how-llms-are-trained-3/800/450)

### Instruction Tuning: A Key Technique

A particularly important fine-tuning technique is **instruction tuning**. Rather than fine-tuning on conversation-style data alone, the model is trained on a diverse set of instructions and their correct executions:

- "Translate this English text to French: ..."
- "Write a Python function that..."
- "Summarize this article in three bullet points: ..."
- "What are the pros and cons of..."

This teaches the model to follow arbitrary instructions, making it a general-purpose assistant rather than a narrow-task model.

### Why This Matters

Understanding the training pipeline has practical implications:

1. **Knowledge cutoff**: Pre-training data has a date range. The model does not know about events after its training data cutoff.
2. **Hallucination**: The model predicts statistically likely text, not verified facts. This is why it can generate plausible-sounding but incorrect information.
3. **Fine-tuning potential**: If the pre-trained model has the knowledge but is not formatted correctly for your use case, fine-tuning can unlock it.
4. **Prompt engineering**: Understanding that the model is fundamentally a next-token predictor helps you write better prompts — frame things in ways the model has seen during training.

The training pipeline is the foundation of every LLM application. Whether you are using these models through an API or fine-tuning your own, understanding how they were trained will make you a more effective practitioner.
