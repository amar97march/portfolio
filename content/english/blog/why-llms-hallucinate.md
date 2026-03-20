---
title: "Why LLMs Hallucinate and How to Fix It"
date: 2026-10-21T10:00:00+05:30
draft: false
description: "A technical deep dive into why Large Language Models generate false information, the types of hallucination, and practical strategies to detect and prevent them."
tags: ["LLM", "Hallucination", "RAG", "AI Safety", "Generative AI"]
categories: ["RAG & Retrieval"]
image: "/images/blogs/pool-rag/1.jpg"
keywords: ["LLM hallucination", "AI hallucination", "why AI makes things up", "fixing AI hallucinations", "grounding LLMs", "factual accuracy AI"]
---

You ask an LLM to list the publications of a researcher, and it confidently invents papers that do not exist. You request a legal citation, and it generates a case number for a case that was never filed. You ask for a code snippet using a library, and it writes code calling functions that the library does not have.

This is **hallucination** — one of the most significant challenges with Large Language Models today. Understanding why it happens is the first step to building systems that mitigate it.

### What Is Hallucination?

Hallucination is when an LLM generates text that is fluent, confident, and plausible-sounding, but factually incorrect or entirely fabricated. The model does not "know" it is wrong — it has no concept of truth. It generates the most statistically likely continuation of the text, and sometimes the most likely continuation is false.

### Types of Hallucination

Not all hallucinations are the same. Researchers have identified several categories:

**1. Factual Hallucination**
The model states something that is factually wrong:
```
Prompt: "When was the Eiffel Tower built?"
Hallucinated response: "The Eiffel Tower was built in 1905."
(Correct: construction completed in 1889)
```

**2. Fabrication**
The model invents entities, events, or references that do not exist:
```
Prompt: "Cite a study on the effects of social media on teens"
Hallucinated: "Smith et al. (2021), 'Digital Minds: A
Longitudinal Study', Journal of Social Psychology, 45(3)."
(This paper, author, and journal issue do not exist)
```

**3. Instruction Hallucination**
The model fails to follow instructions while appearing to comply:
```
Prompt: "List exactly 5 items"
Response: Lists 7 items (or includes made-up items to reach 5)
```

**4. Context Hallucination**
The model contradicts information provided in its own context:
```
Context: "The company was founded in 2019"
Later in response: "Since its founding in 2015, the company..."
```

### Why Hallucination Happens: The Technical Roots

![Root causes of LLM hallucination from training to inference](/images/blogs/pool-rag/3.jpg)


Understanding the root causes helps us design mitigations.

#### Cause 1: The Training Objective

LLMs are trained to predict the next token, not to be truthful. The loss function is cross-entropy on next-token prediction — there is no term for factual accuracy.

When the model sees "The capital of Atlantis is," it generates the most likely completion based on its training data. If enough fictional texts mention a capital of Atlantis, the model will confidently state one — even though Atlantis is mythological.

#### Cause 2: Statistical Smoothing

The model has learned statistical patterns, not a database of facts. When it encounters a query about something it has limited training data for, it "fills in the gaps" by interpolating from similar patterns.

If the model has seen many biographical entries in the format "X was born in Y and graduated from Z," it can generate plausible-sounding biographies for people it knows little about — inventing birthplaces and universities based on statistical patterns.

#### Cause 3: The Pressure to Be Helpful

RLHF training teaches models to be helpful and provide answers. This creates a bias toward generating responses rather than saying "I don't know." The model has learned that users prefer substantive answers, so it generates them even when it should express uncertainty.

#### Cause 4: Knowledge Compression

A 70B parameter model cannot faithfully store all the facts from trillions of tokens of training data. Information is stored as distributed patterns across weights, not as retrievable records. This lossy compression means that fine-grained factual details — dates, names, numbers — are particularly prone to errors.

#### Cause 5: Lack of Source Grounding

The model generates text without referencing specific sources. It does not think "according to source X, the answer is Y." It generates tokens based on aggregate patterns, which means it cannot distinguish between reliable and unreliable information in its training data.

### Detecting Hallucination

Before we can fix hallucination, we need to detect it.

**Confidence-based detection:**
```python
def detect_low_confidence(response, model, prompt):
    """Flag responses where the model shows low confidence."""
    # Generate multiple responses
    responses = [model.generate(prompt, temperature=0.7)
                 for _ in range(5)]

    # Check consistency
    if responses are highly varied:
        return "HIGH HALLUCINATION RISK: inconsistent responses"

    return "Lower risk: consistent across samples"
```

**Self-verification:**
```
Prompt: "You just stated [CLAIM]. Are you confident this
is accurate? Rate your confidence from 1-10 and explain
any uncertainty."
```

**Cross-reference checking:**
```python
def verify_claims(response):
    """Extract claims and verify against known sources."""
    claims = extract_factual_claims(response)
    for claim in claims:
        verification = search_reliable_sources(claim)
        if not verification.confirmed:
            flag_as_potential_hallucination(claim)
```

### Strategies to Reduce Hallucination

![Hallucination detection strategies and verification pipeline](/images/blogs/pool-rag/4.jpg)


#### Strategy 1: Retrieval-Augmented Generation (RAG)

The most effective strategy is to ground the model's responses in retrieved documents. Instead of relying on parametric knowledge (what the model "remembers" from training), you provide the relevant information in the prompt.

```
Based ONLY on the following document, answer the question.
If the answer is not in the document, say "The document
does not contain this information."

Document: [retrieved content]

Question: [user question]
```

We will cover RAG in depth in the next post.

#### Strategy 2: Temperature Control

Lower temperature reduces randomness, making the model more likely to generate common, well-established completions rather than creative interpolations.

```python
# For factual tasks, use low temperature
response = model.generate(
    prompt,
    temperature=0.1  # More deterministic
)

# For creative tasks, higher temperature is fine
response = model.generate(
    prompt,
    temperature=0.8  # More diverse
)
```

#### Strategy 3: Explicit Uncertainty Instructions

Tell the model it is okay to not know:

```
Answer the following question. If you are not certain
about the answer, say "I'm not sure about this" and
explain what you are uncertain about. Do not guess or
make up information. It is better to be honest about
uncertainty than to provide false information.
```

#### Strategy 4: Structured Outputs with Sources

Force the model to cite its reasoning:

```
For each claim you make, indicate your confidence level:
[VERIFIED] - You are highly confident this is accurate
[LIKELY] - This is probably correct but you're not 100% sure
[UNCERTAIN] - You are guessing and this should be verified

Example:
[VERIFIED] Python was created by Guido van Rossum.
[LIKELY] The first release was in February 1991.
[UNCERTAIN] The name was inspired by Monty Python.
```

#### Strategy 5: Constrained Generation

Limit what the model can generate:

```python
# Instead of free-form generation, use structured output
response = model.generate(
    prompt,
    response_format={
        "type": "json_schema",
        "schema": {
            "answer": "string",
            "confidence": "number (0-1)",
            "sources": "array of strings",
            "caveats": "array of strings"
        }
    }
)
```

#### Strategy 6: Chain of Verification

Ask the model to generate a response, then verify its own claims:

```
Step 1: Answer the question
Step 2: List every factual claim in your answer
Step 3: For each claim, assess whether you are confident
        it is accurate or whether it might be hallucinated
Step 4: Revise your answer, removing or flagging any
        claims you are not confident about
```

### The Hallucination Spectrum

It is important to understand that hallucination risk varies by task:

**Low risk:**
- Code generation (can be verified by running it)
- Formatting and restructuring (no factual claims)
- Creative writing (no "ground truth" to violate)
- Translation (can be cross-checked)

**Medium risk:**
- Summarizing provided documents (might add information not in the source)
- Explaining well-known concepts (mostly accurate, occasional errors)
- Code explanations (usually correct, sometimes misattributes behavior)

**High risk:**
- Citing specific facts, dates, or numbers
- Referencing specific people, papers, or events
- Legal, medical, or financial advice
- Any task where accuracy is critical and errors are costly

### Building Hallucination-Aware Systems

![Risk spectrum of hallucination across different task types](/images/blogs/pool-rag/5.jpg)


For production applications, design your system to account for hallucination:

```python
class HallucinationAwareSystem:
    def generate_response(self, query):
        # 1. Retrieve relevant documents
        docs = self.retriever.search(query)

        # 2. Generate response grounded in documents
        response = self.llm.generate(
            query=query,
            context=docs,
            instructions="Only use information from the "
                         "provided documents"
        )

        # 3. Verify response against sources
        verification = self.verify_against_sources(response, docs)

        # 4. Flag or filter unverified claims
        if verification.has_unverified_claims:
            response = self.add_uncertainty_markers(response)

        return response
```

### The Honest Truth

Hallucination is not a bug that will be "fixed" in the next model release. It is a fundamental property of how LLMs work. Models that predict the most likely text will sometimes generate plausible text that is not true.

The solution is not to wait for perfect models — it is to build systems that account for imperfection. Use RAG to ground responses in real data. Use verification to catch errors. Use structured outputs to force transparency. And always, always remind your users that LLM outputs should be verified for critical decisions.

In the next post, we will dive deep into **RAG (Retrieval-Augmented Generation)** — the most powerful technique we have for reducing hallucination and building knowledge-grounded AI applications.
