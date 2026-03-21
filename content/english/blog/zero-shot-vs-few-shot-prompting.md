---
title: "Prompting Techniques: Zero-Shot vs. Few-Shot Learning"
date: 2026-10-03T10:00:00+05:30
draft: false
description: "Master zero-shot and few-shot prompting techniques to dramatically improve LLM outputs — with practical examples for developers."
tags: ["Prompt Engineering", "Zero-Shot", "Few-Shot", "LLM", "AI Techniques"]
categories: ["Prompt Engineering"]
image: "https://picsum.photos/seed/zero-shot-vs-few-shot-prompting-cover/1200/630"
keywords: ["zero-shot prompting", "few-shot prompting", "few-shot learning", "LLM prompting", "in-context learning", "prompt examples"]
---

One of the most remarkable abilities of Large Language Models is **in-context learning** — the ability to learn new tasks from just a few examples provided in the prompt, without any weight updates or fine-tuning. This capability is the foundation of two critical prompting techniques: **zero-shot** and **few-shot** prompting.

Understanding when to use each — and how to design effective examples — is one of the highest-leverage skills in prompt engineering.

### Zero-Shot Prompting: No Examples Needed

In zero-shot prompting, you give the model a task description without any examples. You rely entirely on the model's pre-trained knowledge to understand what you want.

**Example:**
```
Classify the following customer review as positive, negative,
or neutral:

"The product arrived on time but the packaging was damaged.
The item itself works fine though."

Classification:
```

The model has never seen this specific review or this specific classification task. But because it has been trained on vast amounts of text that includes sentiment analysis, reviews, and classification tasks, it can infer what you want and produce the correct answer: **neutral** (or arguably positive).

**When zero-shot works well:**
- The task is common and well-represented in training data
- The instructions are clear and unambiguous
- The expected output format is obvious
- The model is large and capable (GPT-4, Claude 3.5, etc.)

**When zero-shot fails:**
- The task is unusual or domain-specific
- The expected output format is non-obvious
- The task requires following a specific pattern that is not intuitive
- You need consistent formatting across multiple runs

### Few-Shot Prompting: Learning from Examples

Few-shot prompting includes one or more examples of the task before asking the model to perform it. These examples demonstrate the input-output pattern you expect.

**Example (3-shot):**
```
Extract the product name and price from the following
descriptions:

Description: "Get the new AirPods Pro for just $249.99"
Output: {"product": "AirPods Pro", "price": "$249.99"}

Description: "Samsung Galaxy S24 Ultra now available at $1,199"
Output: {"product": "Samsung Galaxy S24 Ultra", "price": "$1,199"}

Description: "Grab the Sony WH-1000XM5 headphones for $348"
Output: {"product": "Sony WH-1000XM5", "price": "$348"}

Description: "The new MacBook Air M3 starts at $1,099"
Output:
```

The model will produce: `{"product": "MacBook Air M3", "price": "$1,099"}` — matching the exact format of the examples.

**Why few-shot works:**
The examples serve multiple purposes:
1. **Format specification**: They show exactly what the output should look like
2. **Task clarification**: They resolve ambiguity about what the task actually is
3. **Edge case handling**: They demonstrate how to handle tricky inputs
4. **Consistency**: They anchor the model to a specific pattern

### How Many Shots? The Practical Guide

![Zero-shot versus few-shot prompting comparison with examples](https://picsum.photos/seed/zero-shot-vs-few-shot-prompting-1/800/450)


Research and experience suggest the following guidelines:

**1-shot (one example):**
- Sufficient for simple formatting tasks
- Good when the task is intuitive and you just need to show the output format
- Risk: the model might overfit to your single example

**3-shot (three examples):**
- The sweet spot for most tasks
- Enough to establish a clear pattern without consuming too many tokens
- Shows enough variation to prevent overfitting

**5-shot (five examples):**
- Good for complex or ambiguous tasks
- Helps when the task has multiple edge cases
- Use when 3-shot results are inconsistent

**10+ shots:**
- Rarely needed with modern models
- Can actually decrease performance (the model gets "lost" in too many examples)
- Consider fine-tuning instead if you need this many examples

### Designing Effective Examples

Not all examples are created equal. Here are principles for designing examples that work:

**1. Cover the range of inputs:**
```
# Bad: All examples are similar
Classify: "I love this!" → Positive
Classify: "This is great!" → Positive
Classify: "Amazing product!" → Positive

# Good: Examples cover different categories
Classify: "I love this!" → Positive
Classify: "Terrible experience" → Negative
Classify: "It works as expected" → Neutral
```

**2. Include edge cases:**
```
# Include tricky examples that clarify boundaries
Classify: "Not bad, but not great either" → Neutral
Classify: "I wouldn't buy it again, but it served its purpose" → Negative
```

**3. Keep examples realistic:**
Use examples that resemble your actual data, not contrived or overly simple cases.

**4. Order matters:**
Place the most representative examples first and the most challenging ones last. The model pays more attention to examples closer to the actual query.

### Zero-Shot vs. Few-Shot: A Comparison

![Designing effective few-shot examples for consistent output](https://picsum.photos/seed/zero-shot-vs-few-shot-prompting-2/800/450)


Here is a practical comparison using a real task — extracting structured data from unstructured text:

**Task:** Extract meeting details from natural language.

**Zero-shot attempt:**
```
Extract the date, time, and attendees from this message:

"Let's meet with Sarah and Tom next Tuesday at 3pm to
discuss the Q4 budget."
```

Output might be inconsistent — sometimes JSON, sometimes plain text, sometimes missing fields.

**Few-shot attempt:**
```
Extract meeting details from the following messages:

Message: "Coffee with Jake on Friday at 10am"
Result: {"date": "Friday", "time": "10:00 AM",
         "attendees": ["Jake"], "topic": "Coffee"}

Message: "Team standup Monday 9:30, all hands on deck"
Result: {"date": "Monday", "time": "9:30 AM",
         "attendees": ["Full team"], "topic": "Standup"}

Message: "Let's meet with Sarah and Tom next Tuesday at
3pm to discuss the Q4 budget."
Result:
```

The few-shot version will reliably produce the exact JSON format with all fields, because the pattern is clearly established.

### Advanced Few-Shot Techniques

**Negative examples (showing what NOT to do):**
```
Good response: "The function has O(n) time complexity
because it iterates through the list once."

Bad response: "It's O(n)."

Good response: "The bug is on line 15 where the loop
counter starts at 1 instead of 0, causing an off-by-one
error that skips the first element."

Bad response: "There's an off-by-one error."

Now analyze this code: [CODE]
```

**Chain-of-thought in examples:**
```
Question: "If a train travels 120km in 2 hours, and then
150km in 3 hours, what is the average speed?"

Thinking: Total distance = 120 + 150 = 270km
Total time = 2 + 3 = 5 hours
Average speed = 270 / 5 = 54 km/h

Answer: 54 km/h

Question: "A factory produces 450 units in 5 days working
8 hours per day. What is the production rate per hour?"
```

### Practical Code Example: Building a Few-Shot Classifier

![Decision guide for choosing between prompting and fine-tuning](https://picsum.photos/seed/zero-shot-vs-few-shot-prompting-3/800/450)


Here is how to implement few-shot classification in Python:

```python
import openai

def classify_support_ticket(ticket_text: str) -> str:
    """Classify a support ticket using few-shot prompting."""

    few_shot_examples = """
Ticket: "I can't log into my account, it says invalid password"
Category: Authentication

Ticket: "My payment was charged twice for order #12345"
Category: Billing

Ticket: "The app crashes when I try to upload a photo"
Category: Bug Report

Ticket: "How do I change my email notification settings?"
Category: Feature Question

Ticket: "I want to cancel my subscription and get a refund"
Category: Cancellation
"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a support ticket classifier. "
                           "Classify tickets into exactly one category."
            },
            {
                "role": "user",
                "content": f"{few_shot_examples}\n"
                           f"Ticket: \"{ticket_text}\"\n"
                           f"Category:"
            }
        ],
        temperature=0.0,
        max_tokens=20
    )

    return response.choices[0].message.content.strip()
```

### When to Use Which

**Use zero-shot when:**
- The task is straightforward and well-defined
- You want to minimize token usage (cost savings)
- You are using a highly capable model (GPT-4, Claude 3.5)
- The output format does not need to be precise

**Use few-shot when:**
- You need consistent output formatting
- The task is domain-specific or unusual
- Zero-shot results are inconsistent
- You are using a less capable model
- Precision matters (data extraction, classification)

**Consider fine-tuning when:**
- You need 10+ examples to get good results
- You are running the same task thousands of times
- The few-shot examples consume too many tokens
- You need the highest possible accuracy

### The Key Takeaway

Zero-shot and few-shot prompting are your first line of defense against poor LLM outputs. Before reaching for complex solutions like fine-tuning or RAG, try adding a few well-chosen examples to your prompt. In my experience, this simple technique solves 80% of output quality issues.

Start with zero-shot. If the results are inconsistent, add three examples. If you need more, you are probably looking at a fine-tuning problem. This simple heuristic will serve you well across virtually every LLM task you encounter.
