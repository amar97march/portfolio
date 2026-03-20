---
title: "Chain of Thought Prompting: Getting AI to Show Its Work"
date: 2026-10-06T10:00:00+05:30
draft: false
description: "Learn how Chain of Thought prompting dramatically improves LLM reasoning on complex tasks — with practical examples and implementation patterns."
tags: ["Prompt Engineering", "Chain of Thought", "LLM", "Reasoning", "AI Techniques"]
categories: ["Prompt Engineering"]
image: "/images/blogs/pool-prompt/1.jpg"
keywords: ["chain of thought prompting", "CoT prompting", "LLM reasoning", "step by step AI", "reasoning prompts", "think step by step"]
---

There is a simple phrase that can dramatically improve an LLM's performance on complex tasks: **"Think step by step."** This is the essence of Chain of Thought (CoT) prompting — a technique that unlocks reasoning capabilities that standard prompting leaves on the table.

Chain of Thought prompting was introduced in a 2022 paper by Google researchers, and it fundamentally changed how we interact with LLMs on tasks that require multi-step reasoning.

### The Problem: LLMs Rush to Answers

By default, LLMs generate responses token by token, and they tend to jump straight to an answer. For simple factual questions, this works fine. But for problems that require reasoning — math, logic, planning, debugging — rushing to an answer leads to errors.

**Example without CoT:**

```
Prompt: "A store sells apples for $2 each. If I buy 3 apples
and pay with a $20 bill, and I also have a 15% off coupon
for the total, how much change do I get?"

Response: "$14.90"  ← Wrong
```

The model jumped to an answer without working through the steps. It might have applied the discount incorrectly or made an arithmetic error, and we have no way to see where it went wrong.

**Example with CoT:**

```
Prompt: "A store sells apples for $2 each. If I buy 3 apples
and pay with a $20 bill, and I also have a 15% off coupon
for the total, how much change do I get?

Please think through this step by step."

Response:
"Step 1: Calculate the cost of 3 apples
3 × $2 = $6.00

Step 2: Apply the 15% discount
Discount = $6.00 × 0.15 = $0.90
Discounted price = $6.00 - $0.90 = $5.10

Step 3: Calculate change from $20
Change = $20.00 - $5.10 = $14.90"
```

In this case, the answer is the same, but now we can verify each step. For more complex problems, CoT consistently produces more accurate results.

### Why Chain of Thought Works

The reason CoT improves performance is rooted in how LLMs generate text. An LLM predicts one token at a time. When it generates intermediate reasoning steps, those steps become part of the context for generating the next token. This effectively gives the model a "scratchpad" to work through the problem.

Without CoT, the model must compress all of its reasoning into the immediate jump to an answer — a lossy process for complex problems.

Think of it this way: if I asked you to multiply 47 by 83 in your head, you might struggle. But give you a piece of paper to write down intermediate steps, and it becomes straightforward. CoT gives the model that piece of paper.


![Illustration of effective prompt engineering techniques and patterns](/images/blogs/pool-prompt/3.jpg)

### Types of Chain of Thought Prompting

#### 1. Zero-Shot CoT

The simplest form. Just append a reasoning instruction to your prompt:

```
[Your question]

Let's think step by step.
```

Or variations like:
- "Think through this carefully before answering."
- "Break this down into steps."
- "Show your reasoning."
- "Work through this problem methodically."

This is surprisingly effective. The original paper showed that just adding "Let's think step by step" improved accuracy on math word problems from 17.7% to 78.7% with the same model.

#### 2. Few-Shot CoT

Provide examples that include reasoning chains:

```
Question: "Roger has 5 tennis balls. He buys 2 more cans
of tennis balls. Each can has 3 balls. How many tennis
balls does he have now?"

Reasoning:
- Roger starts with 5 balls
- He buys 2 cans × 3 balls per can = 6 balls
- Total = 5 + 6 = 11 balls

Answer: 11

Question: "The cafeteria had 23 apples. They used 20 for
lunch and bought 6 more. How many do they have?"

Reasoning:
```

The model learns the pattern of showing work from the examples and applies it to the new question.

#### 3. Self-Consistency with CoT

Generate multiple reasoning chains for the same problem and take the majority answer:

```python
def solve_with_self_consistency(question, n_samples=5):
    answers = []
    for _ in range(n_samples):
        response = generate_with_cot(
            question,
            temperature=0.7  # Allow variation
        )
        answer = extract_final_answer(response)
        answers.append(answer)

    # Return the most common answer
    return most_common(answers)
```

If 4 out of 5 reasoning chains arrive at the same answer, you can be more confident it is correct. This technique catches errors in individual reasoning chains.

### Practical Applications for Developers

CoT is not just for math problems. Here is how I use it in real development work:

#### Debugging with CoT

```
Here is a Python function that is supposed to find the
longest palindromic substring, but it returns incorrect
results for some inputs:

[CODE]

Please analyze this step by step:
1. Trace through the algorithm logic
2. Identify the specific condition or loop where the bug occurs
3. Explain why the bug causes incorrect results
4. Provide the corrected code
```

#### Architecture Decisions with CoT

```
I need to choose between a monolithic and microservices
architecture for a new e-commerce platform. The team is
5 developers, we need to launch in 3 months, and we expect
to scale to 10,000 daily active users in the first year.

Please reason through this decision:
1. What are the requirements and constraints?
2. What are the trade-offs of each approach given our team size?
3. What are the risks of each approach given our timeline?
4. What would you recommend and why?
```

#### Code Review with CoT

```
Review this database query for potential issues:

[SQL QUERY]

Analyze step by step:
1. What does this query do?
2. What is the execution plan likely to look like?
3. Are there any performance concerns?
4. Are there any correctness issues?
5. What indexes would improve performance?
```


![Visual showing how structured prompts guide AI reasoning](/images/blogs/pool-prompt/4.jpg)

### Tree of Thought: CoT Evolved

**Tree of Thought (ToT)** extends CoT by exploring multiple reasoning paths simultaneously, not just one. At each step, the model generates several possible next steps, evaluates them, and pursues the most promising ones.

```
Problem: [COMPLEX PROBLEM]

Step 1 - Consider multiple approaches:
Approach A: [reasoning...]
Approach B: [reasoning...]
Approach C: [reasoning...]

Evaluation: Approach B seems most promising because...

Step 2 - Continue with Approach B:
Option B1: [reasoning...]
Option B2: [reasoning...]

Evaluation: Option B1 is more robust because...

Continue until solution is reached.
```

This is particularly useful for planning and design tasks where there are multiple valid paths.

### Implementation Pattern: Structured CoT

Here is a reusable pattern for implementing CoT in your applications:

```python
def solve_with_cot(problem: str, model: str = "gpt-4o") -> dict:
    """Solve a problem using structured Chain of Thought."""

    system_prompt = """You are a precise problem solver.
    For every problem:
    1. Restate the problem in your own words
    2. Identify the key information and constraints
    3. Break the solution into clear steps
    4. Execute each step, showing your work
    5. Verify your answer
    6. State your final answer clearly

    Format your response as:
    UNDERSTANDING: [your restatement]
    KEY INFO: [bullet points]
    STEPS: [numbered steps with work shown]
    VERIFICATION: [check your answer]
    FINAL ANSWER: [concise answer]"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem}
        ],
        temperature=0.1
    )

    return parse_cot_response(response.choices[0].message.content)
```


![Conceptual image depicting the interaction between human intent and AI output](/images/blogs/pool-prompt/5.jpg)

### When CoT Helps (and When It Doesn't)

**CoT significantly helps with:**
- Mathematical word problems
- Multi-step logical reasoning
- Code debugging and analysis
- Architecture and design decisions
- Planning and scheduling tasks
- Any task where intermediate steps matter

**CoT does NOT help with:**
- Simple factual lookups ("What is the capital of France?")
- Tasks that are already at the model's capability limit
- Creative tasks where you want spontaneous output
- Simple classification tasks

**CoT can actually hurt when:**
- The task is simple — the model overthinks and introduces errors
- Token budget is extremely limited
- Speed is critical and you cannot afford the extra tokens

### The "Let Me Think" Principle

Chain of Thought is fundamentally about giving the model space to think. Modern models like Claude and GPT-4 have internalized this to some degree — they naturally produce more detailed reasoning for complex queries. But explicitly requesting step-by-step thinking still produces measurably better results.

The lesson extends beyond AI. Whether you are debugging code, designing a system, or making a business decision, writing out your reasoning step by step leads to better outcomes. CoT is not just a prompting technique — it is good thinking practice.

In the next post, we will explore another powerful technique: **persona prompting** — and why telling an LLM to "act as a senior developer" actually works at a technical level.
