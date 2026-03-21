---
title: "My Best Prompt Hacks for Getting Better AI Outputs"
date: 2026-10-18T10:00:00+05:30
draft: false
description: "Battle-tested prompt hacks and advanced techniques I use daily to get consistently excellent outputs from ChatGPT, Claude, and other LLMs."
tags: ["Prompt Engineering", "ChatGPT", "Claude", "LLM", "Productivity"]
categories: ["Prompt Engineering"]
image: "https://images.unsplash.com/photo-1593376853899-fbb47a057fa0?w=1200&h=630&fit=crop&auto=format"
keywords: ["prompt hacks", "best prompts", "ChatGPT tips", "Claude tips", "advanced prompting", "AI productivity", "prompt tricks"]
---

Over the past year, I have refined a set of prompting techniques that consistently produce better outputs across ChatGPT, Claude, Gemini, and open-source models. These are not theoretical — they are battle-tested hacks I use daily for real development and writing work.

Here are my best prompt hacks, ordered roughly by impact.

### Hack 1: The "Before You Start" Preface

One of the most powerful techniques is asking the model to plan before it acts.

```
Before writing any code, first:
1. List the key requirements you identify from my description
2. Outline your approach in 3-5 bullet points
3. Identify potential edge cases
4. Only then write the implementation

Task: Build a rate limiter middleware for Express...
```

This prevents the model from rushing into code that misses requirements. The planning phase forces it to think through the problem systematically.

### Hack 2: The "Assume I Know" Shortcut

Skip the basics to get directly to the meat:

```
I am an experienced Python developer. Assume I know
standard library, common patterns, and basic CS concepts.
Skip all introductory explanations and go directly to
the advanced implementation details.

How do I implement a custom async context manager for
database connection pooling?
```

This one hack can cut response length by 50% while increasing the density of useful information.

### Hack 3: The Constraint Sandwich

Put your most important constraints both at the beginning and end of your prompt:

```
IMPORTANT: All code must be TypeScript with strict mode.

[Your detailed task description here, which might be
several paragraphs long...]

REMINDER: TypeScript strict mode only. No 'any' types.
```

LLMs pay more attention to the beginning and end of prompts. Placing critical constraints in both positions reduces the chance they are overlooked.

### Hack 4: The "Grade Yourself" Follow-up

After the model generates a response, ask it to evaluate its own work:

```
Now review what you just wrote. Rate it on a scale of
1-10 for:
- Correctness
- Completeness
- Code quality
- Edge case coverage

For any rating below 8, explain what's missing and
provide an improved version.
```

Models are often surprisingly good at identifying their own shortcomings when explicitly asked. This is a free quality check.

![Advanced prompting techniques for getting better AI outputs](https://picsum.photos/seed/best-prompt-hacks-1/800/450)

### Hack 5: The Negative Example

Show the model what you do NOT want:

```
Write a product description for a wireless mouse.

Do NOT write like this:
"Introducing the revolutionary XYZ Mouse! This
game-changing device will transform your workflow
with its amazing features!"

DO write like this:
"The XYZ Mouse connects via Bluetooth 5.0, has a
4000 DPI sensor, and lasts 6 months on a single AA
battery. It weighs 85g and supports three device
pairing."

Now write the description for: [product details]
```

Negative examples are incredibly effective at steering the model away from patterns you dislike.

### Hack 6: The Structured Output Hack

When you need structured data, provide a template with placeholders:

```
Fill in this template for each of the top 5 React state
management libraries:

### [Library Name]
- **Best for:** [1 sentence]
- **Bundle size:** [approximate size]
- **Learning curve:** [Easy/Medium/Hard]
- **Boilerplate:** [Low/Medium/High]
- **Key feature:** [1 sentence]
- **When to avoid:** [1 sentence]
```

The model will follow the template exactly, producing perfectly formatted, consistent output.

### Hack 7: The "Explain Then Do" Pattern

For complex tasks, ask the model to explain its approach first:

```
I need to implement a WebSocket reconnection strategy with
exponential backoff, jitter, and a maximum retry limit.

First, explain the algorithm you would use and why.
Then implement it in TypeScript.
Then write tests for the key behaviors.
```

The explanation phase activates reasoning that produces better code. It also lets you catch misunderstandings before the model writes hundreds of lines of wrong code.

### Hack 8: The Scope Limiter

Explicitly define what is in scope and what is not:

```
IN SCOPE:
- The API endpoint implementation
- Input validation with Zod
- Error response formatting

OUT OF SCOPE (do not implement):
- Authentication middleware (already exists)
- Database connection setup (already exists)
- Deployment configuration

Assume these imports are available:
import { db } from '../db'
import { requireAuth } from '../middleware/auth'
```

This prevents the model from generating boilerplate for things that already exist, saving tokens and reducing noise.

### Hack 9: The Incremental Refinement Loop

Instead of trying to get the perfect output in one shot, build incrementally:

```
Step 1: "Write a basic version of [thing] that handles
the happy path only."

Step 2: "Now add error handling for these cases: [list]"

Step 3: "Add input validation."

Step 4: "Optimize the performance for the case where
[specific scenario]."

Step 5: "Add JSDoc comments and TypeScript types."
```

Each step is focused and manageable. The final result is typically better than asking for everything at once, because the model can focus its attention on one concern at a time.

![Structured prompts with scope limiters and incremental refinement](https://picsum.photos/seed/best-prompt-hacks-2/800/450)

### Hack 10: The "Write Tests First" Approach

Flip the usual order and ask for tests first:

```
Before writing the implementation, write the test cases
for a function called `parseUserAgent` that:
- Extracts browser name and version from a UA string
- Returns an object { browser, version, os, device }
- Handles Chrome, Firefox, Safari, and Edge
- Returns "Unknown" for unrecognized browsers

Write at least 10 test cases covering normal cases,
edge cases, and error cases.

Then write the implementation that passes all tests.
```

Test-first prompting produces more robust code because the model thinks about edge cases before writing the implementation.

### Hack 11: The XML Tag Hack

For complex prompts, use XML-like tags to clearly delimit sections:

```
<task>
Write a migration to add soft-delete to the users table
</task>

<context>
Database: PostgreSQL 15
ORM: Prisma
Current schema:
model User {
  id    String @id @default(cuid())
  email String @unique
  name  String
}
</context>

<requirements>
- Add deleted_at column (nullable timestamp)
- Add index on deleted_at
- Update all existing queries to exclude soft-deleted users
- Provide a migration file and updated Prisma schema
</requirements>

<constraints>
- Must be backwards compatible
- Must not break existing API responses
- Migration must be reversible
</constraints>
```

This is particularly effective with Claude, which was specifically trained to work well with XML-structured prompts.

### Hack 12: The Comparison Table Hack

When you need to make a decision, ask for a comparison:

```
I need to choose between Redis and Memcached for
session storage in my Node.js app.

Create a comparison table with these criteria:
1. Performance (ops/second for reads and writes)
2. Data persistence options
3. Data structure support
4. Memory efficiency
5. Clustering support
6. Node.js client library maturity

Then give your recommendation for my use case:
[describe use case]
```

Tables force the model to be systematic and cover all criteria evenly. They are also much easier to read than paragraphs of prose.

![XML-tagged prompts and comparison table techniques for complex AI tasks](https://picsum.photos/seed/best-prompt-hacks-3/800/450)

### Hack 13: The "As If I Were" Format

Combine persona and format in one instruction:

```
Explain this concept as if you were:
1. Writing a README for a library
2. Giving a 5-minute tech talk
3. Answering a Stack Overflow question
4. Writing production code comments

The concept: Circuit breaker pattern in microservices
```

Each format produces a different angle on the same concept. This is great for learning or for generating content in multiple formats.

### Hack 14: The "Teach Me to Fish" Hack

Instead of asking for a solution, ask to learn the pattern:

```
Don't just write the regex for me. Instead:
1. Explain the pattern I need to match
2. Build the regex step by step, explaining each part
3. Show me how to test and debug it
4. Give me similar exercises to practice

I need to match email addresses in a text body.
```

This produces responses that teach you the underlying skill, making you less dependent on the model over time.

### Building Your Toolkit

These hacks are not meant to be used all at once. They are tools in a toolbox. Over time, you will develop intuition for which ones apply to which situations:

- **Quick questions**: Hack 2 (Assume I Know) + Hack 5 (Negative Examples)
- **Code generation**: Hack 1 (Before You Start) + Hack 8 (Scope Limiter) + Hack 10 (Tests First)
- **Complex analysis**: Hack 4 (Grade Yourself) + Hack 7 (Explain Then Do) + Hack 12 (Comparison Table)
- **Learning**: Hack 14 (Teach Me to Fish) + Hack 13 (As If I Were)

The best prompt engineers do not memorize techniques — they internalize principles and apply them fluidly. Practice these hacks, adapt them to your workflow, and you will notice a significant improvement in the quality and consistency of your AI interactions.
