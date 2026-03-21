---
title: "The Art of Context: How to Give LLMs Better Information"
date: 2026-10-12T10:00:00+05:30
draft: false
description: "Learn how providing the right context transforms LLM outputs — covering context design, information hierarchy, and practical patterns for developers."
tags: ["Prompt Engineering", "LLM", "Context Window", "AI Techniques"]
categories: ["Prompt Engineering"]
image: "https://picsum.photos/seed/providing-good-context-to-llms-cover/1200/630"
keywords: ["LLM context", "providing context to AI", "prompt context", "context window", "AI context design", "better AI responses"]
---

The single most common reason for poor LLM output is not a model limitation — it is insufficient context. The model does not know about your codebase, your business rules, your team conventions, or your specific requirements unless you tell it. And how you provide that context matters enormously.

Context is the difference between a generic response and a genuinely useful one. In this post, I will share the principles and patterns I use to give LLMs the information they need to produce excellent results.

### The Context Problem

When you ask an LLM a question, it only has access to:

1. Its pre-training knowledge (general, potentially outdated)
2. Whatever you include in the current prompt
3. The conversation history (if any)

It does not have access to:
- Your codebase
- Your project's documentation
- Your team's coding conventions
- Your specific business requirements
- What you have already tried
- Why you are asking this question

Every piece of relevant information you omit is a gap that the model will fill with generic assumptions. The more gaps, the more generic (and less useful) the response.

### The Context Hierarchy

Not all context is equally important. I think about context in layers:

**Layer 1: Essential Context (Always Include)**
- What specific task you want the model to do
- The programming language, framework, and version
- Any constraints or requirements
- The expected output format

**Layer 2: Important Context (Include When Relevant)**
- Relevant code snippets (the actual code, not just descriptions)
- Error messages (complete, not truncated)
- What you have already tried
- The broader system architecture
- Business rules that affect the solution

**Layer 3: Background Context (Include for Complex Tasks)**
- Why you are doing this (the business context)
- Team conventions and coding standards
- Performance requirements
- Deployment environment details
- Related parts of the codebase

### Practical Patterns for Context Design

#### Pattern 1: The Context Block

Structure your context explicitly:

```
## Context
I am building a REST API for an e-commerce platform using
Node.js (v20), Express, and PostgreSQL. The API serves a
React frontend. We use TypeScript with strict mode. Our
team follows REST conventions with JSON:API format.

## Current Code
[relevant code here]

## Problem
The /orders endpoint returns a 500 error when the customer
has no shipping address on file.

## Error
TypeError: Cannot read properties of null (reading 'street')
at formatAddress (/src/utils/address.ts:15)

## What I've Tried
- Added a null check in the controller (didn't help because
  the error is in the utility function)
- Checked the database — the customer record exists but
  the shipping_address field is NULL

## What I Need
Fix the formatAddress utility to handle null addresses
gracefully, returning a default "No address on file" response.
```

This structured approach eliminates ambiguity and gives the model everything it needs in one shot.


![Visual guide to effective LLM interaction techniques](https://picsum.photos/seed/providing-good-context-to-llms-1/800/450)

#### Pattern 2: The Progressive Context

For complex problems, provide context incrementally:

```
Message 1: "I'm working on a Node.js API with this
architecture: [high-level description]"

Message 2: "Here's the specific module I need help with:
[code]"

Message 3: "The issue I'm seeing is: [specific problem]"

Message 4: "Can you suggest a fix that doesn't break
the existing API contract?"
```

This works well in conversational interfaces where the model can ask clarifying questions.

#### Pattern 3: The Reference Context

Include reference material for domain-specific tasks:

```
Here is our API style guide:
- All endpoints return JSON:API format
- Error responses include error codes from our enum
- Pagination uses cursor-based pagination
- All timestamps are ISO 8601 in UTC

Here is an example of a correctly implemented endpoint:
[EXAMPLE CODE]

Now implement a new endpoint for /products that follows
the same patterns.
```

### What to Include vs. What to Omit

**Always include:**
- The actual code (not paraphrased descriptions)
- Complete error messages and stack traces
- Specific versions of languages and frameworks
- Constraints that affect the solution (performance, compatibility)

**Include when relevant:**
- Database schemas for data-related tasks
- API contracts for integration tasks
- Test cases that should pass
- Security requirements

**Omit:**
- Irrelevant code (other modules that are not related)
- Personal opinions about what you think the problem is (let the model analyze independently)
- Information that would fill the context window without adding value


![Diagram showing prompt design patterns and strategies](https://picsum.photos/seed/providing-good-context-to-llms-2/800/450)

### The "Minimal Reproducible Context" Principle

Borrowed from the concept of a minimal reproducible example in bug reports, this principle states: **provide the minimum context needed to reproduce the problem or understand the task, but no less.**

Too little context leads to generic responses. Too much context can confuse the model or waste tokens. The sweet spot is providing exactly what is needed.

```
# Too little context:
"My function doesn't work. Fix it."

# Too much context:
[Entire 2000-line file with only 10 relevant lines]

# Just right:
"This function should return the sum of even numbers in
a list, but it returns the wrong result for negative numbers:

def sum_evens(numbers):
    return sum(n for n in numbers if n % 2 == 0)

Input: [-4, -3, -2, 1, 2, 4]
Expected: 0 (because -4 + -2 + 2 + 4 = 0)
Actual: 0

Wait — actually the output is correct. Let me re-check
my test... [The bug was in my test, not the function]"
```

That last example illustrates an important point: writing out the context carefully sometimes helps you solve the problem yourself. This is the "rubber duck debugging" effect.

### Context for Different Task Types

**For code generation:**
```
Language: TypeScript (5.3+, strict mode)
Framework: Next.js 14 (App Router)
Database: PostgreSQL via Prisma ORM
Auth: NextAuth.js v5

Generate a server action for updating a user's profile
that:
- Validates input with Zod
- Checks the user is authenticated
- Updates only the provided fields
- Returns the updated profile
- Handles errors gracefully

Prisma schema for User:
model User {
  id        String   @id @default(cuid())
  name      String?
  email     String   @unique
  bio       String?
  avatarUrl String?
  updatedAt DateTime @updatedAt
}
```

**For debugging:**
```
Environment: Node.js 20, running in Docker on AWS ECS
Problem: Memory usage grows linearly over 24 hours until
the container is OOM-killed

Metrics:
- Starts at 256MB RSS
- Grows ~10MB per hour
- OOM-killed at 512MB limit after ~24 hours

Relevant code (the suspected leak is in the cache module):
[CODE]

Heap snapshot analysis shows growing Map entries in the
ResponseCache class.
```

**For architecture review:**
```
System context:
- 50K DAU, expected to grow to 200K in 6 months
- Real-time features (chat, notifications)
- Currently monolith, considering microservices
- Team: 5 backend, 3 frontend, 1 DevOps
- Budget: limited (startup)
- Current stack: Django, PostgreSQL, Redis, AWS

Proposed change: Extract the notification system into a
separate service using Go and Kafka.

Question: Is this the right move given our constraints?
What are the risks?
```

### Managing Context Window Limits


![Illustration of prompt engineering workflow and optimization](https://picsum.photos/seed/providing-good-context-to-llms-3/800/450)

Even with models supporting 200K+ tokens, context management matters:

**Prioritize recent and relevant information:**
```python
def build_context(query, codebase, max_tokens=50000):
    context_parts = []

    # 1. Always include the direct query context
    relevant_files = find_relevant_files(query, codebase)
    context_parts.extend(relevant_files[:5])

    # 2. Include dependencies of relevant files
    for f in relevant_files[:3]:
        deps = get_direct_dependencies(f)
        context_parts.extend(deps[:3])

    # 3. Include relevant tests
    tests = find_related_tests(relevant_files)
    context_parts.extend(tests[:3])

    # 4. Truncate if needed
    return truncate_to_tokens(context_parts, max_tokens)
```

**Summarize when you cannot include everything:**
```
I cannot include our full 50-file codebase, but here is
a summary of the architecture:

- /api: Express routes (REST, JSON responses)
- /services: Business logic layer
- /models: Sequelize ORM models
- /middleware: Auth, validation, error handling
- /utils: Shared helpers

The specific files relevant to this task are:
[INCLUDE FULL CODE OF 3-4 RELEVANT FILES]
```

### The Key Insight

Great prompt engineering is largely about great context engineering. The model is incredibly capable — but only when it has the right information. Invest time in structuring your context, and the quality of your LLM interactions will improve dramatically.

Think of it this way: if a brilliant consultant walked into your office knowing nothing about your project, you would not just say "fix the bug." You would spend time bringing them up to speed — the architecture, the problem, what you have tried, and what success looks like. Give your LLM the same courtesy.
