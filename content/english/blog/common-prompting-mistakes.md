---
title: "Common Prompting Mistakes to Avoid with ChatGPT and Claude"
date: 2026-10-15T10:00:00+05:30
draft: false
description: "The most common mistakes developers make when prompting LLMs — and how to fix each one for dramatically better results."
tags: ["Prompt Engineering", "ChatGPT", "Claude", "LLM", "Best Practices"]
categories: ["Prompt Engineering"]
image: "https://images.unsplash.com/photo-1517433670267-08bbd4be890f?w=1200&h=630&fit=crop&auto=format"
keywords: ["prompting mistakes", "bad prompts", "ChatGPT mistakes", "prompt engineering errors", "how to prompt better", "LLM prompting tips"]
---

After spending hundreds of hours working with LLMs and reviewing how other developers use them, I have identified patterns that consistently lead to poor results. These mistakes are common, understandable, and — fortunately — easy to fix once you recognize them.

Here are the most impactful prompting mistakes I see, along with specific fixes for each.

### Mistake 1: Being Too Vague

This is the most common mistake by far. Vague prompts produce vague responses.

**The mistake:**
```
Help me with my database query.
```

**Why it fails:** The model does not know what database, what table, what the query should do, or what "help" means. It will guess, and those guesses will be generic.

**The fix:**
```
I have a PostgreSQL database with a `transactions` table:
- id (UUID, primary key)
- user_id (UUID, foreign key to users)
- amount (DECIMAL)
- created_at (TIMESTAMP)
- status (VARCHAR: 'pending', 'completed', 'failed')

Write a query that returns the total completed transaction
amount per user for the last 30 days, ordered by total
amount descending. Include only users with more than $1000
in total transactions.
```

**Rule of thumb:** If someone else could interpret your prompt in more than one way, it is too vague.

### Mistake 2: Asking Multiple Unrelated Questions at Once

**The mistake:**
```
Can you explain how Docker networking works, also write me
a Python script to parse CSV files, and what are the best
practices for React state management?
```

**Why it fails:** The model tries to address everything, giving shallow treatment to each topic. The context gets muddled, and the response for each question is worse than if asked individually.

**The fix:** Ask one question at a time, or clearly separate related questions:

```
I have three separate questions. Please address each one
under its own heading:

1. Docker Networking: [specific question]
2. CSV Parsing: [specific question]
3. React State: [specific question]
```

Even better: use separate prompts for unrelated topics.


![Illustration of effective prompt engineering techniques and patterns](https://picsum.photos/seed/common-prompting-mistakes-1/800/450)

### Mistake 3: Not Providing the Actual Code

**The mistake:**
```
I have a function that sorts an array but it's returning
wrong results. What's the issue?
```

**Why it fails:** The model cannot debug code it cannot see. It will guess at common sorting bugs, and those guesses may be completely irrelevant to your specific issue.

**The fix:**
```
This function should sort an array of objects by their
'date' field (most recent first), but dates from the same
month are appearing in random order:

function sortByDate(items) {
  return items.sort((a, b) => {
    return new Date(a.date) - new Date(b.date);
  });
}

Example input: [{ date: "2024-03-15" }, { date: "2024-03-01" }]
Expected: most recent first
Actual: oldest first
```

**Rule:** Always include the actual code, the expected behavior, and the actual behavior.

### Mistake 4: Accepting the First Response Without Iteration

**The mistake:** Taking whatever the model generates as the final answer and moving on.

**Why it fails:** LLMs are probabilistic. The first response is rarely the optimal one. It might be correct but not optimal, or correct for the common case but missing edge cases.

**The fix:** Treat the first response as a draft and iterate:

```
Round 1: Generate the solution
Round 2: "What edge cases does this miss?"
Round 3: "How would this perform with 1 million records?"
Round 4: "Can you add error handling for network failures?"
Round 5: "Simplify this — remove any unnecessary complexity."
```

Each round improves the output. The final result after 3-5 rounds is dramatically better than the first attempt.

### Mistake 5: Not Specifying the Output Format

**The mistake:**
```
Compare React and Vue for my project.
```

**Why it fails:** You get a wall of text that may or may not cover what you actually need. The model decides the format, and its choice may not match your needs.

**The fix:**
```
Compare React and Vue for a medium-sized e-commerce SPA.

Format your response as a table with these columns:
| Criteria | React | Vue | Winner for My Use Case |

Cover these criteria:
1. Learning curve for a team of 3 backend developers
2. Performance for a product catalog with 10K+ items
3. State management options
4. Mobile app potential (React Native vs. similar)
5. Job market and hiring
6. Ecosystem maturity

After the table, give a 3-sentence recommendation.
```


![Visual showing how structured prompts guide AI reasoning](https://picsum.photos/seed/common-prompting-mistakes-2/800/450)

### Mistake 6: Giving Too Much Irrelevant Context

**The mistake:**
```
[Pastes entire 1500-line file]

There's a bug somewhere in this code. Find it.
```

**Why it fails:** The model must search through irrelevant code, consuming tokens and attention. The signal-to-noise ratio is terrible.

**The fix:** Narrow down the relevant section:

```
The bug is in the checkout flow. Here is the relevant
function and its dependencies:

[50 lines of relevant code]

Symptom: The discount is applied twice when the user
has both a coupon code and a loyalty discount.
```

If you genuinely do not know where the bug is, at least provide the symptom and the file structure so the model knows where to focus.

### Mistake 7: Using LLMs for Tasks They Are Bad At

**The mistake:**
```
What is 847,293 × 391,847?
```

**Why it fails:** LLMs are not calculators. They predict tokens, not compute arithmetic. They frequently get large arithmetic operations wrong.

**Better approach:**
```
Write a Python function that multiplies two large numbers
and returns the result. Then show the result of calling
it with 847293 and 391847.
```

Or simply: use a calculator for arithmetic, a search engine for current events, and an LLM for what it is good at — understanding, generating, and transforming text.

**Tasks LLMs are bad at:**
- Precise arithmetic with large numbers
- Real-time information (stock prices, weather, news)
- Counting characters, words, or tokens accurately
- Generating truly random outputs
- Tasks requiring deterministic, reproducible results

### Mistake 8: Not Telling the Model What NOT to Do

**The mistake:**
```
Explain Kubernetes to me.
```

You get a 2000-word essay when you needed a quick overview. Or you get a beginner explanation when you needed advanced details.

**The fix:**
```
Explain Kubernetes networking to me. I already understand:
- Pods and containers
- Services and deployments
- Basic networking concepts (TCP/IP, DNS)

Do NOT explain these basics. Focus specifically on:
- How pod-to-pod communication works across nodes
- The role of CNI plugins
- How Services route traffic to pods

Keep it under 500 words.
```

Negative constraints are as important as positive ones.


![Conceptual image depicting the interaction between human intent and AI output](https://picsum.photos/seed/common-prompting-mistakes-3/800/450)

### Mistake 9: Not Leveraging Conversation History

**The mistake:** Starting every prompt from scratch, even in the same conversation.

**Why it fails:** You waste tokens re-establishing context, and you miss the opportunity to build on previous responses.

**The fix:** Use conversation history deliberately:

```
Message 1: "Here's my database schema: [schema]"

Message 2: "Write a query to find inactive users"

Message 3: "Now optimize that query with proper indexes"

Message 4: "Add this query to a Prisma migration"
```

Each message builds on the established context. The model remembers the schema, the query, and the optimization from previous messages.

### Mistake 10: Trusting the Output Blindly

This is not a prompting technique per se, but it is the most dangerous mistake.

**The mistake:** Copy-pasting LLM-generated code into production without review.

**Why it is dangerous:** LLMs can generate code that:
- Looks correct but has subtle bugs
- Has security vulnerabilities (SQL injection, XSS)
- Uses deprecated APIs
- Does not handle edge cases
- Works in the common case but fails silently in edge cases

**The fix:** Treat LLM output like code from a junior developer who is very fast but occasionally careless. Always:

1. Read and understand every line
2. Run the tests
3. Check for security issues
4. Verify edge case handling
5. Ensure it follows your project's conventions

### The Meta-Lesson

All of these mistakes share a common root: treating the LLM as a magic oracle rather than as a tool that requires skillful use. An LLM is like a power tool — incredible when used correctly, dangerous when used carelessly, and mediocre when used without skill.

The investment in learning to prompt well pays dividends on every single interaction. Fix these ten mistakes, and you will immediately see a dramatic improvement in the quality of your LLM outputs.
