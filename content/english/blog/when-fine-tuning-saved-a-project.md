---
title: "Case Study: When Fine-Tuning Saved a Project"
date: 2026-11-29T10:00:00+05:30
draft: false
description: "A real-world case study of how fine-tuning transformed a struggling AI project — from the initial failure to the successful deployment."
tags: ["Fine-Tuning", "Case Study", "LLM", "Production AI", "Generative AI"]
categories: ["Fine-Tuning"]
image: "https://picsum.photos/seed/when-fine-tuning-saved-a-project-cover/1200/630"
keywords: ["fine-tuning case study", "LLM production", "AI project", "fine-tuning success", "real world fine-tuning", "LLM deployment"]
---

Theory is valuable, but nothing teaches like a real project with real stakes. In this post, I will walk through a case study of a project where fine-tuning was the difference between failure and success — covering the initial approach that failed, the pivot to fine-tuning, and the lessons learned along the way.

### The Project: Automated Ticket Classification

A mid-size SaaS company was drowning in support tickets. Their support team received 2,000-3,000 tickets per day, and manually routing each ticket to the right team (billing, technical, account management, feature requests, bugs) was consuming 3 full-time agents doing nothing but triage.

The goal: build an AI system that automatically classifies incoming tickets and routes them to the correct team with at least 90% accuracy.

### Attempt 1: Prompt Engineering (Weeks 1-2)

The natural first approach. We set up a GPT-4o API call with a detailed system prompt:

```python
system_prompt = """You are a support ticket classifier for CloudBase,
a cloud hosting platform. Classify each ticket into exactly one category:

- BILLING: Payment issues, invoices, subscription changes, pricing questions
- TECHNICAL: Server errors, deployment issues, configuration problems, API bugs
- ACCOUNT: Login issues, team management, permissions, account settings
- FEATURE_REQUEST: Suggestions for new features or improvements
- BUG_REPORT: Reports of software defects or unexpected behavior
- SECURITY: Security concerns, vulnerability reports, suspicious activity

Respond with ONLY the category name. Nothing else."""
```

**Results:**
- Accuracy: 78%
- The model struggled with ambiguous tickets that could belong to multiple categories
- It confused TECHNICAL and BUG_REPORT frequently (both involve things not working)
- Tickets written in informal language or with typos were misclassified more often
- Cost: approximately $180/day at 2,500 tickets

78% accuracy sounds reasonable until you realize that means 550 tickets per day were misrouted. That is worse than the manual process for those tickets, because now they have to be identified as misrouted AND re-routed.

### Attempt 2: Few-Shot Prompting (Weeks 3-4)

![Accuracy improvement chart across prompt RAG and fine-tuning attempts](https://picsum.photos/seed/when-fine-tuning-saved-a-project-1/800/450)


We added 10 carefully selected examples to the prompt, covering edge cases and ambiguous tickets:

```python
examples = """
Ticket: "I was charged $49.99 but I'm on the free plan"
Category: BILLING

Ticket: "My app keeps crashing with a 500 error after deploying"
Category: TECHNICAL

Ticket: "The dashboard shows error 500 when I click on analytics"
Category: BUG_REPORT

Ticket: "Can't log in, says my password is wrong but I just changed it"
Category: ACCOUNT

Ticket: "It would be great if you could add dark mode to the dashboard"
Category: FEATURE_REQUEST

...
"""
```

**Results:**
- Accuracy: 84%
- Improvement on clear-cut cases
- Still struggling with edge cases
- Prompt was now 2,000+ tokens, increasing costs
- Cost: approximately $280/day (longer prompts)

84% was better but still below the 90% threshold. And the increased prompt length raised costs significantly.

### Attempt 3: RAG with Historical Tickets (Weeks 5-6)

We indexed 50,000 historical tickets with their correct classifications in a vector database. For each new ticket, we retrieved the 5 most similar historical tickets and their categories as context.

```python
def classify_with_rag(new_ticket):
    similar_tickets = vector_db.search(new_ticket, top_k=5)

    context = "\n".join([
        f"Similar ticket: \"{t['text']}\" → {t['category']}"
        for t in similar_tickets
    ])

    response = llm.generate(
        f"Based on these similar tickets:\n{context}\n\n"
        f"Classify this ticket: \"{new_ticket}\""
    )
    return response
```

**Results:**
- Accuracy: 87%
- Good improvement, especially for common ticket types
- But retrieval sometimes found the wrong similar tickets
- Edge cases and novel tickets were still misclassified
- Latency increased (vector search + LLM call)
- Cost: approximately $350/day (embeddings + longer prompts)

87% was close but still not meeting the 90% target. And the system was getting more complex and expensive with each iteration.

### The Pivot: Fine-Tuning (Weeks 7-9)

![Error analysis showing confusion between ticket categories](https://picsum.photos/seed/when-fine-tuning-saved-a-project-2/800/450)


After three approaches fell short, we decided to fine-tune.

#### Data Preparation

We had a goldmine: 50,000 historically classified tickets from the past two years, manually categorized by experienced support agents.

```python
# Data preparation pipeline
def prepare_training_data(historical_tickets):
    training_examples = []

    for ticket in historical_tickets:
        example = {
            "messages": [
                {
                    "role": "system",
                    "content": "Classify the support ticket into one of: "
                               "BILLING, TECHNICAL, ACCOUNT, FEATURE_REQUEST, "
                               "BUG_REPORT, SECURITY"
                },
                {
                    "role": "user",
                    "content": ticket["text"]
                },
                {
                    "role": "assistant",
                    "content": ticket["category"]
                }
            ]
        }
        training_examples.append(example)

    return training_examples
```

**Data quality steps:**

1. **Removed duplicates**: 50,000 reduced to 42,000 unique tickets
2. **Verified labels**: Sampled 500 tickets and had senior agents verify. Found 8% mislabeled. Used this to estimate noise level.
3. **Balanced categories**: Some categories were underrepresented. We upsampled SECURITY (rare) and downsampled TECHNICAL (very common).
4. **Split data**: 80% training (33,600), 10% validation (4,200), 10% test (4,200)

#### Model Selection and Training

We chose GPT-4o-mini for fine-tuning — good balance of quality and cost.

```python
# Training configuration
job = client.fine_tuning.jobs.create(
    model="gpt-4o-mini-2024-07-18",
    training_file=training_file_id,
    validation_file=validation_file_id,
    hyperparameters={
        "n_epochs": 3,
        "learning_rate_multiplier": 1.8,
        "batch_size": 8,
    }
)
```

Training took about 4 hours and cost approximately $95 in compute.

#### Results: Iteration 1

- **Accuracy on test set: 91.3%**
- **BILLING accuracy: 95%** (up from 82%)
- **TECHNICAL vs. BUG_REPORT: 88%** (still the hardest distinction)
- **SECURITY: 89%** (impressive given low training volume)

We exceeded the 90% threshold on the first fine-tuning attempt.

#### Iteration 2: Addressing Weak Spots

We analyzed the misclassifications:

```python
# Error analysis
confusion_pairs = Counter()
for prediction, actual in misclassifications:
    confusion_pairs[(actual, prediction)] += 1

# Top confusion pairs:
# (BUG_REPORT, TECHNICAL): 43 cases
# (TECHNICAL, BUG_REPORT): 38 cases
# (ACCOUNT, BILLING): 22 cases
```

The TECHNICAL vs. BUG_REPORT confusion made sense — both involve things not working. We added 200 carefully curated examples specifically for this distinction:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "The deployment is failing with error: cannot connect to database"}, {"role": "assistant", "content": "TECHNICAL"}]}
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "The deployment page shows a blank screen instead of the deployment log"}, {"role": "assistant", "content": "BUG_REPORT"}]}
```

The distinction: TECHNICAL means "something in the customer's setup is not working." BUG_REPORT means "something in our product is broken."

**Results after iteration 2: 93.7% accuracy**

### Production Deployment

The fine-tuned model was deployed with several safeguards:

```python
class TicketClassifier:
    def __init__(self):
        self.model_id = "ft:gpt-4o-mini:company::abc123"
        self.confidence_threshold = 0.85

    def classify(self, ticket_text):
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": "Classify the support ticket..."},
                {"role": "user", "content": ticket_text}
            ],
            temperature=0.0,
            logprobs=True,
            top_logprobs=3,
        )

        category = response.choices[0].message.content
        confidence = self._calculate_confidence(response)

        if confidence < self.confidence_threshold:
            return {
                "category": category,
                "confidence": confidence,
                "action": "HUMAN_REVIEW",
                "reason": "Low confidence classification"
            }

        return {
            "category": category,
            "confidence": confidence,
            "action": "AUTO_ROUTE"
        }
```

**Key design decisions:**

1. **Confidence threshold**: Tickets below 85% confidence are flagged for human review rather than auto-routed. This catches the edge cases the model is unsure about.

2. **Monitoring dashboard**: Track accuracy weekly by comparing auto-classifications against human corrections.

3. **Feedback loop**: When agents reclassify a ticket, that becomes training data for the next fine-tuning iteration.

4. **Fallback**: If the API is down, tickets go to a general queue for manual triage.

### The Business Impact

![Production deployment architecture with confidence thresholds](https://picsum.photos/seed/when-fine-tuning-saved-a-project-3/800/450)


After 3 months in production:

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Auto-classification accuracy | N/A | 93.7% | N/A |
| Tickets requiring human triage | 2,500/day | 400/day | -84% |
| Average routing time | 45 min | 2 seconds | -99.9% |
| Triage agents needed | 3 FTE | 0.5 FTE | -83% |
| Monthly triage cost | ~$15,000 | ~$2,500 + $300 API | -81% |
| Customer time to first response | 4.2 hours | 1.1 hours | -74% |

The fine-tuned model cost approximately $300/month in API costs (much less than the prompt engineering approach because the prompts are shorter — no few-shot examples or RAG context needed).

### Lessons Learned

**1. Fine-tuning is not the first step — it is the last step.**
We tried prompt engineering, few-shot, and RAG first. Each attempt taught us what the model struggled with, which informed our fine-tuning data preparation. If we had jumped straight to fine-tuning, we would not have known which examples to focus on.

**2. Data quality is the bottleneck.**
The difference between 91% and 94% accuracy came from 200 carefully curated examples, not from thousands of additional generic ones.

**3. Confidence thresholds are essential.**
Automatic classification with a confidence threshold and human fallback is far more practical than trying to achieve 100% accuracy. The 15% of tickets that go to human review would have gone to human review anyway.

**4. Build a feedback loop from day one.**
Every human correction is a training example for the next iteration. The model gets better over time automatically.

**5. Start with the smallest viable model.**
GPT-4o-mini was sufficient for classification. We did not need GPT-4o. Smaller models are faster, cheaper, and often just as accurate for focused tasks.

Fine-tuning is not magic — it is engineering. Like any engineering project, success comes from understanding the problem, iterating on the solution, and measuring results rigorously. But when applied to the right problem with the right data, it can transform a struggling AI project into a production success.
