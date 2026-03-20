---
title: "Algorithmic Fairness: What Does 'Fair' Even Mean in Machine Learning?"
meta_title: ""
description: "A deep dive into the competing mathematical definitions of fairness in machine learning, why they often conflict with each other, and practical approaches for building fairer AI systems."
date: 2028-03-28
image: "/images/blogs/algorithmic-fairness/cover.jpg"
categories: ["AI Ethics"]
author: "Amar Singh"
tags: ["fairness", "bias", "ethics", "responsible-ai"]
draft: false
---

Ask ten people what "fairness" means and you will get ten different answers. Ask ten machine learning researchers and you will get at least twenty -- because most of them can articulate multiple formal definitions that are all reasonable but mutually incompatible. This is the central paradox of algorithmic fairness: fairness is universally valued but impossible to define in a way that satisfies everyone, and in many cases, satisfying one definition of fairness mathematically guarantees violating another.

This is not an academic puzzle. Algorithmic decision-making systems now influence who gets a loan, who gets hired, who receives medical treatment, who gets released on bail, and who sees which ads online. The fairness properties of these systems have concrete consequences for millions of people. Understanding what fairness means in the context of machine learning -- and why it is so difficult to achieve -- is essential for anyone building, deploying, or affected by AI systems.

## Why Fairness in ML Is Hard

Before diving into formal definitions, it helps to understand why algorithmic fairness is fundamentally difficult.

### The Problem of Measurement

Fairness requires comparison, and comparison requires measurement. But what, exactly, should we measure? Consider a hiring algorithm. We might ask whether the algorithm treats candidates of different races equally. But what does "equally" mean? That it hires the same proportion from each group? That it applies the same criteria to each group? That it achieves the same accuracy in predicting job performance for each group? Each of these is a legitimate interpretation of fairness, and they lead to different assessments.

### The Problem of Historical Bias

Machine learning systems learn from historical data, and historical data reflects historical decisions -- decisions made by humans with biases, operating within systems shaped by discrimination. A hiring algorithm trained on a company's historical hiring decisions will learn the patterns embedded in those decisions, including any biases. A credit scoring model trained on historical lending data will learn the patterns of a lending industry that has historically discriminated against certain groups.

This creates a fundamental tension: an ML system that accurately models historical patterns will also reproduce historical biases. Deviating from historical patterns to achieve fairer outcomes means sacrificing some degree of predictive accuracy as measured against that biased historical record. This accuracy-fairness trade-off is at the heart of many debates in algorithmic fairness.

### The Problem of Proxy Variables

Even when protected attributes like race or gender are excluded from model inputs, other variables that are correlated with those attributes can serve as proxies. A model that uses zip code as a feature effectively uses a proxy for race in many contexts, since residential segregation means that zip codes are strongly correlated with racial demographics. Similarly, educational history can proxy for socioeconomic status, which in turn correlates with race.

The pervasiveness of proxy variables means that achieving fairness requires more than simply removing protected attributes from the feature set. It requires understanding the relationships between features and protected attributes and actively designing the model to prevent proxy-based discrimination.

![Complexities of measuring fairness across different demographic groups](/images/blogs/pool-ethics/3.jpg)

## Formal Definitions of Fairness

Researchers have proposed numerous formal mathematical definitions of fairness. Each captures a different intuition about what fair treatment means. The most important definitions can be grouped into three broad families: group fairness, individual fairness, and causal fairness.

### Group Fairness Definitions

Group fairness definitions require that some statistical property of the algorithm's outputs be equal (or approximately equal) across defined groups, typically groups defined by a protected attribute like race or gender.

#### Demographic Parity (Statistical Parity)

Demographic parity requires that the algorithm's positive outcome rate be equal across groups. In a hiring context, this means that the algorithm should select the same proportion of candidates from each racial group. In a lending context, it means approving loans at the same rate for each group.

Formally, demographic parity requires that the prediction Y-hat be statistically independent of the protected attribute A. That is, P(Y-hat = 1 | A = 0) = P(Y-hat = 1 | A = 1).

**Strengths**: Demographic parity directly addresses disparate impact -- the legal standard in many jurisdictions that examines whether a practice disproportionately affects a protected group. It provides a clear, measurable target for equitable outcomes.

**Weaknesses**: Demographic parity can require selecting less qualified candidates from one group over more qualified candidates from another, which some view as unfair to individuals. It also ignores the possibility that groups may have legitimately different qualification rates due to historical or structural factors, meaning that equal selection rates might not reflect equal treatment of equally qualified individuals.

#### Equalized Odds

Equalized odds requires that the algorithm's true positive rate and false positive rate be equal across groups. In other words, among people who truly deserve the positive outcome (qualified candidates, creditworthy borrowers), the algorithm should select them at the same rate regardless of group membership. Similarly, among people who do not deserve the positive outcome, the algorithm should incorrectly select them at the same rate regardless of group.

Formally, equalized odds requires that Y-hat be conditionally independent of A given the true outcome Y. That is, P(Y-hat = 1 | A = a, Y = y) is the same for all values of a, for each value of y.

**Strengths**: Equalized odds accounts for the true qualification rate of each group, avoiding the issue of ignoring base rate differences. It ensures that the algorithm is equally accurate for each group in a meaningful sense -- equally good at identifying deserving candidates and equally unlikely to incorrectly flag undeserving ones.

**Weaknesses**: Achieving equalized odds may require sacrificing some overall accuracy. In the common case where groups have different base rates (different true positive rates in the population), achieving equalized odds requires accepting different thresholds for different groups, which can be seen as unfair to individuals.

#### Equal Opportunity

Equal opportunity is a relaxation of equalized odds that focuses only on the true positive rate. It requires that among people who truly deserve the positive outcome, the algorithm selects them at the same rate regardless of group membership. It does not constrain the false positive rate.

Formally, equal opportunity requires that P(Y-hat = 1 | A = a, Y = 1) is the same for all values of a.

**Strengths**: Equal opportunity focuses on ensuring that qualified individuals have equal chances of being identified, regardless of group membership. This captures a strong intuition about fairness: people who merit the outcome should have an equal opportunity to receive it.

**Weaknesses**: By not constraining false positive rates, equal opportunity allows the algorithm to impose unequal burdens on different groups among those who do not merit the positive outcome. This can lead to disparities in the harm caused by incorrect predictions.

#### Predictive Parity (Calibration)

Predictive parity requires that the algorithm's predictions be equally accurate across groups, in the sense that a given predicted probability means the same thing for all groups. If the algorithm assigns a 70% risk score, approximately 70% of people with that score should actually experience the predicted outcome, regardless of their group membership.

Formally, predictive parity requires that the true outcome Y be conditionally independent of A given the prediction Y-hat. That is, P(Y = 1 | Y-hat = s, A = a) is the same for all values of a, for each score value s.

**Strengths**: Calibration is the fairness criterion most closely aligned with accuracy. A calibrated model provides useful, interpretable predictions that mean the same thing for everyone. Decision-makers can use the predictions without needing to adjust for group membership.

**Weaknesses**: Calibration is compatible with significant disparities in error rates between groups. A calibrated model can have very different false positive and false negative rates for different groups, which is the pattern observed in the COMPAS controversy.

### The Impossibility Theorems

One of the most important findings in algorithmic fairness research is that these group fairness definitions generally cannot all be satisfied simultaneously. Several impossibility theorems formalize this insight.

The most well-known result shows that when group base rates differ (that is, when the true positive rate is different for different groups), it is mathematically impossible to simultaneously achieve calibration and equalized odds, except in trivial cases (a perfect predictor or a constant predictor).

This impossibility result has profound implications. It means that when we design a fair algorithm, we must choose which fairness criteria to prioritize. There is no algorithm that can be fair in every sense simultaneously. The choice of which criteria to prioritize is a value judgment, not a technical decision.

### Individual Fairness

Individual fairness takes a fundamentally different approach from group fairness. Rather than requiring equal statistics across groups, individual fairness requires that similar individuals receive similar predictions or decisions. The intuition is straightforward: if two people are similar in all relevant respects, the algorithm should treat them similarly, regardless of their group membership.

Formally, individual fairness requires that for a distance metric d that captures meaningful similarity between individuals, the algorithm's outputs for similar individuals should be close. If d(x_i, x_j) is small, then d(f(x_i), f(x_j)) should also be small.

**Strengths**: Individual fairness avoids some of the problems with group fairness. It does not require defining groups or choosing between competing group-level criteria. It focuses on treating each person as an individual rather than as a member of a category.

**Weaknesses**: Individual fairness requires defining a distance metric that captures "relevant similarity" -- and this is enormously difficult. What makes two job applicants "similar"? If we define similarity based on resume features, we may encode historical biases (since access to education and experience opportunities is unequal). If we define similarity based on underlying potential, we face the problem that potential is unobservable. The choice of distance metric effectively encodes all the normative judgments that group fairness criteria make explicit.

### Causal Fairness

Causal fairness definitions use causal reasoning to distinguish between legitimate and illegitimate influences on predictions. The core idea is that a prediction can depend on the protected attribute if the causal pathway goes through legitimate factors, but not if it goes through illegitimate ones.

For example, in a hiring context, gender might legitimately affect predictions if it goes through the causal pathway gender -> relevant experience -> prediction (because gender-related experiences might genuinely affect job qualifications). But gender should not affect predictions through the pathway gender -> name -> prediction (because a gendered name does not affect job qualifications).

**Counterfactual fairness**: The most formal version of causal fairness, counterfactual fairness, requires that an individual's prediction would be the same in a counterfactual world where they belonged to a different group. If a Black applicant would have received the same decision had they been White (holding all else equal in the appropriate causal sense), the decision is counterfactually fair.

**Strengths**: Causal fairness directly addresses the mechanisms through which protected attributes influence outcomes, allowing for more nuanced fairness assessments than group-level statistics.

**Weaknesses**: Causal fairness requires specifying a causal model of the relevant domain, which is often contentious and difficult to validate. Different assumptions about causal relationships lead to different fairness assessments. The counterfactual thought experiment -- "what would have happened if this person had been a different race?" -- is philosophically complex and may not have a well-defined answer.

## Practical Approaches to Fairness

Given the theoretical complexity and impossibility results, how do practitioners actually build fairer systems? Several practical approaches have emerged.

### Pre-Processing Methods

Pre-processing methods modify the training data before model training to reduce bias. Techniques include reweighting training examples to equalize the influence of different groups, transforming features to remove correlations with protected attributes, relabeling training examples to correct for historical biases, and generating synthetic data to balance representation across groups.

Pre-processing methods have the advantage of being model-agnostic -- they can be applied regardless of the model architecture used for training. However, they can also degrade overall model performance and may not fully address bias that enters through complex feature interactions.

### In-Processing Methods

In-processing methods modify the model training procedure to incorporate fairness constraints. Rather than optimizing solely for predictive accuracy, the training objective includes a fairness term that penalizes violations of the chosen fairness criteria.

Common in-processing techniques include adversarial debiasing (training a model that is accurate while simultaneously being unable to predict the protected attribute), constrained optimization (adding fairness constraints to the optimization problem), and regularization approaches that penalize unfair predictions during training.

In-processing methods can achieve better trade-offs between accuracy and fairness than pre-processing methods because they directly optimize for both objectives simultaneously. However, they are more complex to implement and may require significant modifications to standard training pipelines.

### Post-Processing Methods

Post-processing methods adjust the model's outputs after training to improve fairness. The simplest approach is group-specific thresholding: rather than using the same decision threshold for all groups, different thresholds are calibrated for each group to achieve the desired fairness property.

For example, if a model produces well-calibrated probabilities but has different false positive rates for different groups, adjusting the decision threshold for each group can equalize false positive rates. This is straightforward to implement and does not require retraining the model.

Post-processing methods are practical and easy to deploy, but they can reduce overall accuracy and raise concerns about treating individuals differently based on group membership.

### Fairness-Aware Model Selection

Rather than modifying a single model, practitioners can train multiple models with different architectures, features, and hyperparameters, then select the model that achieves the best combination of accuracy and fairness. This model selection approach leverages the natural variation in fairness properties across different models and avoids explicit fairness interventions.

## The Fairness-Accuracy Trade-Off

A common concern about algorithmic fairness is that it requires sacrificing accuracy. This concern is valid in some cases but overstated in many others.

In cases where group base rates differ significantly and the fairness criterion requires equalizing error rates, there is a genuine trade-off between fairness and accuracy. Achieving equalized odds when base rates differ requires accepting some loss of overall accuracy.

However, several important nuances complicate the simple trade-off narrative.

First, the accuracy being sacrificed is accuracy measured against potentially biased labels. If historical data is biased, a model that accurately reproduces those biases is "accurate" only in the sense of replicating discrimination. Sacrificing this kind of accuracy to achieve fairer outcomes may actually produce a system that is more accurate in a deeper sense -- more aligned with true underlying merit rather than biased historical judgments.

Second, the trade-off is often smaller than feared. In many practical applications, models can be made substantially fairer with only modest reductions in overall accuracy. The Pareto frontier of fairness-accuracy trade-offs often reveals that significant fairness improvements are achievable at little cost to accuracy.

Third, fairness constraints can sometimes improve generalization. Models that are forced to achieve consistent performance across groups may learn more robust features that generalize better to new data, particularly data from underrepresented groups.

![Trade-off between prediction accuracy and fairness in ML systems](/images/blogs/pool-ethics/5.jpg)

## Context-Dependent Fairness

One of the most important insights in algorithmic fairness is that the right definition of fairness depends on the context. Different applications, different stakes, and different social contexts may warrant different fairness criteria.

### Criminal Justice

In criminal justice, the stakes are extraordinarily high -- predictions influence decisions about liberty and incarceration. The COMPAS controversy highlighted the tension between calibration (a score means the same thing regardless of race) and equalized error rates (errors are distributed equally across races). The choice between these criteria involves deep questions about the purpose of the criminal justice system, the relative harms of false positives and false negatives, and the role of historical injustice in current outcomes.

### Lending

In lending, fairness concerns intersect with economic considerations and regulatory requirements. The Equal Credit Opportunity Act prohibits discrimination in lending, and the disparate impact framework provides a legal standard for evaluating fairness. Calibration is often emphasized in lending (a predicted default probability should be accurate regardless of group) because lenders need accurate risk assessments to maintain financial viability.

### Hiring

In hiring, fairness considerations include legal requirements (Title VII of the Civil Rights Act), practical concerns about workforce diversity, and the difficulty of measuring "true qualification" for a job. The four-fifths rule (adverse impact is presumed when the selection rate for a protected group is less than 80% of the rate for the highest-selected group) provides a simple but crude benchmark.

### Healthcare

In healthcare, fairness concerns focus on ensuring equal access to care and equal quality of predictions across patient populations. A well-known example revealed that a widely used healthcare algorithm was less likely to refer Black patients for additional care because it used healthcare spending as a proxy for health needs, and systemic inequities meant that less money was spent on Black patients for the same level of health need.

## Intersectionality and Subgroup Fairness

Most fairness definitions focus on fairness across a single protected attribute (race or gender). But individuals belong to multiple groups simultaneously -- a Black woman faces different experiences than a Black man or a White woman. Intersectional fairness asks whether algorithmic systems are fair for all subgroups defined by combinations of protected attributes.

Intersectional fairness is both important and technically challenging. The number of subgroups grows combinatorially with the number of protected attributes considered. Small subgroups may have too few members in the training data for reliable statistical analysis. And fairness criteria that are satisfied for each protected attribute individually may be violated for intersectional subgroups.

Despite these challenges, intersectional fairness is increasingly recognized as essential. Bias that affects intersectional subgroups can be invisible when fairness is assessed only along single dimensions.

## Organizational and Process Approaches

Technical methods for achieving algorithmic fairness are necessary but not sufficient. Organizational and process approaches are equally important.

### Fairness Audits

Regular audits of algorithmic systems for fairness, conducted by teams that include diverse perspectives and domain expertise, can identify bias that technical metrics alone might miss. Audits should examine not just the algorithm's outputs but its inputs (data sources and quality), its design decisions (feature selection, model architecture, fairness criteria), and its real-world impacts.

### Stakeholder Engagement

People affected by algorithmic decisions should have a voice in how those systems are designed and evaluated. Community engagement, participatory design, and public comment processes can surface fairness concerns that technical teams might overlook and ensure that the chosen fairness criteria reflect the values and priorities of affected communities.

### Documentation and Transparency

Model cards, datasheets for datasets, and algorithmic impact assessments provide structured documentation of the design decisions, assumptions, fairness evaluations, and limitations of algorithmic systems. This documentation enables external scrutiny and supports accountability.

### Ongoing Monitoring

Fairness is not a one-time property. As populations change, data distributions shift, and societal context evolves, a system that was fair at deployment may become unfair over time. Continuous monitoring of fairness metrics in production systems is essential for maintaining fairness over time.

![Organizational approaches to building fairer AI systems](/images/blogs/pool-ethics/7.jpg)

## The Path Forward

Algorithmic fairness is a field that is simultaneously making rapid technical progress and confronting fundamental philosophical challenges. The impossibility theorems tell us that we cannot have it all -- we must choose which aspects of fairness to prioritize in each specific context. The proliferation of formal fairness definitions gives us precise language for articulating and comparing different notions of fairness. And the growing toolkit of practical methods gives us the ability to build systems that are meaningfully fairer than unconstrained ones.

But the hardest questions in algorithmic fairness are not technical. They are about values, priorities, and power. Who gets to decide which fairness criteria apply? How should we weigh fairness against other objectives like accuracy, efficiency, and cost? When does algorithmic decision-making create benefits that outweigh its fairness risks, and when should we reject algorithmic approaches entirely?

These questions do not have universal answers. They must be addressed through democratic deliberation, stakeholder engagement, and ongoing societal dialogue. What technology can offer is the analytical tools to make these deliberations more informed, more precise, and more accountable. The responsibility for deciding what fairness means in each context belongs not to algorithms but to the people and communities they affect.

Understanding the landscape of algorithmic fairness -- its formal definitions, impossibility results, practical methods, and contextual complexities -- is the foundation for engaging with these decisions wisely. In a world where algorithmic systems increasingly shape opportunities and outcomes, this understanding is not optional. It is a civic responsibility.
