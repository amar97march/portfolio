---
title: "The Problem of Bias in Data: Garbage In, Garbage Out"
date: 2026-05-21T10:00:00+05:30
draft: false
description: "An exploration of how bias enters machine learning systems through data — the types of bias, real-world consequences, and practical strategies for detection and mitigation."
tags: ["AI Ethics", "Bias", "Fairness", "Data Quality", "Responsible AI"]
categories: ["AI Ethics"]
image: "https://images.unsplash.com/photo-1544197150-b99a580bb7a8?w=1200&h=630&fit=crop&auto=format"
keywords: ["bias in AI", "data bias", "algorithmic fairness", "AI ethics", "biased training data"]
---

Machine learning models are only as good as the data they learn from. When that data reflects the biases, prejudices, and inequalities of the real world — which it almost always does — the model does not correct for those biases. It **amplifies** them.

This is not a theoretical concern. Biased AI systems are making decisions right now about who gets hired, who gets a loan, who gets parole, and who gets medical treatment. When these systems are biased, the consequences are real and harmful.

As ML practitioners, we have a responsibility to understand how bias enters our systems and what we can do about it.

## How Bias Enters Data

### Historical Bias

The training data reflects historical patterns that include discrimination. If you train a hiring model on a decade of hiring decisions, and those decisions were biased against women, the model will learn to discriminate against women. It is not malicious — it is faithfully reproducing the patterns it was trained on.

### Representation Bias

The training data does not represent all groups equally. ImageNet, one of the most influential datasets in computer vision, was sourced predominantly from the United States and Europe. Models trained on it perform significantly worse on images from other parts of the world.

### Measurement Bias

The way data is collected introduces systematic errors. In healthcare, certain conditions are diagnosed less frequently in women and minorities — not because they occur less often, but because of differences in healthcare access and diagnostic criteria. A model trained on diagnostic records inherits this measurement bias.

### Selection Bias

The data collection process systematically excludes certain groups. Surveys conducted online miss people without internet access. Datasets from top hospitals miss patients who cannot afford premium healthcare. Social media data overrepresents younger, more tech-savvy demographics.


![Illustration representing fairness and bias in AI systems](https://picsum.photos/seed/bias-in-data-garbage-in-garbage-out-1/800/450)

### Label Bias

The labels themselves are biased. If human annotators label resumes as "strong" or "weak" candidates, their unconscious biases affect the labels. If police arrest records are used as labels for crime prediction, they reflect policing patterns (which are biased) rather than actual crime rates.

## Types of Algorithmic Bias

### Allocation Bias

The model allocates opportunities or resources unfairly. A lending model that systematically denies loans to applicants from certain neighborhoods (which correlates with race) is exhibiting allocation bias.

### Quality-of-Service Bias

The model works better for some groups than others. Speech recognition systems that perform well on American English but poorly on Indian English or African American Vernacular English demonstrate quality-of-service bias.

### Stereotyping

The model reinforces stereotypes. Language models that associate "doctor" with "he" and "nurse" with "she" are encoding gender stereotypes from their training data.


![Visual depicting the ethical considerations of algorithmic decision-making](https://picsum.photos/seed/bias-in-data-garbage-in-garbage-out-2/800/450)

## Detecting Bias

### Disaggregated Evaluation

The most straightforward approach: evaluate your model separately on different demographic groups.

```python
from sklearn.metrics import accuracy_score, classification_report

def disaggregated_evaluation(y_true, y_pred, groups, group_name):
    """Evaluate model performance across different groups"""
    unique_groups = groups.unique()

    print(f"\n{'='*50}")
    print(f"Disaggregated Evaluation by {group_name}")
    print(f"{'='*50}")

    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {overall_acc:.4f}")

    group_metrics = {}
    for group in unique_groups:
        mask = groups == group
        group_acc = accuracy_score(y_true[mask], y_pred[mask])
        group_size = mask.sum()
        group_metrics[group] = {
            'accuracy': group_acc,
            'size': group_size
        }
        print(f"\n{group_name}={group}:")
        print(f"  Accuracy: {group_acc:.4f}")
        print(f"  Sample size: {group_size}")

    # Calculate disparity
    accuracies = [m['accuracy'] for m in group_metrics.values()]
    disparity = max(accuracies) - min(accuracies)
    print(f"\nAccuracy Disparity: {disparity:.4f}")

    return group_metrics

# Usage
metrics = disaggregated_evaluation(
    y_test, predictions,
    test_data['gender'], 'Gender'
)
```

### Fairness Metrics

```python
def compute_fairness_metrics(y_true, y_pred, protected_attribute):
    """Compute common fairness metrics"""
    groups = protected_attribute.unique()
    metrics = {}

    for group in groups:
        mask = protected_attribute == group
        y_t = y_true[mask]
        y_p = y_pred[mask]

        tp = ((y_t == 1) & (y_p == 1)).sum()
        fp = ((y_t == 0) & (y_p == 1)).sum()
        tn = ((y_t == 0) & (y_p == 0)).sum()
        fn = ((y_t == 1) & (y_p == 0)).sum()

        metrics[group] = {
            'positive_rate': (y_p == 1).mean(),  # Demographic parity
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Equal opportunity
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        }

    # Demographic Parity: positive prediction rates should be equal
    rates = [m['positive_rate'] for m in metrics.values()]
    dp_ratio = min(rates) / max(rates) if max(rates) > 0 else 0

    # Equal Opportunity: true positive rates should be equal
    tpr = [m['true_positive_rate'] for m in metrics.values()]
    eo_ratio = min(tpr) / max(tpr) if max(tpr) > 0 else 0

    print(f"Demographic Parity Ratio: {dp_ratio:.4f} (1.0 = perfect parity)")
    print(f"Equal Opportunity Ratio: {eo_ratio:.4f} (1.0 = perfect equality)")

    return metrics
```

## Mitigation Strategies

### Pre-processing: Fix the Data

**Resampling**: Balance the representation of different groups.

**Relabeling**: Correct labels that reflect historical bias.

**Feature removal**: Remove sensitive attributes and their proxies (though this is not always sufficient — see "proxy discrimination" below).

```python
# Resampling to balance group representation
from sklearn.utils import resample

# Ensure equal representation of groups
balanced_dfs = []
for group in df['demographic'].unique():
    group_df = df[df['demographic'] == group]
    resampled = resample(group_df, n_samples=target_size, random_state=42)
    balanced_dfs.append(resampled)

balanced_df = pd.concat(balanced_dfs)
```


![Conceptual image showing the balance between AI power and responsibility](https://picsum.photos/seed/bias-in-data-garbage-in-garbage-out-3/800/450)

### In-processing: Constrain the Model

**Adversarial debiasing**: Train a second model to predict the protected attribute from the main model's predictions. If it can, the main model is leaking bias. Penalize this.

**Fairness constraints**: Add fairness metrics as constraints during optimization.

### Post-processing: Adjust the Outputs

**Threshold adjustment**: Use different classification thresholds for different groups to equalize error rates.

```python
# Adjust thresholds per group for equal opportunity
def find_fair_thresholds(y_true, y_scores, groups, target_tpr=0.8):
    thresholds = {}
    for group in groups.unique():
        mask = groups == group
        group_true = y_true[mask]
        group_scores = y_scores[mask]

        # Find threshold that achieves target TPR for this group
        best_threshold = 0.5
        best_diff = float('inf')
        for t in np.arange(0.1, 0.9, 0.01):
            preds = (group_scores >= t).astype(int)
            tp = ((group_true == 1) & (preds == 1)).sum()
            fn = ((group_true == 1) & (preds == 0)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            if abs(tpr - target_tpr) < best_diff:
                best_diff = abs(tpr - target_tpr)
                best_threshold = t

        thresholds[group] = best_threshold
    return thresholds
```

## The Proxy Problem

Simply removing sensitive attributes (race, gender, age) is not sufficient. Other features often serve as **proxies** for protected characteristics:

- **ZIP code** correlates with race in many countries
- **Name** correlates with gender and ethnicity
- **University attended** correlates with socioeconomic background
- **Hobbies** can correlate with gender

A model can discriminate without ever seeing the protected attribute directly, by using these proxy features. This makes bias mitigation much harder than simply deleting a column.

## Fairness is Not One Thing

One of the most challenging aspects of algorithmic fairness is that different definitions of "fairness" can be **mathematically incompatible**. You often cannot satisfy all fairness criteria simultaneously.

**Demographic Parity**: The model should make positive predictions at the same rate for all groups. But this ignores base rates — if one group genuinely has higher qualification rates, forcing equal acceptance rates is unfair to both groups.

**Equal Opportunity**: The model should have the same true positive rate for all groups. But this allows different false positive rates.

**Predictive Parity**: The model should have the same precision for all groups. But this is mathematically incompatible with equal opportunity when base rates differ.

There is no universally correct definition of fairness. The appropriate definition depends on the context, the stakes, and the values of the stakeholders.

## What We Can Do as Practitioners

1. **Audit your data** before building models. Look at demographic breakdowns, label distributions, and collection methodology.

2. **Evaluate disaggregated metrics**. Overall accuracy can hide significant disparities between groups.

3. **Document your data and models**. Data sheets and model cards help others understand the limitations and potential biases of your systems.

4. **Include diverse perspectives** in your team and your review process. Bias is easier to spot when multiple viewpoints are represented.

5. **Monitor deployed models** for bias drift. Even a fair model can become biased as the world changes.

6. **Be transparent** about limitations. No model is perfectly fair. Acknowledging this is better than pretending otherwise.

7. **Consider whether ML is appropriate** for the task. Some decisions are too consequential and too prone to bias for automated systems.

Bias in AI is not a problem that technology alone can solve. It is fundamentally a human problem that requires human judgment, diverse perspectives, and ongoing vigilance.

---

*Next: A concrete case study of how a biased dataset led to a failed AI product.*
