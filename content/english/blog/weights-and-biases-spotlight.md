---
title: "Weights & Biases: Enterprise ML Experiment Tracking"
date: 2027-05-01T09:00:00+05:30
draft: false
description: "Weights & Biases (W&B) has become the industry favorite for ML experiment tracking, offering rich visualizations, seamless collaboration, and powerful hyperparameter optimization. Learn how W&B works and when to choose it over alternatives."
tags: ["MLOps", "Weights and Biases", "Experiment Tracking", "Machine Learning", "W&B", "Hyperparameter Tuning"]
categories: ["MLOps"]
image: "https://picsum.photos/seed/weights-and-biases-spotlight-cover/1200/630"
keywords: ["weights and biases tutorial", "wandb ml tracking", "w&b experiment tracking", "wandb vs mlflow", "ml experiment visualization"]
---

Weights & Biases (commonly abbreviated as W&B or wandb) has earned a devoted following in the ML community. While MLflow is the pragmatic workhorse, W&B is the tool that data scientists genuinely enjoy using. Its rich dashboards, real-time collaboration, and seamless integrations have made it the go-to choice for teams that prioritize visibility and communication in their ML workflows.

Let us explore what makes W&B special and how to use it effectively.

---

## Why W&B Stands Out

W&B differentiates itself in several key areas:

1. **Visualization**: Auto-generated dashboards, interactive charts, and custom panels that make experiment comparison intuitive.
2. **Collaboration**: Shared workspaces where teams can annotate, discuss, and compare experiments in real time.
3. **Integrations**: Native support for PyTorch, TensorFlow, Keras, Hugging Face, scikit-learn, XGBoost, and many more.
4. **Sweeps**: Built-in hyperparameter optimization with Bayesian, grid, and random search strategies.
5. **Artifacts**: Versioned dataset and model tracking with lineage graphs.
6. **Reports**: Interactive, shareable documents combining code, visualizations, and narrative.

---

## Getting Started with W&B

### Installation and Setup

```bash
pip install wandb
wandb login  # Authenticate with your API key
```

### Your First Tracked Experiment

```python
import wandb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Initialize a run
wandb.init(
    project="credit-scoring",
    name="gradient_boosting_v1",
    config={
        "model_type": "gradient_boosting",
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "dataset_version": "v2.1",
    }
)

# Train your model
model = GradientBoostingClassifier(
    n_estimators=wandb.config.n_estimators,
    max_depth=wandb.config.max_depth,
    learning_rate=wandb.config.learning_rate,
)
model.fit(X_train, y_train)

# Evaluate and log metrics
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

wandb.log({
    "accuracy": accuracy_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "auc_roc": roc_auc_score(y_test, y_proba),
})

# Finish the run
wandb.finish()
```

That is it. Navigate to your W&B project dashboard and you will see your run with all metrics, system stats (GPU usage, CPU, memory), and even the git diff of your code.

---

![Getting started with Weights and Biases experiment tracking](https://picsum.photos/seed/weights-and-biases-spotlight-1/800/450)

## Logging Rich Data

One of W&B's strengths is its ability to log rich data types beyond simple numbers.

### Training Curves

```python
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
    })
```

W&B automatically generates interactive line charts for these metrics, overlaid across runs for easy comparison.

### Confusion Matrices and ROC Curves

```python
# Log a confusion matrix
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        y_true=y_test,
        preds=y_pred,
        class_names=["legitimate", "fraud"]
    )
})

# Log an ROC curve
wandb.log({
    "roc_curve": wandb.plot.roc_curve(
        y_true=y_test,
        y_probas=model.predict_proba(X_test),
        labels=["legitimate", "fraud"]
    )
})
```

### Images, Audio, and Tables

```python
# Log images (useful for computer vision)
images = [wandb.Image(img, caption=f"Prediction: {pred}")
          for img, pred in zip(sample_images, predictions)]
wandb.log({"predictions": images})

# Log a table of predictions for inspection
table = wandb.Table(
    columns=["input", "prediction", "ground_truth", "confidence"],
    data=prediction_samples
)
wandb.log({"prediction_samples": table})
```

---

## W&B Sweeps: Hyperparameter Optimization

W&B Sweeps provide a declarative way to run hyperparameter searches:

### Define the Sweep Configuration

```python
sweep_config = {
    "method": "bayes",  # bayesian optimization
    "metric": {
        "name": "val/f1_score",
        "goal": "maximize"
    },
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 0.0001,
            "max": 0.1,
        },
        "n_estimators": {
            "values": [100, 200, 300, 500]
        },
        "max_depth": {
            "distribution": "int_uniform",
            "min": 3,
            "max": 15,
        },
        "subsample": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.0,
        }
    }
}
```

### Define the Training Function

```python
def train():
    wandb.init()
    config = wandb.config

    model = xgb.XGBClassifier(
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        subsample=config.subsample,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    wandb.log({"val/f1_score": f1})
    wandb.finish()
```

### Launch the Sweep

```python
sweep_id = wandb.sweep(sweep_config, project="credit-scoring")
wandb.agent(sweep_id, function=train, count=50)
```

W&B runs 50 trials with Bayesian optimization, automatically selecting promising hyperparameter combinations based on previous results. The dashboard shows parallel coordinate plots, parameter importance analysis, and a leaderboard of all trials.

---

![Hyperparameter optimization with W&B Sweeps](https://picsum.photos/seed/weights-and-biases-spotlight-2/800/450)

## W&B Artifacts: Dataset and Model Versioning

W&B Artifacts provide built-in versioning for datasets and models:

```python
# Log a dataset as an artifact
artifact = wandb.Artifact("training_data", type="dataset")
artifact.add_file("data/train.csv")
artifact.add_file("data/test.csv")
wandb.log_artifact(artifact)

# Log a model as an artifact
model_artifact = wandb.Artifact("credit_model", type="model")
model_artifact.add_file("models/model.pkl")
wandb.log_artifact(model_artifact)
```

```python
# In a later run, use a specific artifact version
run = wandb.init(project="credit-scoring")
artifact = run.use_artifact("training_data:v3")
artifact_dir = artifact.download()

# W&B tracks the lineage: which runs produced and consumed which artifacts
```

The artifact lineage graph in the W&B UI shows the complete provenance chain: which dataset version was used to train which model version, which was evaluated on which test set, and so on.

---

## W&B Reports: Sharing Results

Reports are interactive documents that combine narrative text, embedded charts from your experiments, and code:

```python
# Reports are typically created in the W&B UI, but you can also
# programmatically create them using the API

import wandb
api = wandb.Api()

# Fetch runs for a report
runs = api.runs("your-team/credit-scoring", filters={"state": "finished"})
for run in runs:
    print(f"{run.name}: F1={run.summary.get('f1_score', 'N/A')}")
```

Reports are particularly valuable for:

- **Weekly team updates**: Share experiment progress with the team.
- **Stakeholder communication**: Present results to non-technical audiences with curated visualizations.
- **Decision documentation**: Record why a particular model was chosen for production.
- **Knowledge base**: Build a library of what was tried, what worked, and what did not.

---

## Integration with Deep Learning Frameworks

### PyTorch

```python
import wandb

wandb.init(project="image-classification")

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            wandb.log({"train_loss": loss.item()})

    # Log learning rate
    wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})
```

### Hugging Face Transformers

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",  # This single line enables W&B integration
    num_train_epochs=3,
    per_device_train_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

The Hugging Face integration automatically logs all training metrics, model configuration, and even sample predictions.

---

![Comparing W&B and MLflow for ML experiment management](https://picsum.photos/seed/weights-and-biases-spotlight-3/800/450)

## W&B vs MLflow: When to Choose Which

| Aspect | MLflow | W&B |
|--------|--------|-----|
| **Cost** | Free (self-hosted) | Free tier + paid plans |
| **Hosting** | Self-hosted or Databricks | Cloud-hosted (SaaS) or self-hosted |
| **Visualization** | Basic but functional | Rich and interactive |
| **Collaboration** | Limited | Strong (shared workspaces, reports) |
| **Model Registry** | Yes (built-in) | Yes (via Model Registry feature) |
| **Hyperparameter Optimization** | No (use Optuna separately) | Yes (built-in Sweeps) |
| **Vendor Lock-in** | None | Some (data in W&B cloud) |
| **Setup Effort** | Medium (need to host server) | Low (cloud-hosted) |

**Choose MLflow when**: You need full control, want no vendor dependency, are cost-sensitive, or are in a regulated environment that requires self-hosting.

**Choose W&B when**: You want the best-in-class visualization and collaboration, your team values sharing and discussing experiments, and you are comfortable with a SaaS tool.

Many teams use both: MLflow for the model registry and deployment pipeline, and W&B for experiment tracking and visualization during development.

---

## Conclusion

Weights & Biases has earned its popularity by making experiment tracking not just functional but enjoyable. Its visualizations surface insights that would take hours to find in raw logs, and its collaboration features bring transparency to the often opaque process of model development.

Whether you choose W&B, MLflow, or both, the important thing is that you track your experiments systematically. The specific tool matters less than the discipline of logging everything, comparing rigorously, and documenting decisions.

In the next post, we will explore Kubeflow, the platform that brings Kubernetes-native orchestration to ML pipelines, enabling scalable, reproducible workflows on any infrastructure.
