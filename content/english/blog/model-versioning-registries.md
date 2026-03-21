---
title: "Model Versioning and Registries: Managing Your ML Models"
date: 2027-04-19T09:00:00+05:30
draft: false
description: "A model registry is the single source of truth for your trained models. Learn why model versioning matters, how model registries work, and how to implement one using MLflow's Model Registry."
tags: ["MLOps", "Model Registry", "Model Versioning", "MLflow", "Machine Learning", "Production ML"]
categories: ["MLOps"]
image: "https://picsum.photos/seed/model-versioning-registries-cover/1200/630"
keywords: ["model versioning", "model registry", "mlflow model registry", "ml model management", "model lifecycle"]
---

You have trained 47 models over the past three months. Some used XGBoost, some used neural networks. Some were trained on the full dataset, others on a filtered subset. Some had custom feature engineering, others used raw features.

Now a critical question: which exact model is serving predictions in production right now?

If you cannot answer that question in under 30 seconds, you need a model registry.

---

## The Problem: Model Chaos

Without a model registry, teams fall into predictable patterns of chaos:

- **The shared drive pattern**: Models are saved as files in a shared folder. Naming conventions are inconsistent. `model_final.pkl`, `model_v2_johns_version.pkl`, `best_model_DO_NOT_DELETE.h5`.
- **The notebook pattern**: Models exist only inside Jupyter notebooks. Deploying means someone manually exports the model from a notebook and copies it somewhere.
- **The "it works on my machine" pattern**: A data scientist has a model that scores 95% accuracy on their laptop. No one else can reproduce the result because the exact environment and data are unknown.
- **The mystery production model**: A model is running in production, but no one knows exactly how it was trained, what data it used, or who deployed it.

These patterns are not just inconvenient. They are dangerous. When a model starts making bad predictions, you need to know its exact lineage to diagnose the problem. When a regulator asks for an audit, you need to produce the complete history of a model.

---


![Machine learning operations and infrastructure](https://picsum.photos/seed/model-versioning-registries-1/800/450)

## What is a Model Registry?

A **model registry** is a centralized repository that manages the full lifecycle of ML models. Think of it as a combination of:

- A **package registry** (like npm or PyPI) for ML models
- A **deployment tracker** that knows which model version is in production
- An **audit log** that records every model's lineage and history

A model registry stores:

| Information | Purpose |
|-------------|---------|
| Model artifacts | The trained model files (weights, parameters) |
| Metadata | Training data version, hyperparameters, git commit |
| Metrics | Performance metrics from evaluation |
| Stage/Status | Whether the model is in staging, production, or archived |
| Lineage | Full history of how the model was created |
| Tags and descriptions | Human-readable context about the model |

---

## Model Lifecycle Stages

Models in a registry typically move through well-defined stages:

```
                 ┌──────────┐
                 │  None     │  (Just registered)
                 └─────┬────┘
                       │
                       ▼
                 ┌──────────┐
                 │ Staging   │  (Being tested/validated)
                 └─────┬────┘
                       │
                       ▼
                 ┌──────────┐
                 │Production │  (Serving live traffic)
                 └─────┬────┘
                       │
                       ▼
                 ┌──────────┐
                 │ Archived  │  (Retired, kept for audit)
                 └──────────┘
```

Transitioning between stages can be manual (a human approves the promotion) or automated (a CI/CD pipeline promotes the model if it passes all validation checks).

---


![Production ML pipeline and deployment workflow](https://picsum.photos/seed/model-versioning-registries-2/800/450)

## Implementing a Model Registry with MLflow

MLflow is the most widely adopted open-source tool for model registries. Let us walk through a complete workflow.

### Registering a Model After Training

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("customer_churn")

with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("data_version", "v3.2.0")
    mlflow.log_param("git_commit", "abc123def")

    # Train model
    model = RandomForestClassifier(n_estimators=200, max_depth=10)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Register the model in the registry
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="customer_churn_model"
    )
```

### Managing Model Stages

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get the latest version of the model
latest_version = client.get_latest_versions(
    "customer_churn_model",
    stages=["None"]
)[0]

print(f"Latest version: {latest_version.version}")
print(f"Run ID: {latest_version.run_id}")

# Promote to staging
client.transition_model_version_stage(
    name="customer_churn_model",
    version=latest_version.version,
    stage="Staging"
)

# After validation, promote to production
client.transition_model_version_stage(
    name="customer_churn_model",
    version=latest_version.version,
    stage="Production"
)

# Archive the old production model
client.transition_model_version_stage(
    name="customer_churn_model",
    version=old_version,
    stage="Archived"
)
```

### Loading a Model from the Registry

```python
import mlflow

# Load the current production model
model = mlflow.pyfunc.load_model(
    model_uri="models:/customer_churn_model/Production"
)

# Load a specific version
model_v3 = mlflow.pyfunc.load_model(
    model_uri="models:/customer_churn_model/3"
)

# Use it for prediction
predictions = model.predict(new_data)
```

This is powerful. Your serving code does not need to know the file path of the model. It just asks the registry for the production version. When you promote a new model, the serving code automatically picks it up.

---

## Automating Model Validation Before Promotion

A robust MLOps pipeline does not rely on humans to validate models. Instead, you build automated validation gates:

```python
def validate_model_for_production(model_name, new_version):
    """Automated validation before promoting a model to production."""
    client = MlflowClient()

    # Load the candidate model
    candidate = mlflow.pyfunc.load_model(
        f"models:/{model_name}/{new_version}"
    )

    # Load the current production model for comparison
    try:
        production = mlflow.pyfunc.load_model(
            f"models:/{model_name}/Production"
        )
        has_production = True
    except Exception:
        has_production = False

    # Run validation checks
    validation_data = load_validation_dataset()
    candidate_metrics = evaluate(candidate, validation_data)

    # Check 1: Minimum performance thresholds
    if candidate_metrics["accuracy"] < 0.85:
        return False, "Accuracy below minimum threshold (0.85)"

    if candidate_metrics["latency_p99_ms"] > 100:
        return False, "P99 latency exceeds 100ms"

    # Check 2: Must outperform current production model
    if has_production:
        prod_metrics = evaluate(production, validation_data)
        if candidate_metrics["accuracy"] < prod_metrics["accuracy"]:
            return False, "Does not outperform current production model"

    # Check 3: Fairness checks
    for group in ["group_a", "group_b", "group_c"]:
        group_accuracy = evaluate_group(candidate, validation_data, group)
        if group_accuracy < 0.80:
            return False, f"Accuracy below 0.80 for {group}"

    return True, "All validation checks passed"
```

---


![MLOps tooling and automation systems](https://picsum.photos/seed/model-versioning-registries-3/800/450)

## Model Registry Best Practices

### 1. Establish a Naming Convention

Use consistent, descriptive names for registered models:

```
{team}_{use_case}_{model_type}
# Examples:
# fraud_detection_xgboost
# recommendation_collaborative_filter
# churn_prediction_neural_net
```

### 2. Always Log Lineage Information

Every registered model should have complete lineage:

```python
mlflow.log_param("training_data_version", data_version)
mlflow.log_param("git_commit", get_git_commit())
mlflow.log_param("feature_pipeline_version", feature_version)
mlflow.log_param("trained_by", os.environ.get("USER"))
mlflow.log_param("training_duration_minutes", duration)
```

### 3. Add Descriptions and Tags

```python
client.update_model_version(
    name="customer_churn_model",
    version=new_version,
    description="Trained on Q1 2027 data with new engagement features. "
                "Improved F1 by 3.2% over previous version."
)

client.set_model_version_tag(
    name="customer_churn_model",
    version=new_version,
    key="validation_status",
    value="passed"
)
```

### 4. Never Delete Production Model History

Even when archiving old models, keep them in the registry. Regulatory audits, debugging production issues, and understanding model evolution all require historical records.

### 5. Integrate with CI/CD

Model promotion should be part of your CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
name: Model Promotion
on:
  workflow_dispatch:
    inputs:
      model_version:
        description: "Model version to promote"
        required: true

jobs:
  validate-and-promote:
    runs-on: ubuntu-latest
    steps:
      - name: Run validation suite
        run: python scripts/validate_model.py --version ${{ inputs.model_version }}

      - name: Run integration tests
        run: python scripts/integration_test.py --version ${{ inputs.model_version }}

      - name: Promote to production
        if: success()
        run: python scripts/promote_model.py --version ${{ inputs.model_version }}
```

---

## Comparing Model Registry Tools

| Feature | MLflow | SageMaker Registry | Vertex AI Registry | Azure ML Registry |
|---------|--------|-------------------|-------------------|-------------------|
| Open Source | Yes | No | No | No |
| Cloud Agnostic | Yes | AWS only | GCP only | Azure only |
| Stage Management | Yes | Yes | Yes | Yes |
| Automated Approval | Via API | Built-in | Built-in | Built-in |
| Serving Integration | Manual | SageMaker Endpoints | Vertex Endpoints | Azure Endpoints |
| Cost | Free (self-hosted) | Pay-per-use | Pay-per-use | Pay-per-use |

---

## Conclusion

A model registry is the backbone of production ML. It answers the questions that matter: What model is running? How was it trained? Is it still performing well? Can we safely roll back?

Without a registry, you are flying blind. With one, you have complete visibility and control over your model lifecycle.

In the next post, we will explore the other critical tracking mechanism in MLOps: experiment tracking, and why every ML project, no matter how small, needs it from day one.
