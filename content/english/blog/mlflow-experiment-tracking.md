---
title: "MLflow: The Open-Source Experiment Tracking Platform"
date: 2027-04-28T09:00:00+05:30
draft: false
description: "MLflow is the most widely adopted open-source platform for ML experiment tracking, model registry, and deployment. Learn how to set it up, use its core components, and integrate it into your ML workflow."
tags: ["MLOps", "MLflow", "Experiment Tracking", "Model Registry", "Machine Learning", "Python"]
categories: ["MLOps"]
image: "/images/blogs/pool-mlops/1.jpg"
keywords: ["mlflow tutorial", "mlflow experiment tracking", "mlflow setup", "mlflow model registry", "mlflow guide"]
---

If you had to pick one tool to start your MLOps journey, MLflow would be the safest bet. It is open-source, vendor-neutral, backed by Databricks, and has become the de facto standard for experiment tracking in the ML community.

MLflow is not a single tool. It is a platform with four core components, each addressing a different aspect of the ML lifecycle. In this post, we will explore each component hands-on and build a complete workflow from experiment to registered model.

---

## The Four Components of MLflow

```
┌──────────────────────────────────────────────────┐
│                    MLflow                         │
├──────────────┬──────────────┬──────────┬─────────┤
│  Tracking    │   Projects   │  Models  │ Registry│
│              │              │          │         │
│ Log params,  │ Reproducible │ Package  │ Version │
│ metrics,     │ ML code      │ models   │ and     │
│ artifacts    │ packaging    │ for any  │ stage   │
│              │              │ platform │ models  │
└──────────────┴──────────────┴──────────┴─────────┘
```

### 1. MLflow Tracking

The most used component. It provides an API and UI for logging parameters, metrics, code versions, and artifacts during ML experiments.

### 2. MLflow Projects

A standard format for packaging ML code into reusable, reproducible projects. Think of it as a Dockerfile for ML experiments.

### 3. MLflow Models

A convention for packaging ML models in a format that can be deployed to diverse serving environments (REST API, batch inference, cloud platforms).

### 4. MLflow Model Registry

A centralized store for managing model versions, stage transitions, and annotations.

---


![Machine learning operations and infrastructure](/images/blogs/pool-mlops/3.jpg)

## Setting Up MLflow

### Local Setup (Quick Start)

```bash
# Install MLflow
pip install mlflow

# Start the tracking server with a local SQLite backend
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

Open `http://localhost:5000` in your browser to see the MLflow UI.

### Production Setup (Docker Compose)

For a team setup, you want a proper database backend and remote artifact storage:

```yaml
# docker-compose.yml
version: "3.8"

services:
  mlflow:
    image: python:3.10-slim
    command: >
      bash -c "pip install mlflow boto3 psycopg2-binary &&
      mlflow server
      --backend-store-uri postgresql://mlflow:password@postgres:5432/mlflow
      --default-artifact-root s3://my-mlflow-bucket/artifacts
      --host 0.0.0.0
      --port 5000"
    ports:
      - "5000:5000"
    depends_on:
      - postgres
    environment:
      AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mlflow
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

---

## MLflow Tracking: Hands-On

### Basic Experiment Logging

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Point to your tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# Create or get an experiment
mlflow.set_experiment("credit_scoring_model")

# Generate sample data
X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Run an experiment
with mlflow.start_run(run_name="random_forest_baseline"):
    # Log parameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42,
    }
    mlflow.log_params(params)

    # Train
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_proba),
    }
    mlflow.log_metrics(metrics)

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    # Log additional artifacts
    mlflow.log_text(str(params), "config.txt")

    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Metrics: {metrics}")
```

### Logging Training Curves

For iterative models (neural networks, gradient boosting with early stopping), you can log metrics at each step:

```python
import mlflow
import xgboost as xgb

with mlflow.start_run(run_name="xgboost_with_curves"):
    mlflow.log_params({
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
    })

    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        eval_metric="logloss",
    )

    # Train with evaluation logging
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )

    # Log the training curves
    results = model.evals_result()
    for epoch, (train_loss, val_loss) in enumerate(
        zip(results["validation_0"]["logloss"],
            results["validation_1"]["logloss"])
    ):
        mlflow.log_metric("train_logloss", train_loss, step=epoch)
        mlflow.log_metric("val_logloss", val_loss, step=epoch)

    mlflow.xgboost.log_model(model, "model")
```

### Hyperparameter Sweeps

Combine MLflow with Optuna for tracked hyperparameter optimization:

```python
import optuna
import mlflow

def objective(trial):
    with mlflow.start_run(nested=True):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("f1_score", f1)

        return f1

with mlflow.start_run(run_name="hyperparameter_search"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_f1", study.best_value)
```

---


![Production ML pipeline and deployment workflow](/images/blogs/pool-mlops/4.jpg)

## MLflow Models: Packaging for Deployment

MLflow Models package your trained model in a standardized format that can be deployed anywhere:

```python
# The model is saved with a standard directory structure:
# model/
# ├── MLmodel           # Metadata about the model
# ├── conda.yaml        # Conda environment specification
# ├── requirements.txt  # pip requirements
# ├── python_env.yaml   # Python environment
# └── model.pkl         # The actual model file

# Load and serve the model
import mlflow.pyfunc

model = mlflow.pyfunc.load_model("runs:/abc123/model")
predictions = model.predict(new_data)
```

### Serving a Model as a REST API

```bash
# Serve a model from a run
mlflow models serve -m "runs:/abc123/model" -p 1234

# Or serve from the model registry
mlflow models serve -m "models:/credit_scoring_model/Production" -p 1234
```

```bash
# Test the endpoint
curl -X POST http://localhost:1234/invocations \
    -H "Content-Type: application/json" \
    -d '{"inputs": [[1.0, 2.0, 3.0, ...]]}'
```

### Custom Model Flavors

For models that do not fit standard frameworks, use the `pyfunc` interface:

```python
class CustomPreprocessingModel(mlflow.pyfunc.PythonModel):
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, context, model_input):
        processed = self.preprocessor.transform(model_input)
        return self.model.predict(processed)

# Log the custom model
mlflow.pyfunc.log_model(
    artifact_path="custom_model",
    python_model=CustomPreprocessingModel(preprocessor, trained_model),
    conda_env="conda.yaml",
)
```

---

## MLflow Model Registry: Production Management

The model registry provides a central hub for model lifecycle management:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register a model from an experiment run
result = mlflow.register_model(
    model_uri="runs:/abc123/model",
    name="credit_scoring_model"
)

# Add a description
client.update_registered_model(
    name="credit_scoring_model",
    description="Credit scoring model for loan approval decisions"
)

# Transition through stages
client.transition_model_version_stage(
    name="credit_scoring_model",
    version=result.version,
    stage="Staging"
)

# After validation...
client.transition_model_version_stage(
    name="credit_scoring_model",
    version=result.version,
    stage="Production"
)
```

---


![MLOps tooling and automation systems](/images/blogs/pool-mlops/5.jpg)

## Querying the Tracking Server Programmatically

MLflow's search API lets you find experiments and runs programmatically:

```python
# Find all runs with F1 > 0.90, sorted by AUC
runs = mlflow.search_runs(
    experiment_names=["credit_scoring_model"],
    filter_string="metrics.f1_score > 0.90",
    order_by=["metrics.auc_roc DESC"],
)

print(runs[["run_id", "params.model_type", "metrics.f1_score", "metrics.auc_roc"]])
```

```python
# Find runs from the last 7 days
from datetime import datetime, timedelta

one_week_ago = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
runs = mlflow.search_runs(
    experiment_names=["credit_scoring_model"],
    filter_string=f"attributes.start_time > {one_week_ago}",
)
```

---

## Best Practices for MLflow in Production

1. **Use a central tracking server**: Do not let each developer run their own local MLflow instance. Centralize for visibility and collaboration.
2. **Use remote artifact storage**: Store artifacts in S3/GCS/Azure Blob, not on the tracking server's filesystem.
3. **Set up authentication**: MLflow does not have built-in auth. Put it behind a reverse proxy (nginx, Traefik) with authentication.
4. **Back up the database**: The tracking database contains your entire experiment history. Back it up regularly.
5. **Tag runs consistently**: Use tags for team, project, sprint, and purpose to make searching easier.
6. **Automate cleanup**: Old experiment runs accumulate. Set up policies to archive or delete runs older than a threshold.

---

## Conclusion

MLflow is not the only experiment tracking tool, but it is the most versatile and widely adopted. Its open-source nature means no vendor lock-in, and its modular design means you can adopt it incrementally: start with tracking, add the model registry when you need it, and explore projects and serving as your practice matures.

In the next post, we will look at Weights & Biases, a tool that takes a different approach to experiment tracking with a focus on rich visualization and team collaboration.
