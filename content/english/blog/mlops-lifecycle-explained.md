---
title: "The MLOps Lifecycle: From Data to Deployment to Monitoring"
date: 2027-04-13T09:00:00+05:30
draft: false
description: "The MLOps lifecycle spans far beyond model training. Walk through every stage from data collection and validation to deployment, monitoring, and retraining, and understand how they connect into a continuous loop."
tags: ["MLOps", "Machine Learning", "ML Pipeline", "Data Engineering", "Model Deployment", "ML Monitoring"]
categories: ["MLOps"]
image: "https://images.unsplash.com/photo-1522199755839-a2bacb67c546?w=1200&h=630&fit=crop&auto=format"
keywords: ["mlops lifecycle", "ml pipeline stages", "machine learning workflow", "mlops stages explained", "end to end ml pipeline"]
---

Most introductions to machine learning focus on the training loop: load data, build model, train, evaluate, repeat. This represents maybe 10-20% of the work in a production ML system.

The remaining 80-90% is everything else: getting clean data, validating it, deploying the model, monitoring it, and retraining when it degrades. This is the MLOps lifecycle, and understanding it end-to-end is critical for anyone building ML systems that need to work in the real world.

---

## The Full MLOps Lifecycle

Here is the complete lifecycle, visualized as a continuous loop:

```
┌─────────────────────────────────────────────────┐
│                                                   │
│   Data Collection                                 │
│        ↓                                          │
│   Data Validation & Quality Checks                │
│        ↓                                          │
│   Data Preprocessing & Feature Engineering        │
│        ↓                                          │
│   Model Training & Experimentation                │
│        ↓                                          │
│   Model Evaluation & Validation                   │
│        ↓                                          │
│   Model Registry & Versioning                     │
│        ↓                                          │
│   Model Deployment (Staging → Production)         │
│        ↓                                          │
│   Model Serving & Inference                       │
│        ↓                                          │
│   Monitoring & Alerting                           │
│        ↓                                          │
│   Trigger Retraining ─────────────────────┐       │
│        ↑                                  │       │
│        └──────────────────────────────────┘       │
└─────────────────────────────────────────────────┘
```

Let us walk through each stage in detail.

---

## Stage 1: Data Collection

Everything begins with data. This stage involves:

- **Identifying data sources**: Databases, APIs, event streams, third-party vendors, web scraping, sensor data.
- **Ingestion pipelines**: Automated systems that pull data on a schedule or in real-time.
- **Raw storage**: Landing the data in a data lake or warehouse in its original format.

The key principle here is **traceability**. For every piece of data, you should be able to answer: Where did it come from? When was it collected? What system produced it?

```python
# Example: A simple data ingestion pipeline using Apache Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def ingest_user_events():
    """Pull user events from the event stream and store in data lake."""
    events = event_stream.consume(topic="user_actions", since=last_run)
    data_lake.write(
        data=events,
        path=f"raw/user_events/{datetime.now().strftime('%Y/%m/%d')}",
        format="parquet"
    )

dag = DAG("user_events_ingestion", schedule_interval="@hourly")
ingest_task = PythonOperator(
    task_id="ingest_user_events",
    python_callable=ingest_user_events,
    dag=dag,
)
```

---

## Stage 2: Data Validation and Quality Checks

Raw data is unreliable. Before it enters your ML pipeline, you need to validate it. This stage catches problems like:

- **Schema violations**: A column that should be numeric contains strings.
- **Missing values**: A critical feature has 40% nulls today when it normally has 2%.
- **Distribution shifts**: The average order value suddenly doubled (is that real or a data bug?).
- **Duplicate records**: The same transaction appears three times.

Tools like **Great Expectations** and **TensorFlow Data Validation (TFDV)** automate these checks:

```python
import great_expectations as gx

# Define expectations for your dataset
validator = context.get_validator(datasource="user_events")
validator.expect_column_values_to_not_be_null("user_id")
validator.expect_column_values_to_be_between("age", min_value=0, max_value=150)
validator.expect_column_mean_to_be_between("purchase_amount", min_value=10, max_value=500)

# Run validation
results = validator.validate()
if not results.success:
    alert_team("Data validation failed", results)
    halt_pipeline()
```

If validation fails, the pipeline should stop and alert the team. Never let bad data flow silently into your training pipeline.


![Machine learning operations and infrastructure](https://picsum.photos/seed/mlops-lifecycle-explained-1/800/450)

---

## Stage 3: Data Preprocessing and Feature Engineering

Once validated, data needs to be transformed into features that the model can consume. This stage includes:

- **Cleaning**: Handling missing values, removing outliers, fixing inconsistencies.
- **Transformation**: Scaling, encoding categorical variables, normalizing text.
- **Feature engineering**: Creating new features from raw data (e.g., "days since last purchase" from a timestamp).
- **Feature selection**: Removing features that add noise rather than signal.

A critical MLOps principle: **the preprocessing pipeline must be identical between training and inference**. If you scale features using the training set mean and standard deviation, the serving pipeline must use those exact same values, not recompute them from production data.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Define the preprocessing pipeline
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), ["age", "income", "purchase_count"]),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), ["city", "device_type"]),
])

# Fit on training data
preprocessor.fit(train_data)

# Save for use in serving
joblib.dump(preprocessor, "artifacts/preprocessor.joblib")
```

---

## Stage 4: Model Training and Experimentation

This is the stage most data scientists are familiar with. It involves:

- **Experiment design**: Deciding which architectures, hyperparameters, and approaches to try.
- **Training runs**: Executing experiments, often in parallel across multiple configurations.
- **Experiment tracking**: Logging every parameter, metric, and artifact for reproducibility.

The difference between experimentation in a notebook and experimentation in an MLOps pipeline is **rigor and traceability**. Every experiment should be logged with:

- The exact code version (git commit)
- The exact data version (DVC hash or data snapshot ID)
- All hyperparameters
- All evaluation metrics
- The trained model artifact

```python
import mlflow

mlflow.set_experiment("churn_prediction_v2")

with mlflow.start_run():
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("data_version", "v2.3.1")

    model = train_xgboost(train_data, params)
    metrics = evaluate(model, test_data)

    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("f1_score", metrics["f1"])
    mlflow.log_metric("auc_roc", metrics["auc_roc"])

    mlflow.sklearn.log_model(model, "model")
```

---

## Stage 5: Model Evaluation and Validation

Before a model can be promoted to production, it must pass a series of validation gates:

- **Performance benchmarks**: Does it meet minimum thresholds for accuracy, latency, and other metrics?
- **Comparison to baseline**: Does it outperform the current production model?
- **Fairness checks**: Does it perform consistently across different demographic groups?
- **Stress testing**: Does it handle edge cases, adversarial inputs, and unexpected data formats?

This is often implemented as an automated validation step in the pipeline that produces a go/no-go decision.


![Production ML pipeline and deployment workflow](https://picsum.photos/seed/mlops-lifecycle-explained-2/800/450)

---

## Stage 6: Model Registry and Versioning

Validated models are stored in a **model registry**, which serves as the single source of truth for all production-ready models. The registry tracks:

- Model versions and their lineage
- Which version is currently deployed to staging and production
- Performance metrics for each version
- Model metadata (training data version, hyperparameters, etc.)

Think of it as a package registry (like npm or PyPI) but for ML models.

---

## Stage 7: Model Deployment

Deployment involves getting the model from the registry into a serving environment. There are several patterns:

- **Blue-green deployment**: Run two identical environments, switch traffic between them.
- **Canary deployment**: Route a small percentage of traffic to the new model, monitor, then gradually increase.
- **Shadow deployment**: Run the new model alongside the old one, compare predictions without serving the new model's results to users.

The choice depends on your risk tolerance and the cost of bad predictions.

---

## Stage 8: Model Serving and Inference

Once deployed, the model needs to handle incoming prediction requests. This involves:

- **API endpoints**: REST or gRPC services that accept feature vectors and return predictions.
- **Batch processing**: Scheduled jobs that generate predictions for large datasets.
- **Edge deployment**: Running models on devices (phones, IoT sensors) for low-latency inference.

Each pattern has different requirements for latency, throughput, and infrastructure.


![MLOps tooling and automation systems](https://picsum.photos/seed/mlops-lifecycle-explained-3/800/450)

---

## Stage 9: Monitoring and Alerting

The model is live. Now you need to watch it constantly. Monitoring covers:

- **Model performance**: Are predictions still accurate? (Requires ground truth labels, which may arrive with a delay.)
- **Data drift**: Has the distribution of input features changed since training?
- **Concept drift**: Has the relationship between features and outcomes changed?
- **System health**: Latency, throughput, error rates, resource utilization.

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Compare production data to training data
report = Report(metrics=[DataDriftPreset()])
report.run(
    reference_data=training_data,
    current_data=production_data_last_24h,
)

drift_detected = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
if drift_detected:
    alert_team("Data drift detected in production features")
```

---

## Stage 10: Trigger Retraining

When monitoring detects degradation, the system should trigger a retraining cycle. This can be:

- **Scheduled**: Retrain every week regardless of performance.
- **Triggered**: Retrain when performance drops below a threshold or drift is detected.
- **Continuous**: Retrain incrementally as new labeled data becomes available.

And with retraining, the cycle begins again: new data flows in, gets validated, a new model is trained, evaluated, registered, deployed, and monitored.

---

## The Lifecycle is a Loop, Not a Line

The most important insight is that the MLOps lifecycle is **continuous**. There is no "done" state. A deployed model is not a finished product; it is a living system that requires ongoing care.

This loop is what separates mature ML organizations from those that are still stuck in the "train once, deploy once, pray it works" paradigm.

In the next posts, we will dive deeper into specific stages, starting with data versioning using DVC, a tool that brings git-like version control to your datasets.
