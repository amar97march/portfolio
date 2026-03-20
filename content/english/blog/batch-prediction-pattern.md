---
title: "Deployment Pattern: Batch Prediction for Offline Inference"
date: 2027-05-19T09:00:00+05:30
draft: false
description: "Batch prediction is the simplest and most cost-effective way to deploy ML models. Learn when to use batch inference, how to design batch prediction pipelines, and common patterns for scheduling and storing predictions."
tags: ["ML Deployment", "Batch Prediction", "Machine Learning", "MLOps", "Data Engineering", "Inference"]
categories: ["ML Deployment"]
image: "/images/blogs/default-ai-cover.png"
keywords: ["batch prediction ml", "batch inference pattern", "offline prediction", "ml deployment patterns", "batch scoring pipeline"]
---

Not every ML model needs to serve predictions in real time. In fact, for many use cases, generating predictions in advance and storing them for later consumption is simpler, cheaper, and more reliable than a real-time inference service.

This is the **batch prediction** pattern, and it is the workhorse of production ML deployments.

---

## What is Batch Prediction?

Batch prediction (also called batch inference or batch scoring) is a deployment pattern where predictions are generated for a large set of inputs at once, typically on a schedule, and stored for later use.

```
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐
│  Input   │───→│  Model   │───→│ Predictions  │───→│  Storage │
│  Dataset │    │  (Batch) │    │   Dataset    │    │  (DB/S3) │
└──────────┘    └──────────┘    └──────────────┘    └──────────┘
     ▲                                                    │
     │                                                    │
  Scheduled                                          Application
  (daily/hourly)                                     reads from
                                                     storage
```

The key distinction from real-time inference: predictions are computed before they are needed, not at the moment they are requested.

---

## When to Use Batch Prediction

Batch prediction is the right choice when:

### 1. Predictions Do Not Need to Be Instant

- **Recommendation emails**: "Here are products you might like" emails can be generated overnight.
- **Risk scores**: Credit risk scores can be computed daily for all customers.
- **Content moderation**: Scanning uploaded content can happen minutes after upload.
- **Forecasting**: Sales forecasts, demand predictions, and inventory planning are inherently batch operations.

### 2. The Input Set is Known in Advance

If you know all the entities you need to predict for (all customers, all products, all locations), batch prediction is efficient. You process them all at once rather than one at a time.

### 3. Latency Requirements are Relaxed

If users do not need predictions within milliseconds, batch processing is simpler. Acceptable latency might be minutes, hours, or even a day.

### 4. Cost is a Priority

Batch jobs can run on cheaper compute (spot instances, preemptible VMs) and scale down to zero between runs. Real-time endpoints need to be always on.

![Batch prediction architecture computing and storing predictions on a schedule](/images/blogs/pool-deploy/3.jpg)

---

## Designing a Batch Prediction Pipeline

### Step 1: Data Collection

Gather the features needed for prediction:

```python
import pandas as pd
from sqlalchemy import create_engine

def collect_features():
    """Collect features for all active customers."""
    engine = create_engine("postgresql://...")

    query = """
    SELECT
        customer_id,
        age,
        total_purchases_30d,
        avg_order_value_30d,
        days_since_last_purchase,
        support_tickets_30d,
        login_frequency_7d
    FROM customer_features
    WHERE is_active = true
    """

    return pd.read_sql(query, engine)
```

### Step 2: Load the Model

```python
import joblib
import mlflow

def load_production_model():
    """Load the current production model from the registry."""
    model = mlflow.pyfunc.load_model("models:/churn_predictor/Production")
    return model
```

### Step 3: Generate Predictions

```python
def generate_predictions(model, features_df):
    """Generate predictions for all customers."""
    predictions = model.predict(features_df.drop("customer_id", axis=1))
    probabilities = model.predict_proba(
        features_df.drop("customer_id", axis=1)
    )[:, 1]

    results = pd.DataFrame({
        "customer_id": features_df["customer_id"],
        "churn_prediction": predictions,
        "churn_probability": probabilities,
        "predicted_at": pd.Timestamp.now(),
        "model_version": get_model_version(),
    })

    return results
```

### Step 4: Store Predictions

```python
def store_predictions(predictions_df):
    """Store predictions for downstream consumption."""
    engine = create_engine("postgresql://...")

    predictions_df.to_sql(
        "churn_predictions",
        engine,
        if_exists="replace",  # or "append" with date partitioning
        index=False,
    )

    # Also store to S3 for archival
    predictions_df.to_parquet(
        f"s3://predictions/churn/{datetime.now().strftime('%Y/%m/%d')}/predictions.parquet"
    )
```

### Step 5: Validate Predictions

```python
def validate_predictions(predictions_df, features_df):
    """Sanity check the predictions before storing."""
    # Check completeness
    assert len(predictions_df) == len(features_df), \
        f"Expected {len(features_df)} predictions, got {len(predictions_df)}"

    # Check value ranges
    assert predictions_df["churn_probability"].between(0, 1).all(), \
        "Probabilities outside [0, 1] range"

    # Check for distribution anomalies
    churn_rate = predictions_df["churn_prediction"].mean()
    if churn_rate > 0.5 or churn_rate < 0.01:
        alert_team(f"Unusual churn rate: {churn_rate:.2%}")

    # Check for null predictions
    null_count = predictions_df["churn_probability"].isna().sum()
    if null_count > 0:
        alert_team(f"{null_count} null predictions detected")
```

---

## The Complete Pipeline

Putting it all together with error handling:

```python
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def run_batch_prediction_pipeline():
    """Execute the complete batch prediction pipeline."""
    start_time = datetime.now()
    logger.info("Starting batch prediction pipeline")

    try:
        # Step 1: Collect features
        logger.info("Collecting features...")
        features_df = collect_features()
        logger.info(f"Collected features for {len(features_df)} customers")

        # Step 2: Load model
        logger.info("Loading production model...")
        model = load_production_model()

        # Step 3: Generate predictions
        logger.info("Generating predictions...")
        predictions_df = generate_predictions(model, features_df)

        # Step 4: Validate
        logger.info("Validating predictions...")
        validate_predictions(predictions_df, features_df)

        # Step 5: Store
        logger.info("Storing predictions...")
        store_predictions(predictions_df)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Pipeline completed in {duration:.1f}s. "
            f"Generated {len(predictions_df)} predictions."
        )

        # Log metrics
        log_pipeline_metrics({
            "num_predictions": len(predictions_df),
            "duration_seconds": duration,
            "avg_churn_probability": predictions_df["churn_probability"].mean(),
            "churn_rate": predictions_df["churn_prediction"].mean(),
        })

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        alert_team(f"Batch prediction pipeline failed: {e}")
        raise
```

![Complete batch prediction pipeline with validation and error handling](/images/blogs/pool-deploy/5.jpg)

---

## Scheduling Batch Predictions

### Using Apache Airflow

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["ml-team@company.com"],
}

dag = DAG(
    "churn_batch_predictions",
    default_args=default_args,
    schedule_interval="0 6 * * *",  # Daily at 6 AM
    catchup=False,
)

collect_task = PythonOperator(
    task_id="collect_features",
    python_callable=collect_features,
    dag=dag,
)

predict_task = PythonOperator(
    task_id="generate_predictions",
    python_callable=run_batch_prediction_pipeline,
    dag=dag,
)

collect_task >> predict_task
```

### Using Cron (Simple Alternative)

```bash
# crontab entry: run daily at 6 AM
0 6 * * * cd /app && python -m batch_prediction.pipeline >> /var/log/batch_predict.log 2>&1
```

---

## Consuming Batch Predictions

Downstream applications read from the prediction store:

```python
# Application code: look up precomputed predictions
def get_churn_risk(customer_id):
    """Get precomputed churn risk for a customer."""
    result = db.execute(
        "SELECT churn_probability, predicted_at FROM churn_predictions WHERE customer_id = %s",
        (customer_id,)
    ).fetchone()

    if result is None:
        return {"risk": "unknown", "note": "No prediction available"}

    return {
        "customer_id": customer_id,
        "churn_probability": result["churn_probability"],
        "predicted_at": result["predicted_at"],
        "risk_level": categorize_risk(result["churn_probability"]),
    }
```

![Downstream applications consuming precomputed batch predictions](/images/blogs/pool-deploy/7.jpg)

---

## Scaling Batch Predictions

For large datasets, you need to scale horizontally:

### Using Spark

```python
from pyspark.sql import SparkSession
import mlflow

spark = SparkSession.builder.appName("batch_predictions").getOrCreate()

# Load data as a Spark DataFrame
features_df = spark.read.parquet("s3://data/customer_features/")

# Load model as a Spark UDF
model_udf = mlflow.pyfunc.spark_udf(spark, "models:/churn_predictor/Production")

# Generate predictions in parallel across the cluster
predictions_df = features_df.withColumn(
    "churn_probability",
    model_udf(*feature_columns)
)

predictions_df.write.parquet("s3://predictions/churn/latest/")
```

### Using Dask

```python
import dask.dataframe as dd

features_ddf = dd.read_parquet("s3://data/customer_features/")

# Apply predictions in parallel
predictions = features_ddf.map_partitions(
    lambda partition: predict_partition(model, partition)
)

predictions.to_parquet("s3://predictions/churn/latest/")
```

---

## Batch Prediction Best Practices

1. **Always validate predictions** before writing them to the production store.
2. **Version your predictions**: Include model version and timestamp with every prediction.
3. **Keep historical predictions**: Do not overwrite. Store daily snapshots for analysis and debugging.
4. **Monitor for freshness**: Alert if the batch job fails and predictions become stale.
5. **Idempotent pipelines**: Running the pipeline twice should produce the same result.
6. **Use spot/preemptible instances**: Batch jobs are perfect for cheap, interruptible compute.

---

## Conclusion

Batch prediction is underappreciated. It is simpler, cheaper, and more reliable than real-time inference for a large class of ML use cases. Before reaching for a real-time serving solution, ask yourself: do these predictions really need to be computed on demand?

If the answer is no, batch prediction is your friend.

In the next post, we will explore the other side of the coin: real-time inference, for use cases where predictions must be generated the moment they are requested.
