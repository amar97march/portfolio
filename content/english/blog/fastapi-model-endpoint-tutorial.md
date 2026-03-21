---
title: "Code Tutorial: A Minimal FastAPI Model Serving Endpoint"
date: 2027-06-06T09:00:00+05:30
draft: false
description: "A complete, hands-on tutorial for building a minimal but production-ready FastAPI model serving endpoint. From training a model to serving it as an API, containerizing it, and testing it end to end."
tags: ["ML Deployment", "FastAPI", "Python", "Tutorial", "Docker", "Model Serving"]
categories: ["ML Deployment"]
image: "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=1200&h=630&fit=crop&auto=format"
keywords: ["fastapi model serving tutorial", "ml api tutorial", "python model endpoint", "deploy ml model api", "fastapi ml tutorial step by step"]
---

Theory is important, but nothing beats building something with your own hands. In this tutorial, we will build a complete model serving pipeline from scratch: train a model, save it, wrap it in a FastAPI endpoint, write tests, containerize it with Docker, and verify everything works end to end.

By the end, you will have a working, deployable ML API.

---

## What We Are Building

A churn prediction API that:
1. Accepts customer features as JSON
2. Returns a churn prediction with probability
3. Includes health checks, input validation, and error handling
4. Runs in a Docker container

---

## Step 1: Set Up the Project

```bash
mkdir churn-api && cd churn-api

# Create project structure
mkdir -p src artifacts tests

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn scikit-learn joblib numpy pydantic pytest httpx
```

Create `requirements.txt`:

```txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
scikit-learn==1.4.0
joblib==1.3.2
numpy==1.26.3
pydantic==2.5.3
```

---

## Step 2: Train and Save a Model

```python
# src/train.py
"""Train a simple churn prediction model and save artifacts."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

def train_and_save():
    # Generate synthetic churn data
    X, y = make_classification(
        n_samples=5000,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        random_state=42,
        weights=[0.7, 0.3],
    )

    # Feature names for documentation
    feature_names = [
        "age", "income", "num_products",
        "has_credit_card", "is_active_member"
    ]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocess
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print("Model Performance:")
    print(classification_report(y_test, y_pred, target_names=["retained", "churned"]))

    # Save artifacts
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    joblib.dump(model, os.path.join(artifacts_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.joblib"))

    print(f"Artifacts saved to {artifacts_dir}")

if __name__ == "__main__":
    train_and_save()
```

Run the training:

```bash
python src/train.py
```

---

![Building a model serving API with FastAPI](https://picsum.photos/seed/fastapi-model-endpoint-tutorial-1/800/450)

## Step 3: Build the FastAPI Application

```python
# src/app.py
"""FastAPI application for churn prediction."""
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_VERSION = "1.0.0"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

# Global state
model = None
scaler = None
start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, scaler, start_time

    logger.info("Loading model artifacts...")
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    start_time = time.time()
    logger.info(f"Model v{MODEL_VERSION} loaded successfully")

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Churn Prediction API",
    description="Predict whether a customer will churn based on their profile.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# --- Schemas ---

class CustomerInput(BaseModel):
    """Input features for a single customer."""
    age: float = Field(..., ge=0, le=150, description="Customer age")
    income: float = Field(..., ge=0, description="Annual income")
    num_products: int = Field(..., ge=0, le=20, description="Number of products")
    has_credit_card: bool = Field(..., description="Has a credit card")
    is_active_member: bool = Field(..., description="Is an active member")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "age": 35.0,
                "income": 75000.0,
                "num_products": 2,
                "has_credit_card": True,
                "is_active_member": True,
            }]
        }
    }


class PredictionOutput(BaseModel):
    """Output of a prediction."""
    customer_data: CustomerInput
    prediction: str = Field(..., description="'churned' or 'retained'")
    churn_probability: float = Field(..., ge=0, le=1)
    model_version: str
    timestamp: str


class HealthOutput(BaseModel):
    """Health check response."""
    status: str
    model_version: str
    uptime_seconds: float


# --- Endpoints ---

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Churn Prediction API", "docs": "/docs"}


@app.get("/health", response_model=HealthOutput)
async def health():
    """Check if the service is healthy."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthOutput(
        status="healthy",
        model_version=MODEL_VERSION,
        uptime_seconds=round(time.time() - start_time, 1),
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(customer: CustomerInput):
    """Predict churn for a single customer."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert to feature array
    features = np.array([[
        customer.age,
        customer.income,
        customer.num_products,
        float(customer.has_credit_card),
        float(customer.is_active_member),
    ]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    return PredictionOutput(
        customer_data=customer,
        prediction="churned" if prediction == 1 else "retained",
        churn_probability=round(float(probability), 4),
        model_version=MODEL_VERSION,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict/batch", response_model=list[PredictionOutput])
async def predict_batch(customers: list[CustomerInput]):
    """Predict churn for multiple customers."""
    if len(customers) > 100:
        raise HTTPException(status_code=400, detail="Max batch size is 100")

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert all customers to feature matrix
    features = np.array([
        [c.age, c.income, c.num_products,
         float(c.has_credit_card), float(c.is_active_member)]
        for c in customers
    ])

    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)[:, 1]

    results = []
    for customer, pred, prob in zip(customers, predictions, probabilities):
        results.append(PredictionOutput(
            customer_data=customer,
            prediction="churned" if pred == 1 else "retained",
            churn_probability=round(float(prob), 4),
            model_version=MODEL_VERSION,
            timestamp=datetime.now().isoformat(),
        ))

    return results
```

---

## Step 4: Test the Application

```python
# tests/test_app.py
"""Tests for the churn prediction API."""
import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Churn Prediction API" in response.json()["message"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data


def test_predict_valid():
    response = client.post("/predict", json={
        "age": 35,
        "income": 75000,
        "num_products": 2,
        "has_credit_card": True,
        "is_active_member": True,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["churned", "retained"]
    assert 0 <= data["churn_probability"] <= 1
    assert "model_version" in data
    assert "timestamp" in data


def test_predict_invalid_age():
    response = client.post("/predict", json={
        "age": -5,
        "income": 75000,
        "num_products": 2,
        "has_credit_card": True,
        "is_active_member": True,
    })
    assert response.status_code == 422  # Validation error


def test_predict_missing_field():
    response = client.post("/predict", json={
        "age": 35,
        "income": 75000,
        # Missing num_products and others
    })
    assert response.status_code == 422


def test_predict_batch():
    customers = [
        {"age": 25, "income": 40000, "num_products": 1,
         "has_credit_card": True, "is_active_member": True},
        {"age": 55, "income": 120000, "num_products": 4,
         "has_credit_card": False, "is_active_member": False},
    ]
    response = client.post("/predict/batch", json=customers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_batch_too_large():
    customers = [
        {"age": 30, "income": 50000, "num_products": 1,
         "has_credit_card": True, "is_active_member": True}
    ] * 101
    response = client.post("/predict/batch", json=customers)
    assert response.status_code == 400
```

Run the tests:

```bash
pytest tests/ -v
```

---

![Testing and validating ML prediction endpoints](https://picsum.photos/seed/fastapi-model-endpoint-tutorial-2/800/450)

## Step 5: Containerize with Docker

```dockerfile
# Dockerfile
FROM python:3.10-slim AS runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and artifacts
COPY src/ src/
COPY artifacts/ artifacts/

# Create non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
# Build
docker build -t churn-api:v1 .

# Run
docker run -p 8000:8000 churn-api:v1

# Test
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"age": 35, "income": 75000, "num_products": 2, "has_credit_card": true, "is_active_member": true}'
```

---

## Step 6: Try It Out

Open your browser and navigate to `http://localhost:8000/docs`. You will see the Swagger UI with interactive documentation for all endpoints.

Try the `/predict` endpoint:

```json
{
    "age": 42,
    "income": 95000,
    "num_products": 3,
    "has_credit_card": true,
    "is_active_member": false
}
```

Expected response:

```json
{
    "customer_data": {
        "age": 42.0,
        "income": 95000.0,
        "num_products": 3,
        "has_credit_card": true,
        "is_active_member": false
    },
    "prediction": "churned",
    "churn_probability": 0.7234,
    "model_version": "1.0.0",
    "timestamp": "2027-06-06T10:30:00.123456"
}
```

---

## The Complete File Structure

```
churn-api/
├── Dockerfile
├── requirements.txt
├── artifacts/
│   ├── model.joblib
│   └── scaler.joblib
├── src/
│   ├── __init__.py
│   ├── app.py
│   └── train.py
└── tests/
    ├── __init__.py
    └── test_app.py
```

---

![Containerized ML application ready for deployment](https://picsum.photos/seed/fastapi-model-endpoint-tutorial-3/800/450)

## What We Built

In under 200 lines of application code, we have:

1. A trained classification model saved as artifacts.
2. A FastAPI application with input validation, error handling, and health checks.
3. Single and batch prediction endpoints.
4. Automated tests with full coverage of happy paths and edge cases.
5. A Docker container ready for deployment anywhere.
6. Interactive API documentation generated automatically.

This is a minimal but complete foundation. From here, you can add monitoring (Prometheus metrics), logging (structured JSON logs), authentication (API keys or JWT), and CI/CD pipelines.

The entire project can be deployed to any platform that supports Docker: Kubernetes, AWS ECS, Google Cloud Run, Azure Container Apps, or even a simple VM.

In the next series of posts, we will shift focus to what happens after deployment: monitoring ML models in production, detecting data drift, concept drift, and model decay, and building automated retraining pipelines.
