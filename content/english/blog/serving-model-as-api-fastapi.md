---
title: "How to Serve a Model as an API with FastAPI"
date: 2027-05-25T09:00:00+05:30
draft: false
description: "FastAPI is the go-to framework for building ML model serving APIs in Python. Learn how to build a production-ready model serving endpoint with input validation, error handling, health checks, and async support."
tags: ["ML Deployment", "FastAPI", "Python", "API", "Model Serving", "REST API"]
categories: ["ML Deployment"]
image: "/images/blogs/pool-deploy/1.jpg"
keywords: ["fastapi model serving", "serve ml model api", "python model api", "fastapi ml deployment", "model serving endpoint"]
---

The most common pattern for serving ML models in production is wrapping them in a REST API. A client sends features, the API runs inference, and returns the prediction. Simple in concept, but building a production-ready serving endpoint requires attention to several details: input validation, error handling, model loading, health checks, and monitoring.

FastAPI is the Python framework that makes this easiest. It is fast, modern, and built on type hints and Pydantic, which means your API gets automatic validation, documentation, and serialization with minimal boilerplate.

---

## Why FastAPI for Model Serving?

- **Performance**: Built on Starlette and Uvicorn, FastAPI is one of the fastest Python web frameworks.
- **Automatic validation**: Pydantic models validate inputs automatically, catching bad data before it reaches your model.
- **Automatic docs**: Swagger UI and ReDoc are generated automatically from your type annotations.
- **Async support**: Handle concurrent requests efficiently with async/await.
- **Type safety**: Type hints make the code self-documenting and catch errors early.

---

![Setting up a FastAPI project for model serving](/images/blogs/pool-deploy/3.jpg)

## A Minimal Model Serving API

Let us start with the simplest possible example and build up:

```python
# app.py
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="ML Model API", version="1.0.0")

# Load model at startup
model = joblib.load("model.joblib")

@app.post("/predict")
def predict(features: list[float]):
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}
```

```bash
# Run it
uvicorn app:app --host 0.0.0.0 --port 8000
```

```bash
# Test it
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [25, 50000, 3, 1, 0]}'
```

This works, but it is not production-ready. Let us fix that.

---

## Production-Ready Model Serving

### Step 1: Define Request and Response Schemas

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    """Input schema for prediction requests."""
    age: int = Field(..., ge=0, le=150, description="Customer age")
    income: float = Field(..., ge=0, description="Annual income")
    num_products: int = Field(..., ge=0, description="Number of products")
    has_credit_card: bool = Field(..., description="Has credit card")
    is_active_member: bool = Field(..., description="Is active member")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 35,
                    "income": 75000.0,
                    "num_products": 2,
                    "has_credit_card": True,
                    "is_active_member": True,
                }
            ]
        }
    }

class PredictionResponse(BaseModel):
    """Output schema for prediction responses."""
    prediction: int = Field(..., description="Predicted class (0 or 1)")
    probability: float = Field(..., description="Prediction probability")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
```

### Step 2: Implement the Application

```python
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Global state
model = None
preprocessor = None
MODEL_VERSION = "v2.3.1"
START_TIME = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, preprocessor, START_TIME

    logger.info("Loading model and preprocessor...")
    try:
        model = joblib.load("artifacts/model.joblib")
        preprocessor = joblib.load("artifacts/preprocessor.joblib")
        START_TIME = time.time()
        logger.info(f"Model {MODEL_VERSION} loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield  # Application runs here

    logger.info("Shutting down, cleaning up resources...")

app = FastAPI(
    title="Churn Prediction API",
    description="Real-time churn prediction service",
    version=MODEL_VERSION,
    lifespan=lifespan,
)
```

### Step 3: Implement Endpoints

```python
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service is healthy and the model is loaded."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=MODEL_VERSION,
        uptime_seconds=time.time() - START_TIME,
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate a churn prediction for a customer."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert request to feature array
        features = np.array([[
            request.age,
            request.income,
            request.num_products,
            int(request.has_credit_card),
            int(request.is_active_member),
        ]])

        # Preprocess
        features_scaled = preprocessor.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            model_version=MODEL_VERSION,
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(requests: list[PredictionRequest]):
    """Generate predictions for multiple customers at once."""
    if len(requests) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum of 100"
        )

    features = np.array([
        [r.age, r.income, r.num_products, int(r.has_credit_card), int(r.is_active_member)]
        for r in requests
    ])

    features_scaled = preprocessor.transform(features)
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)[:, 1]

    return [
        PredictionResponse(
            prediction=int(pred),
            probability=float(prob),
            model_version=MODEL_VERSION,
            timestamp=datetime.now(),
        )
        for pred, prob in zip(predictions, probabilities)
    ]
```

### Step 4: Add Middleware for Logging and Metrics

```python
import time
from fastapi import Request

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with timing information."""
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"duration={duration:.3f}s"
    )

    response.headers["X-Process-Time"] = str(duration)
    response.headers["X-Model-Version"] = MODEL_VERSION

    return response
```

---

![Production-ready model API with validation and error handling](/images/blogs/pool-deploy/4.jpg)

## Running in Production

### With Uvicorn Workers

```bash
# Multiple workers for CPU-bound models
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn as process manager
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Configuration with Environment Variables

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str = "artifacts/model.joblib"
    preprocessor_path: str = "artifacts/preprocessor.joblib"
    model_version: str = "v1.0.0"
    max_batch_size: int = 100
    log_level: str = "INFO"
    workers: int = 4

    class Config:
        env_prefix = "ML_"

settings = Settings()
```

---

## Testing the API

```python
# test_app.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction():
    response = client.post("/predict", json={
        "age": 35,
        "income": 75000.0,
        "num_products": 2,
        "has_credit_card": True,
        "is_active_member": True,
    })
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1

def test_invalid_input():
    response = client.post("/predict", json={
        "age": -5,  # Invalid: negative age
        "income": 75000.0,
        "num_products": 2,
        "has_credit_card": True,
        "is_active_member": True,
    })
    assert response.status_code == 422  # Validation error

def test_batch_prediction():
    customers = [
        {"age": 25, "income": 40000, "num_products": 1,
         "has_credit_card": True, "is_active_member": True},
        {"age": 45, "income": 90000, "num_products": 3,
         "has_credit_card": False, "is_active_member": False},
    ]
    response = client.post("/predict/batch", json=customers)
    assert response.status_code == 200
    assert len(response.json()) == 2
```

---

![Testing and deploying the model API endpoint](/images/blogs/pool-deploy/5.jpg)

## Automatic API Documentation

FastAPI generates interactive documentation automatically. Navigate to:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These pages let you test the API interactively, inspect request/response schemas, and share the API specification with consumers.

---

## Conclusion

FastAPI provides the perfect balance of simplicity and power for ML model serving. With Pydantic validation, async support, automatic documentation, and excellent performance, it handles the boilerplate so you can focus on the ML logic.

The key principles for production-ready model serving:
- Validate all inputs with Pydantic schemas.
- Handle errors gracefully with proper HTTP status codes.
- Include health checks for load balancer integration.
- Log requests and prediction metadata for debugging and monitoring.
- Version your API and model.
- Test everything.

In the next post, we will take this FastAPI application and containerize it with Docker, making it deployable on any infrastructure.
