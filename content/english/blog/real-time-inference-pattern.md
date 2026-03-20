---
title: "Deployment Pattern: Real-Time Inference for Online Systems"
date: 2027-05-22T09:00:00+05:30
draft: false
description: "Real-time inference serves predictions on demand with low latency. Learn when you need real-time serving, how to architect inference services, and the key challenges of latency, throughput, and reliability."
tags: ["ML Deployment", "Real-Time Inference", "Machine Learning", "API", "Model Serving", "Latency"]
categories: ["ML Deployment"]
image: "/images/blogs/default-ai-cover.png"
keywords: ["real time inference ml", "online prediction serving", "ml model api", "low latency inference", "real time ml deployment"]
---

Some predictions cannot wait. When a user searches for a product, the recommendation model needs to respond in milliseconds. When a credit card transaction occurs, the fraud detection model must decide before the payment completes. When a user speaks to a voice assistant, the speech recognition model processes audio in real time.

This is real-time inference: generating predictions on demand, as fast as the request arrives.

---

## What is Real-Time Inference?

Real-time inference (also called online prediction or synchronous inference) is a deployment pattern where a model serves predictions through an API endpoint, responding to individual requests with low latency.

```
Client Request     Model Service       Response
    │                    │                 │
    │  POST /predict     │                 │
    │  {"features": ...} │                 │
    │───────────────────→│                 │
    │                    │  Load features  │
    │                    │  Run inference  │
    │                    │  Format output  │
    │                    │←────────────────│
    │  {"prediction": x} │                 │
    │←───────────────────│                 │
    │                    │                 │
    │   < 100ms total    │                 │
```

---

## When to Use Real-Time Inference

Real-time inference is necessary when:

- **The input is not known in advance**: Search queries, user interactions, and sensor readings cannot be predicted ahead of time.
- **Freshness matters**: The prediction must reflect the most current state (e.g., real-time fraud detection).
- **Latency is critical**: The application cannot tolerate the delay of looking up precomputed predictions.
- **The prediction space is too large**: If there are billions of possible input combinations, precomputing all predictions is infeasible.

Common use cases:
- Search ranking and recommendation
- Fraud detection
- Dynamic pricing
- Content personalization
- Chatbots and language models
- Image classification in mobile apps
- Autonomous vehicle perception

---

## Anatomy of a Real-Time Inference Service

A production inference service has several layers:

```
┌─────────────────────────────────────────────┐
│              Load Balancer                   │
├─────────────────────────────────────────────┤
│           API Gateway / Router               │
├─────────────────────────────────────────────┤
│          Request Preprocessing               │
│   (validation, feature lookup, transform)    │
├─────────────────────────────────────────────┤
│            Model Inference                   │
│        (the actual prediction)               │
├─────────────────────────────────────────────┤
│          Response Postprocessing             │
│    (formatting, filtering, enrichment)       │
├─────────────────────────────────────────────┤
│          Logging & Monitoring                │
│   (latency, predictions, feature values)     │
└─────────────────────────────────────────────┘
```

### Request Preprocessing

Often the most time-consuming step. It includes:

- **Input validation**: Check that the request has the expected fields and types.
- **Feature lookup**: Fetch precomputed features from a feature store or database.
- **Feature computation**: Calculate real-time features from the request data.
- **Transformation**: Apply the same scaling, encoding, and normalization used during training.


![Illustration of scalable AI serving patterns](/images/blogs/pool-deploy/5.jpg)

```python
async def preprocess_request(request_data):
    """Transform raw request into model-ready features."""
    # Validate input
    validate_schema(request_data)

    # Fetch precomputed features from feature store
    user_features = await feature_store.get_features(
        entity_id=request_data["user_id"],
        feature_names=["purchase_history_30d", "avg_session_duration"]
    )

    # Compute real-time features
    real_time_features = {
        "hour_of_day": datetime.now().hour,
        "day_of_week": datetime.now().weekday(),
        "request_device": request_data.get("device_type", "unknown"),
    }

    # Combine and transform
    features = {**user_features, **real_time_features}
    feature_vector = preprocessor.transform(features)

    return feature_vector
```

### Model Inference

The actual prediction, typically the fastest step:

```python
def predict(feature_vector):
    """Run model inference."""
    prediction = model.predict(feature_vector)
    confidence = model.predict_proba(feature_vector)
    return prediction, confidence
```

### Response Postprocessing

Format the output for the client:

```python
def postprocess(prediction, confidence, metadata):
    """Format the response."""
    return {
        "prediction": int(prediction),
        "confidence": float(confidence),
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now().isoformat(),
    }
```

---

## Latency Optimization

In real-time inference, every millisecond counts. Here are the key optimization strategies:

### 1. Model Optimization

```python
# Quantize a PyTorch model for faster inference
import torch

model = torch.load("model.pt")
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8,
)

# Convert to ONNX for cross-platform optimization
torch.onnx.export(model, sample_input, "model.onnx")

# Use ONNX Runtime for optimized inference
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
result = session.run(None, {"input": feature_array})
```

### 2. Caching


![Visual representation of MLOps pipeline and monitoring](/images/blogs/pool-deploy/4.jpg)

```python
from functools import lru_cache
import hashlib

# Cache predictions for identical inputs
@lru_cache(maxsize=10000)
def cached_predict(feature_hash):
    features = decode_features(feature_hash)
    return model.predict(features)

def predict_with_cache(features):
    feature_hash = hashlib.md5(features.tobytes()).hexdigest()
    return cached_predict(feature_hash)
```

### 3. Batching

Group individual requests into mini-batches for more efficient GPU utilization:

```python
import asyncio
from collections import deque

class BatchPredictor:
    def __init__(self, model, max_batch_size=32, max_wait_ms=10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = deque()

    async def predict(self, features):
        future = asyncio.Future()
        self.queue.append((features, future))

        if len(self.queue) >= self.max_batch_size:
            await self._process_batch()
        else:
            await asyncio.sleep(self.max_wait_ms / 1000)
            if not future.done():
                await self._process_batch()

        return await future

    async def _process_batch(self):
        batch = []
        futures = []
        while self.queue and len(batch) < self.max_batch_size:
            features, future = self.queue.popleft()
            batch.append(features)
            futures.append(future)

        if batch:
            results = self.model.predict(np.array(batch))
            for future, result in zip(futures, results):
                future.set_result(result)
```

### 4. Hardware Selection

| Model Type | Recommended Hardware |
|-----------|---------------------|
| Small tabular models (XGBoost, sklearn) | CPU (cost-effective) |
| Medium neural networks | GPU (T4 for inference) |
| Large language models | GPU (A100, H100) |
| Edge/mobile deployment | NPU, Apple Neural Engine |

---

## Reliability Patterns

### Health Checks

```python
@app.get("/health")
async def health_check():
    """Verify the service is healthy."""
    try:
        # Check model is loaded
        assert model is not None

        # Run a smoke test prediction
        test_result = model.predict(SMOKE_TEST_INPUT)
        assert test_result is not None

        return {"status": "healthy", "model_version": MODEL_VERSION}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )
```

### Graceful Degradation


![Diagram of model deployment architecture and infrastructure](/images/blogs/pool-deploy/3.jpg)

```python
@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        result = await run_inference(request)
        return result
    except TimeoutError:
        # Fall back to a simpler model
        return await run_fallback_model(request)
    except FeatureStoreUnavailable:
        # Use default features
        return await run_with_defaults(request)
    except Exception:
        # Return a safe default
        return {"prediction": DEFAULT_PREDICTION, "fallback": True}
```

### Circuit Breaker

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
async def call_feature_store(user_id):
    """Fetch features with circuit breaker protection."""
    return await feature_store.get_features(user_id)
```

---

## Monitoring Real-Time Inference

Critical metrics to track:

```python
import time
from prometheus_client import Histogram, Counter, Gauge

LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
REQUESTS = Counter("prediction_requests_total", "Total predictions")
ERRORS = Counter("prediction_errors_total", "Failed predictions")
MODEL_VERSION = Gauge("model_version", "Current model version")

@app.post("/predict")
async def predict(request: PredictRequest):
    REQUESTS.inc()
    start = time.time()
    try:
        result = await run_inference(request)
        LATENCY.observe(time.time() - start)
        return result
    except Exception as e:
        ERRORS.inc()
        raise
```

---

## Real-Time vs. Batch: The Decision Framework

| Factor | Batch | Real-Time |
|--------|-------|-----------|
| Latency requirement | Hours/minutes OK | Milliseconds needed |
| Input known in advance? | Yes | No |
| Cost sensitivity | Lower cost | Higher cost (always-on) |
| Complexity | Lower | Higher |
| Freshness requirement | Stale OK | Must be current |
| Scale pattern | Predictable | Variable/bursty |

Many production systems use both: batch predictions for the common cases, with real-time inference for new or uncommon inputs.

---

## Conclusion

Real-time inference is more complex than batch prediction, but it is essential for interactive applications. The key to success is understanding that inference is just one step in a pipeline that includes preprocessing, feature lookup, prediction, postprocessing, and monitoring.

Optimize the full pipeline, not just the model. Often the biggest latency wins come from caching features, reducing network hops, or simplifying preprocessing, not from making the model faster.

In the next post, we will build a complete real-time inference service using FastAPI, walking through the code from request handling to model serving.
