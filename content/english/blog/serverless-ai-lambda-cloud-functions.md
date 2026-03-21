---
title: "Serverless AI: Running Models on AWS Lambda and Cloud Functions"
date: 2027-06-03T09:00:00+05:30
draft: false
description: "Serverless computing offers a compelling alternative for ML inference: zero infrastructure management, automatic scaling, and pay-per-invocation pricing. Learn when serverless works for ML, its limitations, and how to deploy models on Lambda and Cloud Functions."
tags: ["ML Deployment", "Serverless", "AWS Lambda", "Cloud Functions", "Machine Learning", "Infrastructure"]
categories: ["ML Deployment"]
image: "https://picsum.photos/seed/serverless-ai-lambda-cloud-functions-cover/1200/630"
keywords: ["serverless ml inference", "aws lambda machine learning", "cloud functions ml model", "serverless ai deployment", "lambda model serving"]
---

What if you could deploy an ML model without managing any servers at all? No EC2 instances, no Kubernetes clusters, no Docker orchestration. Just upload your code and model, and the cloud handles everything else.

This is the promise of serverless computing for ML inference. Services like AWS Lambda, Google Cloud Functions, and Azure Functions run your code in response to events, automatically scaling from zero to thousands of concurrent executions.

---

## What is Serverless ML?

In a serverless architecture, you deploy a function (not a server) that receives an event, processes it, and returns a result. The cloud provider manages all the infrastructure:

```
Request → Cloud Gateway → Function Instance → Response
                              ↓
                         Your Code
                         + Model
                         (runs only when invoked)
```

**Key characteristics:**
- **No servers to manage**: The cloud provider handles provisioning, scaling, and patching.
- **Auto-scaling**: From zero instances to thousands, automatically.
- **Pay-per-invocation**: You pay only when your function runs, not for idle time.
- **Ephemeral**: Each function invocation is independent. No persistent state.

---

## When Serverless Works for ML

Serverless is a great fit when:

- **Traffic is sporadic**: Predictions are needed occasionally, not continuously. Paying for an always-on server is wasteful.
- **Models are small**: The model fits within the memory and package size limits.
- **Latency tolerance is moderate**: Cold starts add 1-10 seconds on the first invocation.
- **You want minimal operations**: No desire to manage Kubernetes, Docker, or servers.
- **Batch event processing**: Processing uploads, webhook events, or queue messages.

---

## When Serverless Does Not Work for ML

Serverless has limitations that make it unsuitable for some ML workloads:

- **Large models**: Lambda has a 10 GB deployment size limit, 10 GB memory limit. Large neural networks may not fit.
- **GPU inference**: Lambda and Cloud Functions do not offer GPU instances.
- **Low-latency requirements**: Cold starts can add significant latency.
- **Long-running inference**: Lambda has a 15-minute timeout. Some models take longer.
- **High throughput**: At high request volumes, the cost per invocation can exceed the cost of a dedicated server.

---

## Deploying an ML Model on AWS Lambda

### Project Structure

```
lambda-ml-project/
├── handler.py          # Lambda function code
├── model/
│   └── model.joblib    # Trained model
├── requirements.txt
└── serverless.yml      # Serverless Framework config
```

### The Lambda Handler


![Illustration of data processing pipeline and feature analysis](https://picsum.photos/seed/serverless-ai-lambda-cloud-functions-1/800/450)

```python
# handler.py
import json
import joblib
import numpy as np
import os

# Load model outside the handler for reuse across invocations
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.joblib")
model = joblib.load(MODEL_PATH)

def predict(event, context):
    """AWS Lambda handler for predictions."""
    try:
        # Parse input
        body = json.loads(event.get("body", "{}"))
        features = body.get("features")

        if features is None:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'features' field"}),
            }

        # Run inference
        feature_array = np.array([features])
        prediction = model.predict(feature_array)[0]
        probability = model.predict_proba(feature_array)[0].tolist()

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "prediction": int(prediction),
                "probabilities": probability,
            }),
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }
```

### Deploying with the Serverless Framework

```yaml
# serverless.yml
service: churn-predictor

provider:
  name: aws
  runtime: python3.10
  region: us-east-1
  memorySize: 1024
  timeout: 30

functions:
  predict:
    handler: handler.predict
    events:
      - http:
          path: predict
          method: post

plugins:
  - serverless-python-requirements

custom:
  pythonRequirements:
    dockerizePip: true
    slim: true
    strip: false
```

```bash
# Deploy
serverless deploy

# Test
curl -X POST https://abc123.execute-api.us-east-1.amazonaws.com/dev/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [35, 75000, 2, 1, 1]}'
```

### Using Lambda Container Images

For larger models, use container images instead of zip packages:

```dockerfile
# Dockerfile for Lambda
FROM public.ecr.aws/lambda/python:3.10

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model/ ${LAMBDA_TASK_ROOT}/model/
COPY handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.predict"]
```

```bash
# Build and push to ECR
docker build -t churn-predictor-lambda .
docker tag churn-predictor-lambda:latest \
    123456789.dkr.ecr.us-east-1.amazonaws.com/churn-predictor-lambda:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/churn-predictor-lambda:latest
```

Container image-based Lambda functions support up to 10 GB images, accommodating larger models.


![Diagram showing algorithm comparison and performance metrics](https://picsum.photos/seed/serverless-ai-lambda-cloud-functions-2/800/450)

---

## Deploying on Google Cloud Functions

```python
# main.py
import joblib
import numpy as np
import json
import functions_framework

# Load model globally for reuse
model = joblib.load("model/model.joblib")

@functions_framework.http
def predict(request):
    """Google Cloud Function for predictions."""
    if request.method != "POST":
        return json.dumps({"error": "Only POST allowed"}), 405

    request_json = request.get_json(silent=True)
    if not request_json or "features" not in request_json:
        return json.dumps({"error": "Missing 'features' field"}), 400

    features = np.array([request_json["features"]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].tolist()

    return json.dumps({
        "prediction": int(prediction),
        "probabilities": probability,
    })
```

```bash
# Deploy
gcloud functions deploy predict \
    --runtime python310 \
    --trigger-http \
    --allow-unauthenticated \
    --memory 1024MB \
    --timeout 60s \
    --region us-central1
```

---

## Dealing with Cold Starts

The biggest challenge with serverless ML is **cold starts**: the delay when a new function instance is created.

Cold start timeline:
```
Request → Provision Instance → Load Runtime → Load Dependencies → Load Model → Process Request → Response
          ←─── Cold Start (1-10 seconds) ──→   ←── Warm (ms) ──→
```

### Mitigation Strategies

**1. Provisioned Concurrency (Lambda)**

Keep instances warm:

```yaml
functions:
  predict:
    handler: handler.predict
    provisionedConcurrency: 5  # Keep 5 instances warm
```

**2. Minimize Package Size**

Smaller packages load faster:

```bash
# Use Lambda layers for large dependencies
aws lambda publish-layer-version \
    --layer-name sklearn-layer \
    --compatible-runtimes python3.10 \
    --zip-file fileb://sklearn-layer.zip
```

**3. Load Model Lazily**


![Visual representation of machine learning model architecture and data flow](https://picsum.photos/seed/serverless-ai-lambda-cloud-functions-3/800/450)

```python
model = None

def get_model():
    global model
    if model is None:
        model = joblib.load("model/model.joblib")
    return model

def predict(event, context):
    m = get_model()
    # ...
```

**4. Use Lighter Model Formats**

```python
# Convert to ONNX for faster loading
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
# ONNX models typically load 2-5x faster than pickle
```

---

## Cost Comparison: Serverless vs. Dedicated

| Scenario | Serverless (Lambda) | Dedicated (t3.medium) |
|----------|-------------------|---------------------|
| 1,000 requests/day | ~$0.03/month | ~$30/month |
| 10,000 requests/day | ~$0.30/month | ~$30/month |
| 100,000 requests/day | ~$3/month | ~$30/month |
| 1,000,000 requests/day | ~$30/month | ~$30/month |
| 10,000,000 requests/day | ~$300/month | ~$60/month (larger instance) |

The crossover point is typically around 1-5 million requests per day. Below that, serverless is cheaper. Above that, dedicated compute is more cost-effective.

---

## Serverless ML Architecture Patterns

### Pattern 1: API Gateway + Lambda

```
Client → API Gateway → Lambda (predict) → Response
```

Simple and direct. Good for synchronous, low-volume predictions.

### Pattern 2: Async Processing with SQS

```
Client → API Gateway → SQS Queue → Lambda (predict) → S3 (results)
                           ↓
                     DynamoDB (status)
```

Good for non-urgent predictions where the client can poll for results.

### Pattern 3: Event-Driven Prediction

```
S3 Upload (new image) → Lambda (classify) → DynamoDB (result)
                                          → SNS (notification)
```

Trigger predictions automatically when new data arrives.

---

## Best Practices

1. **Load the model outside the handler** so it persists across warm invocations.
2. **Use provisioned concurrency** for latency-sensitive endpoints.
3. **Monitor cold start rates** and adjust concurrency settings.
4. **Keep dependencies minimal**: only include libraries you actually use.
5. **Set appropriate timeouts and memory**: more memory also means more CPU.
6. **Use container images** for models larger than the 50 MB zip limit.
7. **Test locally** before deploying with tools like SAM Local or Functions Framework.

---

## Conclusion

Serverless ML is not a replacement for Kubernetes or dedicated infrastructure. It is an alternative for specific use cases: sporadic traffic, small models, and teams that want zero infrastructure management.

The sweet spot for serverless ML is clear: low-to-moderate traffic, small-to-medium models, and use cases where cold start latency is acceptable. For these scenarios, serverless eliminates infrastructure complexity and reduces costs significantly.

In the next post, we will put theory into practice with a complete code tutorial: building a minimal FastAPI model serving endpoint from scratch, step by step.
