---
title: "Scaling Your AI Application with Kubernetes"
date: 2027-05-31T09:00:00+05:30
draft: false
description: "Kubernetes is the industry standard for orchestrating containerized applications at scale. Learn how to deploy ML models on Kubernetes with auto-scaling, rolling updates, health checks, and GPU scheduling."
tags: ["ML Deployment", "Kubernetes", "Docker", "Scaling", "Machine Learning", "Infrastructure"]
categories: ["ML Deployment"]
image: "https://picsum.photos/seed/scaling-ai-with-kubernetes-cover/1200/630"
keywords: ["kubernetes ml deployment", "scale ai kubernetes", "k8s model serving", "kubernetes machine learning", "ml auto scaling"]
---

Your containerized model serving application works great on a single machine. But what happens when traffic increases? When you need high availability? When you want to update the model without downtime? When you need GPU scheduling?

This is where Kubernetes comes in. It orchestrates your containers across a cluster of machines, handling scaling, scheduling, networking, and failure recovery automatically.

---

## Why Kubernetes for ML?

Kubernetes provides capabilities that are essential for production ML:

- **Auto-scaling**: Automatically add or remove instances based on traffic.
- **Rolling updates**: Deploy new model versions without downtime.
- **Health management**: Automatically restart failed containers.
- **Resource scheduling**: Place GPU workloads on GPU nodes, CPU workloads on CPU nodes.
- **Service discovery**: Other services find your model API automatically.
- **Configuration management**: Manage environment variables, secrets, and config files.

---

## Kubernetes Basics for ML Engineers

If you are new to Kubernetes, here are the key concepts:

```
Cluster
├── Node (a machine)
│   ├── Pod (smallest deployable unit, contains 1+ containers)
│   │   └── Container (your Docker image)
│   ├── Pod
│   └── Pod
├── Node
│   ├── Pod
│   └── Pod
└── Node
    └── Pod
```

- **Pod**: The smallest unit. Usually one container, but can be multiple.
- **Deployment**: Manages a set of identical Pods. Handles scaling and updates.
- **Service**: Exposes Pods to network traffic. Provides stable endpoints.
- **HorizontalPodAutoscaler (HPA)**: Automatically scales Pods based on metrics.
- **ConfigMap/Secret**: External configuration and sensitive data.

---

## Deploying a Model Serving Application

### The Deployment Manifest

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-predictor
  labels:
    app: churn-predictor
    version: v2.3.1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-predictor
  template:
    metadata:
      labels:
        app: churn-predictor
        version: v2.3.1
    spec:
      containers:
        - name: model-server
          image: registry.example.com/churn-predictor:v2.3.1
          ports:
            - containerPort: 8000
          env:
            - name: MODEL_VERSION
              value: "v2.3.1"
            - name: LOG_LEVEL
              value: "INFO"
            - name: WORKERS
              value: "2"
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
            failureThreshold: 3
          startupProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
            failureThreshold: 12  # Allow 60s for model loading
```

### The Service


![Illustration of scalable AI serving patterns](https://picsum.photos/seed/scaling-ai-with-kubernetes-1/800/450)

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: churn-predictor-service
spec:
  selector:
    app: churn-predictor
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP
```

### Apply the Manifests

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check the status
kubectl get pods -l app=churn-predictor
kubectl get service churn-predictor-service
```

---

## Auto-Scaling

### Horizontal Pod Autoscaler

Scale based on CPU utilization or custom metrics:

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: churn-predictor-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: churn-predictor
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 4
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Pods
          value: 1
          periodSeconds: 120
```

The `behavior` section prevents thrashing: scale up quickly (4 pods per minute) but scale down slowly (1 pod every 2 minutes).

---

## Rolling Updates and Canary Deployments

### Rolling Update Strategy

```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1         # Create 1 extra pod during update
      maxUnavailable: 0   # Never reduce below desired count
```

Deploy a new version:


![Visual representation of MLOps pipeline and monitoring](https://picsum.photos/seed/scaling-ai-with-kubernetes-2/800/450)

```bash
# Update the image
kubectl set image deployment/churn-predictor \
    model-server=registry.example.com/churn-predictor:v2.4.0

# Watch the rollout
kubectl rollout status deployment/churn-predictor

# If something goes wrong, rollback
kubectl rollout undo deployment/churn-predictor
```

### Canary Deployment Pattern

Run two deployments simultaneously with traffic splitting:

```yaml
# canary-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-predictor-canary
spec:
  replicas: 1  # Small number of canary pods
  selector:
    matchLabels:
      app: churn-predictor
      track: canary
  template:
    metadata:
      labels:
        app: churn-predictor
        track: canary
        version: v2.4.0
    spec:
      containers:
        - name: model-server
          image: registry.example.com/churn-predictor:v2.4.0
          # ... same configuration as main deployment
```

With Istio or a similar service mesh, you can control exact traffic percentages:

```yaml
# istio-virtual-service.yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: churn-predictor
spec:
  hosts:
    - churn-predictor-service
  http:
    - route:
        - destination:
            host: churn-predictor-service
            subset: stable
          weight: 95
        - destination:
            host: churn-predictor-service
            subset: canary
          weight: 5
```

---

## GPU Scheduling

For models that need GPU inference:

```yaml
spec:
  containers:
    - name: model-server
      image: registry.example.com/llm-server:v1
      resources:
        limits:
          nvidia.com/gpu: 1  # Request 1 GPU
  nodeSelector:
    cloud.google.com/gke-accelerator: nvidia-tesla-t4
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
```

Kubernetes schedules this Pod only on nodes with available GPUs.

---

## Configuration Management

### ConfigMaps for Non-Sensitive Config

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
data:
  MODEL_VERSION: "v2.3.1"
  LOG_LEVEL: "INFO"
  MAX_BATCH_SIZE: "100"
  FEATURE_STORE_URL: "http://feature-store:8080"
```

### Secrets for Sensitive Data


![Diagram of model deployment architecture and infrastructure](https://picsum.photos/seed/scaling-ai-with-kubernetes-3/800/450)

```bash
kubectl create secret generic model-secrets \
    --from-literal=aws-access-key=AKIA... \
    --from-literal=aws-secret-key=...
```

Reference in the deployment:

```yaml
spec:
  containers:
    - name: model-server
      envFrom:
        - configMapRef:
            name: model-config
        - secretRef:
            name: model-secrets
```

---

## Monitoring on Kubernetes

### Prometheus Service Monitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: churn-predictor-monitor
spec:
  selector:
    matchLabels:
      app: churn-predictor
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
```

### Resource Monitoring

```bash
# Check resource usage
kubectl top pods -l app=churn-predictor

# View logs
kubectl logs -l app=churn-predictor --tail=100 -f

# Describe a pod for debugging
kubectl describe pod churn-predictor-abc123
```

---

## Common Patterns for ML on Kubernetes

### Pattern 1: Sidecar for Feature Lookup

```yaml
spec:
  containers:
    - name: model-server
      image: churn-predictor:v2
    - name: feature-cache
      image: redis:7
      ports:
        - containerPort: 6379
```

### Pattern 2: Init Container for Model Download

```yaml
spec:
  initContainers:
    - name: download-model
      image: amazon/aws-cli:latest
      command: ["aws", "s3", "cp", "s3://models/latest/", "/models/", "--recursive"]
      volumeMounts:
        - name: model-volume
          mountPath: /models
  containers:
    - name: model-server
      image: churn-predictor:v2
      volumeMounts:
        - name: model-volume
          mountPath: /app/artifacts
  volumes:
    - name: model-volume
      emptyDir: {}
```

---

## Conclusion

Kubernetes is the standard platform for running ML inference services at scale. It handles the undifferentiated heavy lifting of scaling, health management, deployment orchestration, and resource scheduling, so you can focus on building better models.

The key Kubernetes concepts for ML deployment: Deployments for managing replicas, Services for networking, HPAs for auto-scaling, and proper health checks for reliability.

In the next post, we will explore serverless alternatives: running ML models on AWS Lambda and Cloud Functions, for use cases where the simplicity of serverless outweighs the flexibility of Kubernetes.
