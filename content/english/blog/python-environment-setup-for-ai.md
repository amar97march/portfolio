---
title: "How I Set Up My Python Environment for AI Projects"
date: 2026-05-03T10:00:00+05:30
draft: false
description: "A practical guide to setting up a clean, reproducible Python environment for machine learning — covering conda, virtual environments, GPU configuration, and essential tools."
tags: ["Python", "Environment Setup", "DevOps", "ML Tools", "Best Practices"]
categories: ["AI Tools & Ecosystem"]
image: "/images/blogs/pool-tools/1.jpg"
keywords: ["python environment setup", "ml environment", "conda setup", "python virtual environment", "GPU setup python"]
---

One of the most underrated skills in machine learning is environment management. I have seen talented data scientists lose hours to dependency conflicts, broken CUDA installations, and the dreaded "it works on my machine" problem. A clean, reproducible environment is not glamorous, but it is the foundation that everything else rests on.

Here is exactly how I set up my Python environment for AI projects, from scratch.

## The Golden Rule: Isolation

Never install ML packages into your system Python. Never. Every project gets its own isolated environment. This prevents dependency conflicts (project A needs TensorFlow 2.12, project B needs 2.15) and makes your work reproducible.

You have two main options for isolation: **conda** and **venv + pip**. I use both, depending on the situation.

## Option 1: Conda (My Recommendation for ML)

Conda is a package manager and environment manager that handles both Python packages and non-Python dependencies (like CUDA, cuDNN, and system libraries). This is a significant advantage for ML, where GPU-related dependencies are a constant pain point.

### Installing Conda

I recommend **Miniconda** (minimal installation) over Anaconda (bloated with packages you may not need):

```bash
# Download Miniconda (Linux)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# macOS
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

### Creating Project Environments

```bash
# Create an environment with a specific Python version
conda create -n my-ml-project python=3.11

# Activate it
conda activate my-ml-project

# Install packages
conda install numpy pandas scikit-learn matplotlib jupyter

# Install PyTorch with CUDA support (one command handles everything!)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Or TensorFlow
pip install tensorflow  # (TensorFlow is typically installed via pip, even in conda)
```

The killer feature: `conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia` installs PyTorch AND the correct CUDA toolkit AND cuDNN, all matched for compatibility. With pip alone, you would need to install CUDA and cuDNN separately and pray the versions are compatible.

### Environment Files

For reproducibility, export your environment to a file:

```bash
# Export
conda env export > environment.yml

# Recreate on another machine
conda env create -f environment.yml
```

A typical `environment.yml`:

```yaml
name: my-ml-project
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy=1.26
  - pandas=2.1
  - scikit-learn=1.3
  - matplotlib=3.8
  - jupyter=1.0
  - pytorch=2.1
  - torchvision=0.16
  - pytorch-cuda=12.1
  - pip:
    - transformers==4.35
    - datasets==2.15
    - wandb==0.16
```

## Option 2: venv + pip


![Diagram showing software ecosystem for machine learning projects](/images/blogs/pool-tools/8.jpg)

For simpler projects or when you want a lighter-weight solution:

```bash
# Create a virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install packages
pip install numpy pandas scikit-learn matplotlib jupyter

# Freeze dependencies
pip freeze > requirements.txt

# Recreate
pip install -r requirements.txt
```

### My requirements.txt Template

```text
# Core scientific stack
numpy>=1.24
pandas>=2.0
scipy>=1.11
matplotlib>=3.7
seaborn>=0.12

# Machine learning
scikit-learn>=1.3
xgboost>=2.0

# Deep learning (choose one)
torch>=2.1
torchvision>=0.16
# tensorflow>=2.14

# NLP
transformers>=4.35
tokenizers>=0.15
datasets>=2.15

# Experiment tracking
wandb>=0.16
tensorboard>=2.14

# Utilities
tqdm>=4.66
python-dotenv>=1.0
jupyter>=1.0
ipywidgets>=8.1
black>=23.0
ruff>=0.1
```

## GPU Setup

Getting GPU acceleration working is often the trickiest part. Here is my approach.

### Verify Your GPU

```bash
# Check if NVIDIA GPU is detected
nvidia-smi
```

### PyTorch GPU Verification

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

### TensorFlow GPU Verification


![Illustration of developer workflow and toolchain integration](/images/blogs/pool-tools/7.jpg)

```python
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")

# Limit GPU memory growth (prevents TF from grabbing all GPU memory)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

## Project Structure

I follow a consistent project structure for all my ML projects:

```
my-ml-project/
├── data/
│   ├── raw/              # Original, immutable data
│   ├── processed/        # Cleaned, transformed data
│   └── external/         # Third-party data
├── notebooks/
│   ├── 01-exploration.ipynb
│   ├── 02-preprocessing.ipynb
│   └── 03-modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data/             # Data loading and processing
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models/           # Model definitions
│   │   ├── __init__.py
│   │   └── classifier.py
│   ├── training/         # Training logic
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils/            # Helper functions
│       ├── __init__.py
│       └── metrics.py
├── configs/
│   └── config.yaml
├── tests/
│   └── test_model.py
├── outputs/
│   ├── models/           # Saved model weights
│   ├── figures/          # Generated plots
│   └── logs/             # Training logs
├── .env                  # Environment variables (API keys, etc.)
├── .gitignore
├── environment.yml       # or requirements.txt
├── pyproject.toml
└── Makefile             # Common commands
```

### The .gitignore

Essential for ML projects — model weights and data files should not be in git:

```gitignore
# Data
data/raw/
data/processed/
*.csv
*.parquet
*.h5

# Model weights
*.pth
*.pt
*.keras
*.onnx
outputs/models/

# Environment
.venv/
*.egg-info/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# OS
.DS_Store
__pycache__/

# Secrets
.env
```

## Essential Tools Beyond the Libraries


![Visual overview of AI development tools and frameworks](/images/blogs/pool-tools/6.jpg)

### Experiment Tracking: Weights & Biases

Tracking experiments manually (in spreadsheets or text files) does not scale. I use Weights & Biases (wandb) to log hyperparameters, metrics, and artifacts:

```python
import wandb

wandb.init(project="my-classifier", config={
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "architecture": "ResNet50"
})

for epoch in range(50):
    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)

    wandb.log({
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "epoch": epoch
    })

wandb.finish()
```

### Code Quality: Black + Ruff

```bash
# Format code
black src/ notebooks/

# Lint
ruff check src/
```

### Makefile for Common Commands

```makefile
.PHONY: setup train evaluate clean

setup:
	conda env create -f environment.yml

train:
	python src/training/trainer.py --config configs/config.yaml

evaluate:
	python src/training/evaluate.py --model outputs/models/best.pth

clean:
	rm -rf outputs/models/* outputs/logs/*
```

## Tips From Experience

1. **Pin your versions.** "It worked last week" is the most frustrating debugging experience. Pin everything in your environment file.

2. **Use a seed everywhere.** Set random seeds for reproducibility:
```python
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

3. **Keep raw data immutable.** Never modify original data files. Write processing scripts that read from `data/raw/` and write to `data/processed/`.

4. **Version your models.** Save model weights with meaningful names that include the date, hyperparameters, or performance metrics.

5. **Use environment variables for secrets.** API keys, database credentials, and cloud storage paths go in `.env`, never in code.

A well-configured environment saves you countless hours of debugging and makes your work reproducible, shareable, and professional.

---

*We are now transitioning from tools to data. Next up: Why data quality matters more than algorithm choice.*
