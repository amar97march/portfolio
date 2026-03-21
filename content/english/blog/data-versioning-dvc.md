---
title: "Data Versioning with DVC: Track Your Datasets Like Code"
date: 2027-04-16T09:00:00+05:30
draft: false
description: "DVC (Data Version Control) brings git-like versioning to datasets and ML artifacts. Learn why data versioning matters, how DVC works under the hood, and how to set it up for your ML projects."
tags: ["MLOps", "DVC", "Data Versioning", "Machine Learning", "Git", "Data Engineering"]
categories: ["MLOps"]
image: "https://images.unsplash.com/photo-1526948128573-703ee1aeb6fa?w=1200&h=630&fit=crop&auto=format"
keywords: ["dvc data version control", "data versioning ml", "track datasets git", "dvc tutorial", "ml data management"]
---

Git changed how we manage code. Before git, developers emailed zip files, maintained folders named `project_final_v2_FINAL_actually_final`, and prayed that no one edited the same file at the same time.

Now imagine that same chaos, but with datasets that are gigabytes or terabytes in size. That is the state of data management in most ML projects today.

**DVC (Data Version Control)** is the tool that brings git-like discipline to your data.

---

## Why Data Versioning Matters

Consider this scenario: Your team trained a model six months ago that performed exceptionally well. Now you want to retrain it. But which exact dataset did you use? Was it before or after you cleaned those duplicate records? Did it include the December holiday data? Which version of the feature engineering script processed it?

Without data versioning, these questions are unanswerable. And without answers, your results are irreproducible.

Data versioning gives you:

- **Reproducibility**: Check out any historical version of your data and recreate the exact experiment.
- **Traceability**: Know exactly which data produced which model.
- **Collaboration**: Team members can share and switch between dataset versions without confusion.
- **Rollback**: If new data turns out to be corrupted, roll back to the last known good version.

---

## Why Git Alone Does Not Work

The obvious question: why not just use git to version data?

Git was designed for text files (source code). It stores the complete content of every version of every file. This works fine for code, but for large datasets it creates several problems:

1. **Size**: A 10 GB dataset versioned 20 times would require 200 GB in your git repository.
2. **Speed**: Git operations (clone, pull, push) become unbearably slow with large binary files.
3. **Hosting**: GitHub and GitLab have file size limits (typically 100 MB).
4. **Diff**: Git cannot meaningfully diff binary files or large CSV files.

Git LFS (Large File Storage) partially addresses this, but it has limitations around branching, cost, and integration with ML workflows.

DVC was designed from the ground up to handle this problem.

---

## How DVC Works

DVC works alongside git, not as a replacement. The core idea is elegant:

1. **Your data files** are stored in a remote storage backend (S3, GCS, Azure Blob, NFS, or even a local directory).
2. **Small metadata files** (`.dvc` files) are stored in git. These files contain a hash of the data, pointing to the exact version in remote storage.
3. **Git tracks the metadata**, DVC tracks the actual data.

```
Your Git Repository              Remote Storage (S3, GCS, etc.)
├── data/                         ├── ab/
│   └── training.csv.dvc  ───────→│   └── cd1234...  (actual data)
├── models/                       ├── ef/
│   └── model.pkl.dvc    ────────→│   └── 5678ab...  (actual model)
├── src/
│   └── train.py
└── .gitignore
```

When you checkout a different git branch or a historical commit, the `.dvc` files change, and `dvc checkout` restores the corresponding data files from remote storage.

---

![Version control workflow for datasets and models](https://picsum.photos/seed/data-versioning-dvc-1/800/450)

## Setting Up DVC: A Practical Walkthrough

### Installation

```bash
# Install DVC with your preferred remote storage support
pip install dvc[s3]     # For AWS S3
pip install dvc[gs]     # For Google Cloud Storage
pip install dvc[azure]  # For Azure Blob Storage
pip install dvc         # For local storage only
```

### Initialize DVC in Your Project

```bash
# Navigate to your git repository
cd my-ml-project

# Initialize DVC
dvc init

# This creates:
# .dvc/          - DVC configuration directory
# .dvcignore     - Similar to .gitignore but for DVC
```

### Configure Remote Storage

```bash
# Set up an S3 bucket as your remote storage
dvc remote add -d myremote s3://my-ml-bucket/dvc-store

# Or use Google Cloud Storage
dvc remote add -d myremote gs://my-ml-bucket/dvc-store

# Or even a local directory (for testing)
dvc remote add -d myremote /tmp/dvc-store
```

### Start Tracking Data

```bash
# Add a dataset to DVC tracking
dvc add data/training.csv

# This creates:
# data/training.csv.dvc   - Metadata file (tracked by git)
# data/.gitignore          - Tells git to ignore the actual data file

# Commit the metadata to git
git add data/training.csv.dvc data/.gitignore
git commit -m "Track training dataset with DVC"

# Push data to remote storage
dvc push
```

### What is Inside a .dvc File?

```yaml
# data/training.csv.dvc
outs:
- md5: ab1234cd5678ef9012345678abcdef01
  size: 1073741824
  hash: md5
  path: training.csv
```

It is just a hash and some metadata. This tiny file is what git tracks. The actual 1 GB file lives in remote storage, addressable by its hash.

---

## Working with Versions

### Switching Between Data Versions

```bash
# You are on the main branch with the latest data
# Switch to a branch with older data
git checkout experiment-v1

# Restore the data files for this branch
dvc checkout

# Now data/training.csv contains the exact data from experiment-v1
```

### Comparing Data Versions

```bash
# See which data files changed between branches
dvc diff main experiment-v1
```

### Tagging Important Versions

```bash
# Tag a specific data version
git tag -a "dataset-v2.0" -m "Cleaned dataset with December data included"
git push origin --tags

# Later, restore this exact dataset
git checkout dataset-v2.0
dvc checkout
```

---

![Tracking data versions across experiment branches](https://picsum.photos/seed/data-versioning-dvc-2/800/450)

## DVC Pipelines: Versioning the Entire Workflow

DVC does not just version data files. It can also version and automate your entire ML pipeline.

Define your pipeline in a `dvc.yaml` file:

```yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - src/preprocess.py
      - data/raw/
    outs:
      - data/processed/

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/
    params:
      - train.learning_rate
      - train.max_depth
    outs:
      - models/model.pkl
    metrics:
      - metrics/scores.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/test/
    metrics:
      - metrics/eval.json:
          cache: false
```

Now you can reproduce the entire pipeline:

```bash
# Run the full pipeline
dvc repro

# DVC automatically determines which stages need to rerun
# based on changes to dependencies
```

If you change the preprocessing script, DVC knows to rerun preprocessing and everything downstream. If you only change the evaluation script, DVC only reruns evaluation.

---

## DVC Metrics and Experiments

DVC also provides experiment tracking capabilities:

```bash
# Compare metrics across git branches or tags
dvc metrics diff main experiment-v1

# Output:
# Path              Metric    Old      New      Change
# metrics/eval.json accuracy  0.912    0.934    0.022
# metrics/eval.json f1_score  0.889    0.911    0.022
```

And with DVC experiments, you can run multiple parameter variations:

```bash
# Run an experiment with modified parameters
dvc exp run --set-param train.learning_rate=0.01

# List all experiments
dvc exp show

# Apply the best experiment to your workspace
dvc exp apply exp-abc123
```

---

![MLOps pipeline with automated data management](https://picsum.photos/seed/data-versioning-dvc-3/800/450)

## Best Practices for Data Versioning with DVC

### 1. Version Early, Version Often

Do not wait until your project is "mature" to start versioning data. Initialize DVC at the beginning of every project.

### 2. Use Meaningful Git Commit Messages

Since DVC metadata lives in git, your commit messages should describe data changes:

```bash
git commit -m "Add Q4 2026 user interaction data, 2.3M new records"
```

### 3. Separate Raw and Processed Data

```
data/
├── raw/          # Original, unmodified data (DVC tracked)
├── processed/    # Transformed data (DVC pipeline output)
└── external/     # Third-party data (DVC tracked)
```

### 4. Always Push After Committing

```bash
git push        # Push metadata to git remote
dvc push        # Push data to DVC remote storage
```

If you only push git, your teammates will have the `.dvc` files but not the actual data.

### 5. Document Data Changes

Maintain a `DATA_CHANGELOG.md` that describes what changed in each data version, similar to a software changelog.

---

## When to Use DVC vs. Other Tools

| Scenario | Tool |
|----------|------|
| Versioning datasets alongside code | DVC |
| Data lake with branching capabilities | LakeFS |
| Versioning database tables | Delta Lake / Apache Iceberg |
| Versioning only model artifacts | MLflow Model Registry |
| Simple file tracking without pipelines | Git LFS |

DVC is best suited for projects where datasets are stored as files (CSV, Parquet, images, etc.) and where tight integration with git workflows is important.

---

## Conclusion

Data versioning is not optional in a mature MLOps practice. Without it, you cannot reproduce experiments, you cannot audit models, and you cannot safely roll back when things go wrong.

DVC provides an elegant solution by extending git's philosophy to large files and datasets. It is free, open-source, and integrates with every major cloud storage provider.

If you are still managing datasets by copying files into folders named `data_v3_final_FINAL`, it is time to make the switch.

In the next post, we will look at the other side of the versioning coin: model versioning and model registries, the systems that track your trained models from experiment to production.
