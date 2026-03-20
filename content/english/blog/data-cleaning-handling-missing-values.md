---
title: "Data Preprocessing: What is Cleaning Data? Handling Missing Values"
date: 2026-05-09T10:00:00+05:30
draft: false
description: "A hands-on guide to data cleaning and handling missing values — the essential first step in any machine learning project, with practical Python examples."
tags: ["Data Cleaning", "Missing Values", "Preprocessing", "Pandas", "Data Science"]
categories: ["Data Science"]
image: "/images/blogs/pool-data/1.jpg"
keywords: ["data cleaning", "handling missing values", "data preprocessing", "imputation methods", "pandas data cleaning"]
---

Data cleaning is the least glamorous and most important part of any machine learning project. You will never see a headline about a data scientist spending three days fixing inconsistent date formats, but that work is often the difference between a model that works and one that does not.

In this post, I am going to walk through the practical reality of data cleaning, with a special focus on the most common challenge: **handling missing values**.

## What is Data Cleaning?

Data cleaning (also called data cleansing or data scrubbing) is the process of detecting and correcting errors, inconsistencies, and inaccuracies in a dataset. It includes:

- Handling missing values
- Removing duplicates
- Fixing inconsistent formatting
- Correcting data types
- Identifying and dealing with outliers
- Resolving conflicting records

Let me walk through each of these with real code.

## The Reality of Raw Data

Here is what a real-world dataset might look like before cleaning:

```python
import pandas as pd
import numpy as np

# Simulating a messy real-world dataset
data = {
    'name': ['Alice Smith', 'bob jones', 'CHARLIE BROWN', 'Diana Prince', None,
             'Alice Smith', 'Frank Castle', 'Grace Hopper'],
    'age': [28, 34, -5, 45, 31, 28, None, 200],
    'salary': [75000, None, 55000, 120000, 85000, 75000, 68000, 95000],
    'department': ['Engineering', 'marketing', 'ENGINEERING', 'Management',
                   'Marketing', 'Engineering', 'Engg', 'Engineering'],
    'hire_date': ['2020-01-15', '01/15/2019', '2021-06-01', 'March 2018',
                  '2022-09-01', '2020-01-15', '2023-03-15', '2019-11-01'],
    'email': ['alice@company.com', 'bob@company', 'charlie@company.com',
              'diana@company.com', None, 'alice@company.com',
              'frank@company.com', 'grace@company.com']
}

df = pd.DataFrame(data)
print(df)
```

This dataset has at least seven problems. Let me fix them one by one.

## 1. Handling Missing Values

Missing values are represented as `NaN` (Not a Number) in Pandas. They are ubiquitous in real data and require careful handling.

### Detecting Missing Values

```python
# Count missing values per column
print(df.isnull().sum())
# name          1
# age           1
# salary        1
# department    0
# hire_date     0
# email         1

# Percentage missing
print((df.isnull().sum() / len(df) * 100).round(1))

# Visualize missing patterns
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.tight_layout()
plt.savefig('missing_values.png')
```


![Illustration of data preprocessing and transformation pipelines](/images/blogs/pool-data/3.jpg)

### Types of Missing Data

Understanding **why** data is missing is crucial for choosing the right strategy:

**MCAR (Missing Completely At Random)**: The missingness has no relationship to any values. Example: a sensor randomly fails. This is the easiest to handle.

**MAR (Missing At Random)**: The missingness depends on observed data but not the missing value itself. Example: younger employees are less likely to fill in a salary field. You can use other columns to predict the missing values.

**MNAR (Missing Not At Random)**: The missingness depends on the unobserved value itself. Example: people with high incomes are less likely to report their income. This is the hardest to handle and requires domain knowledge.

### Strategy 1: Deletion

```python
# Drop rows with any missing values
df_clean = df.dropna()

# Drop rows where specific columns are missing
df_clean = df.dropna(subset=['name', 'salary'])

# Drop columns with too many missing values (e.g., more than 50%)
threshold = len(df) * 0.5
df_clean = df.dropna(axis=1, thresh=threshold)
```

**When to use deletion:**
- When missing data is MCAR and represents a small percentage (<5%) of the dataset
- When you have plenty of data and can afford to lose some

**When NOT to use deletion:**
- When data is not MCAR (deletion introduces bias)
- When you have limited data
- When the missing data is in your most important features

### Strategy 2: Simple Imputation

Replace missing values with a statistical measure:

```python
# Fill with mean (for normally distributed numeric data)
df['salary'] = df['salary'].fillna(df['salary'].mean())

# Fill with median (for skewed numeric data - more robust to outliers)
df['age'] = df['age'].fillna(df['age'].median())

# Fill with mode (for categorical data)
df['department'] = df['department'].fillna(df['department'].mode()[0])

# Fill with a constant
df['email'] = df['email'].fillna('unknown@company.com')

# Forward fill (for time series)
df['value'] = df['value'].fillna(method='ffill')

# Backward fill
df['value'] = df['value'].fillna(method='bfill')
```

### Strategy 3: Group-Based Imputation

Fill missing values based on similar groups in the data:

```python
# Fill missing salary with the median salary of the same department
df['salary'] = df.groupby('department')['salary'].transform(
    lambda x: x.fillna(x.median())
)

# If group median is also NaN, fall back to global median
df['salary'] = df['salary'].fillna(df['salary'].median())
```


![Visual showing techniques for cleaning and augmenting training datasets](/images/blogs/pool-data/4.jpg)

### Strategy 4: Predictive Imputation

Use a model to predict missing values:

```python
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# KNN Imputer - fills based on nearest neighbors
imputer = KNNImputer(n_neighbors=5)
df_numeric = df[['age', 'salary', 'years_experience']]
df_imputed = pd.DataFrame(
    imputer.fit_transform(df_numeric),
    columns=df_numeric.columns
)

# Iterative Imputer (MICE) - models each feature as a function of others
imputer = IterativeImputer(max_iter=10, random_state=42)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df_numeric),
    columns=df_numeric.columns
)
```

### Strategy 5: Indicator Variables

Sometimes the fact that a value is missing is itself informative:

```python
# Create a binary indicator for missingness
df['salary_missing'] = df['salary'].isnull().astype(int)

# Then impute the salary column
df['salary'] = df['salary'].fillna(df['salary'].median())

# The model now has two features: the imputed salary AND whether it was missing
```

## 2. Removing Duplicates

```python
# Find duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# See the duplicate rows
print(df[df.duplicated(keep=False)])

# Remove duplicates (keep first occurrence)
df = df.drop_duplicates()

# Remove duplicates based on specific columns
df = df.drop_duplicates(subset=['name', 'email'], keep='first')
```

## 3. Fixing Inconsistent Formatting

```python
# Standardize names: title case
df['name'] = df['name'].str.strip().str.title()

# Standardize departments: map inconsistent values
department_mapping = {
    'engineering': 'Engineering',
    'ENGINEERING': 'Engineering',
    'Engg': 'Engineering',
    'marketing': 'Marketing',
}
df['department'] = df['department'].replace(department_mapping)

# Or use string methods for simple cases
df['department'] = df['department'].str.strip().str.title()

# Standardize dates
df['hire_date'] = pd.to_datetime(df['hire_date'], format='mixed', dayfirst=False)
```


![Conceptual image of turning raw data into high-quality model inputs](/images/blogs/pool-data/5.jpg)

## 4. Handling Outliers

```python
# Detect outliers using IQR method
def detect_outliers_iqr(series, factor=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return (series < lower) | (series > upper)

# Find age outliers
age_outliers = detect_outliers_iqr(df['age'])
print(f"Age outliers: {age_outliers.sum()}")
print(df[age_outliers]['age'])

# Option 1: Remove outliers
df_clean = df[~detect_outliers_iqr(df['age'])]

# Option 2: Cap outliers (winsorization)
def cap_outliers(series, factor=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return series.clip(lower, upper)

df['age'] = cap_outliers(df['age'])

# Option 3: Replace with NaN and then impute
df.loc[detect_outliers_iqr(df['age']), 'age'] = np.nan
df['age'] = df['age'].fillna(df['age'].median())
```

## 5. Validating Data Types

```python
# Check and fix data types
print(df.dtypes)

# Convert string numbers to numeric
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['salary'] = pd.to_numeric(df['salary'], errors='coerce')

# Validate ranges
assert df['age'].between(18, 100).all(), "Invalid ages found"
assert df['salary'].gt(0).all(), "Invalid salaries found"
```

## A Complete Cleaning Pipeline

Here is a reusable cleaning function that brings it all together:

```python
def clean_dataset(df):
    """Comprehensive data cleaning pipeline"""
    df = df.copy()

    # 1. Remove exact duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicates")

    # 2. Standardize string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    # 3. Fix data types
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

    # 4. Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 5. Cap numeric outliers
    for col in numeric_cols:
        df[col] = cap_outliers(df[col])

    print(f"Final shape: {df.shape}")
    print(f"Remaining missing: {df.isnull().sum().sum()}")
    return df

df_clean = clean_dataset(df)
```

## Key Takeaways

1. Data cleaning is where you spend most of your time in real ML projects — accept and embrace this
2. Understand **why** data is missing (MCAR, MAR, MNAR) before choosing an imputation strategy
3. Simple imputation (mean, median, mode) works well for small amounts of MCAR data
4. For MAR data, use group-based or predictive imputation
5. Missing value indicators can add information even after imputation
6. Always check for duplicates, inconsistent formatting, invalid values, and outliers
7. Build reusable cleaning pipelines — you will use them on every project

Clean data is the foundation of good models. Invest the time here, and everything downstream becomes easier.

---

*Next: Feature engineering — the art of creating new, informative variables from raw data.*
