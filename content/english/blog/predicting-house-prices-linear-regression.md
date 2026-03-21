---
title: "Use Case: Predicting House Prices with Linear Regression"
date: 2026-06-02T10:00:00+05:30
draft: false
description: "A hands-on walkthrough of building a house price prediction model using linear regression — from data exploration to feature engineering to model evaluation."
tags: ["Linear Regression", "Use Case", "House Prices", "Scikit-Learn", "Machine Learning"]
categories: ["Machine Learning"]
image: "https://picsum.photos/seed/predicting-house-prices-linear-regression-cover/1200/630"
keywords: ["house price prediction", "linear regression example", "real estate ML", "sklearn regression", "ML use case"]
---

The best way to understand an algorithm is to apply it to a real problem. In this post, we are going to build a **house price prediction model** using linear regression — walking through the entire ML workflow from data exploration to final evaluation.

This is one of the most classic ML use cases, and for good reason. Everyone understands housing. The features are intuitive (bigger house = more expensive). And the problem is genuinely useful — real estate platforms, banks, and property assessors all rely on models like this.

## Setting Up the Project

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# For reproducibility
np.random.seed(42)
```

## Creating a Realistic Dataset

While you would normally load a real dataset (like the Boston Housing dataset or Kaggle's Ames Housing), let me create a realistic synthetic dataset so you can follow along:

```python
n_samples = 2000

data = pd.DataFrame({
    'square_feet': np.random.normal(1800, 500, n_samples).clip(600, 5000),
    'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.15, 0.40, 0.30, 0.10]),
    'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples,
                                   p=[0.10, 0.10, 0.30, 0.20, 0.15, 0.10, 0.05]),
    'garage_cars': np.random.choice([0, 1, 2, 3], n_samples, p=[0.10, 0.30, 0.45, 0.15]),
    'year_built': np.random.randint(1950, 2024, n_samples),
    'lot_size': np.random.normal(8000, 3000, n_samples).clip(2000, 25000),
    'neighborhood_quality': np.random.choice([1, 2, 3, 4, 5], n_samples,
                                              p=[0.10, 0.20, 0.35, 0.25, 0.10]),
    'has_pool': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
    'condition': np.random.choice([1, 2, 3, 4, 5], n_samples,
                                   p=[0.05, 0.15, 0.35, 0.30, 0.15]),
})

# Generate price with known relationships + noise
data['price'] = (
    120 * data['square_feet']
    + 15000 * data['bedrooms']
    + 20000 * data['bathrooms']
    + 25000 * data['garage_cars']
    + 500 * (data['year_built'] - 1950)
    + 5 * data['lot_size']
    + 40000 * data['neighborhood_quality']
    + 30000 * data['has_pool']
    + 15000 * data['condition']
    + np.random.normal(0, 30000, n_samples)  # Random noise
    - 50000  # Base adjustment
)

data['price'] = data['price'].clip(50000, 1500000)

print(f"Dataset shape: {data.shape}")
print(f"\nPrice statistics:")
print(data['price'].describe())
```

## Step 1: Exploratory Data Analysis

Before building any model, you must understand your data.


![Illustration of data processing pipeline and feature analysis](https://picsum.photos/seed/predicting-house-prices-linear-regression-1/800/450)

```python
# Overview
print(data.info())
print("\n", data.describe().round(0))

# Check for missing values
print(f"\nMissing values:\n{data.isnull().sum()}")

# Target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(data['price'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title('Price Distribution')
axes[0].set_xlabel('Price ($)')

axes[1].hist(np.log1p(data['price']), bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_title('Log Price Distribution')
axes[1].set_xlabel('Log Price')

plt.tight_layout()
plt.savefig('price_distribution.png')

# Correlation analysis
correlation = data.corr()['price'].sort_values(ascending=False)
print("\nCorrelation with price:")
print(correlation)
```

```python
# Scatter plots of top features vs price
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
top_features = ['square_feet', 'neighborhood_quality', 'bathrooms',
                'garage_cars', 'year_built', 'condition']

for ax, feature in zip(axes.flatten(), top_features):
    ax.scatter(data[feature], data['price'], alpha=0.3, s=10)
    ax.set_xlabel(feature)
    ax.set_ylabel('Price')
    ax.set_title(f'{feature} vs Price')

plt.tight_layout()
plt.savefig('feature_scatter.png')
```

## Step 2: Feature Engineering

```python
# Create informative features
data['age'] = 2026 - data['year_built']
data['total_rooms'] = data['bedrooms'] + data['bathrooms']
data['sqft_per_room'] = data['square_feet'] / data['total_rooms']
data['lot_to_house'] = data['lot_size'] / data['square_feet']
data['is_new'] = (data['age'] < 10).astype(int)
data['quality_x_condition'] = data['neighborhood_quality'] * data['condition']
data['sqft_x_quality'] = data['square_feet'] * data['neighborhood_quality']

# Log transform of skewed features
data['log_square_feet'] = np.log1p(data['square_feet'])
data['log_lot_size'] = np.log1p(data['lot_size'])

print("New features added. Shape:", data.shape)
```

## Step 3: Prepare the Data

```python
# Select features
feature_columns = [
    'square_feet', 'bedrooms', 'bathrooms', 'garage_cars',
    'lot_size', 'neighborhood_quality', 'has_pool', 'condition',
    'age', 'total_rooms', 'sqft_per_room', 'lot_to_house',
    'is_new', 'quality_x_condition', 'sqft_x_quality'
]

X = data[feature_columns]
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Step 4: Train and Evaluate Models


![Diagram showing algorithm comparison and performance metrics](https://picsum.photos/seed/predicting-house-prices-linear-regression-2/800/450)

```python
# Model 1: Basic Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Model 2: Ridge Regression
ridge = Ridge(alpha=10.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

# Model 3: Lasso Regression
lasso = Lasso(alpha=100.0)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

# Evaluation function
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n{name}:")
    print(f"  RMSE:  ${rmse:,.0f}")
    print(f"  MAE:   ${mae:,.0f}")
    print(f"  R²:    {r2:.4f}")
    print(f"  MAPE:  {mape:.2f}%")
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Ridge Regression", y_test, y_pred_ridge)
evaluate_model("Lasso Regression", y_test, y_pred_lasso)
```

## Step 5: Cross-Validation

```python
# Cross-validation for more robust evaluation
from sklearn.model_selection import cross_val_score

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=10)': Ridge(alpha=10),
    'Lasso (alpha=100)': Lasso(alpha=100),
}

print("\n5-Fold Cross-Validation Results:")
print("=" * 50)

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train,
                             cv=5, scoring='r2')
    rmse_scores = cross_val_score(model, X_train_scaled, y_train,
                                   cv=5, scoring='neg_root_mean_squared_error')
    print(f"\n{name}:")
    print(f"  R² = {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"  RMSE = ${-rmse_scores.mean():,.0f} (+/- ${rmse_scores.std():,.0f})")
```

## Step 6: Interpret the Model

```python
# Feature importance (coefficients)
coef_df = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': lr.coef_,
    'Abs_Coefficient': np.abs(lr.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Coefficients (standardized):")
print(coef_df.to_string(index=False))

# Visualize
plt.figure(figsize=(12, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.xlabel('Coefficient (Standardized)')
plt.title('Feature Importance in Linear Regression')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

## Step 7: Residual Analysis


![Visual representation of machine learning model architecture and data flow](https://picsum.photos/seed/predicting-house-prices-linear-regression-3/800/450)

```python
# Check residuals to validate model assumptions
residuals = y_test - y_pred_lr

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals vs Predicted
axes[0, 0].scatter(y_pred_lr, residuals, alpha=0.3, s=10)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Predicted Price')
axes[0, 0].set_ylabel('Residual')
axes[0, 0].set_title('Residuals vs Predicted')

# Residual distribution
axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_title('Residual Distribution')

# Q-Q plot
from scipy import stats
stats.probplot(residuals, plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')

# Actual vs Predicted
axes[1, 1].scatter(y_test, y_pred_lr, alpha=0.3, s=10)
axes[1, 1].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('Actual Price')
axes[1, 1].set_ylabel('Predicted Price')
axes[1, 1].set_title('Actual vs Predicted')

plt.tight_layout()
plt.savefig('residual_analysis.png')
```

## Step 8: Making Predictions

```python
# Predict price for a new house
new_house = pd.DataFrame({
    'square_feet': [2200],
    'bedrooms': [4],
    'bathrooms': [2.5],
    'garage_cars': [2],
    'lot_size': [9000],
    'neighborhood_quality': [4],
    'has_pool': [1],
    'condition': [4],
    'age': [5],
    'total_rooms': [6.5],
    'sqft_per_room': [2200/6.5],
    'lot_to_house': [9000/2200],
    'is_new': [1],
    'quality_x_condition': [16],
    'sqft_x_quality': [2200*4],
})

new_house_scaled = scaler.transform(new_house)
predicted_price = lr.predict(new_house_scaled)

print(f"\nPredicted price: ${predicted_price[0]:,.0f}")
```

## What We Learned

This end-to-end project demonstrates several important lessons:

1. **EDA first, model second.** Understanding your data through visualization and statistics guides every subsequent decision.

2. **Feature engineering matters.** The engineered features (age, total_rooms, sqft_per_room, interaction terms) often contribute significantly to model performance.

3. **Cross-validation over single splits.** A single train-test split can give misleading results. Cross-validation provides a more reliable performance estimate.

4. **Residual analysis validates assumptions.** Checking residuals reveals whether linear regression's assumptions hold for your data.

5. **Regularization helps.** Ridge and Lasso can improve generalization, especially with many features or multicollinearity.

6. **Interpretability is a feature.** Linear regression tells you exactly how each feature contributes to the prediction — invaluable for stakeholders who need to understand the model.

---

*Next: Building a spam detector with logistic regression — applying classification to a practical problem.*
