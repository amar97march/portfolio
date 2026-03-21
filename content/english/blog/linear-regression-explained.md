---
title: "Algorithm Spotlight: Linear Regression — Drawing the Best Fit Line"
date: 2026-05-27T10:00:00+05:30
draft: false
description: "A thorough explanation of linear regression — the foundational ML algorithm that fits a line to data, covering the math, implementation, assumptions, and practical considerations."
tags: ["Linear Regression", "Machine Learning", "Algorithms", "Statistics", "Regression"]
categories: ["Machine Learning"]
image: "https://images.unsplash.com/photo-1611162616305-c69b3fa7fbe0?w=1200&h=630&fit=crop&auto=format"
keywords: ["linear regression", "best fit line", "ordinary least squares", "regression algorithm", "ML algorithms"]
---

If machine learning has a "Hello World" algorithm, it is **linear regression**. It is the simplest, most interpretable, and most widely used predictive model in all of statistics and machine learning. And despite its simplicity, it remains remarkably powerful for a wide range of real-world problems.

Linear regression is the foundation upon which more complex algorithms are built. Understanding it deeply — not just how to call `model.fit()`, but the math, the assumptions, and the failure modes — makes everything else in ML more approachable.

## The Core Idea

Linear regression models the relationship between a dependent variable (what you want to predict) and one or more independent variables (features) by fitting a straight line through the data.

For a single feature, the model is:

```
y = mx + b
```

Or in ML notation:

```
y = w*x + b
```

Where:
- `y` is the predicted value
- `x` is the input feature
- `w` is the weight (slope)
- `b` is the bias (y-intercept)

For multiple features:

```
y = w1*x1 + w2*x2 + w3*x3 + ... + b
```

The goal is to find the values of `w` and `b` that minimize the difference between the predicted values and the actual values.

## The Math: Ordinary Least Squares

The most common method for finding the best fit line is **Ordinary Least Squares (OLS)**. It minimizes the sum of squared differences between predicted and actual values:

```
Loss = Σ (y_actual - y_predicted)²
     = Σ (y_i - (w*x_i + b))²
```

Why squared? Because we want to penalize both positive and negative errors (simple differences would cancel out), and we want to penalize large errors more than small ones.

### The Closed-Form Solution

For linear regression, there is actually a mathematical formula that gives the optimal weights directly:

```
w = (X^T X)^{-1} X^T y
```

```python
import numpy as np

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Add bias column (column of ones)
X_b = np.c_[np.ones((100, 1)), X]

# Closed-form solution (Normal Equation)
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(f"Intercept (b): {theta[0][0]:.4f}")  # Should be close to 4
print(f"Slope (w): {theta[1][0]:.4f}")      # Should be close to 3
```

### Gradient Descent Approach

For large datasets, the closed-form solution is computationally expensive (matrix inversion is O(n^3)). Gradient descent is the alternative:

```python
# Gradient descent for linear regression
learning_rate = 0.1
n_iterations = 1000
m = len(X_b)

theta = np.random.randn(2, 1)  # Random initialization

for iteration in range(n_iterations):
    # Compute predictions
    predictions = X_b.dot(theta)

    # Compute gradients
    gradients = (2/m) * X_b.T.dot(predictions - y)

    # Update parameters
    theta -= learning_rate * gradients

    if iteration % 200 == 0:
        mse = np.mean((predictions - y) ** 2)
        print(f"Iteration {iteration}: MSE = {mse:.4f}")

print(f"\nFinal: intercept={theta[0][0]:.4f}, slope={theta[1][0]:.4f}")
```


![Visual representation of machine learning model training and optimization](https://picsum.photos/seed/linear-regression-explained-1/800/450)

## Using Scikit-Learn

In practice, you will use scikit-learn:

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate data
np.random.seed(42)
X = 2 * np.random.rand(200, 1)
y = 4 + 3 * X.flatten() + np.random.randn(200)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit
model = LinearRegression()
model.fit(X_train, y_train)

print(f"Intercept: {model.intercept_:.4f}")
print(f"Coefficient: {model.coef_[0]:.4f}")

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
```

## Multiple Linear Regression

Real problems have multiple features:

```python
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# House price prediction with multiple features
np.random.seed(42)
n = 500

data = pd.DataFrame({
    'square_feet': np.random.uniform(800, 4000, n),
    'bedrooms': np.random.randint(1, 6, n),
    'bathrooms': np.random.randint(1, 4, n),
    'age': np.random.uniform(0, 50, n),
})

# Generate target with known relationship
data['price'] = (
    150 * data['square_feet']
    + 20000 * data['bedrooms']
    + 15000 * data['bathrooms']
    - 1000 * data['age']
    + 50000
    + np.random.normal(0, 30000, n)
)

# Fit model
X = data[['square_feet', 'bedrooms', 'bathrooms', 'age']]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# Interpret coefficients
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: ${coef:,.0f} per unit increase")
print(f"Base price (intercept): ${model.intercept_:,.0f}")

# Evaluate
y_pred = model.predict(X_test)
print(f"\nR² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
```

## Understanding R-Squared

**R-squared (R2)** measures how much of the variance in the target variable is explained by the model.

- R2 = 1.0: The model explains all variance (perfect prediction)
- R2 = 0.0: The model explains no variance (no better than predicting the mean)
- R2 < 0: The model is worse than predicting the mean

```python
# R² intuition
y_mean = y_test.mean()
ss_total = np.sum((y_test - y_mean) ** 2)     # Total variance
ss_residual = np.sum((y_test - y_pred) ** 2)   # Unexplained variance
r2 = 1 - (ss_residual / ss_total)
print(f"R²: {r2:.4f}")
```

## Assumptions of Linear Regression

Linear regression makes several assumptions. Violating them can lead to unreliable results:


![Data flowing through a machine learning pipeline illustration](https://picsum.photos/seed/linear-regression-explained-2/800/450)

### 1. Linearity
The relationship between features and target must be approximately linear. Check with scatter plots.

### 2. Independence
Observations should be independent of each other. Violated in time series data.

### 3. Homoscedasticity
The variance of errors should be constant across all levels of the features.

### 4. Normality of Residuals
The errors should be approximately normally distributed. Check with a Q-Q plot.


![Visualization of algorithm performance and evaluation metrics](https://picsum.photos/seed/linear-regression-explained-3/800/450)

### 5. No Multicollinearity
Features should not be highly correlated with each other.

```python
# Check for multicollinearity using VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
# VIF > 10 indicates problematic multicollinearity
```

## Regularization: Ridge and Lasso

When you have many features or multicollinearity, regularization helps prevent overfitting:

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge (L2): penalizes large coefficients
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso (L1): can set coefficients to exactly zero (feature selection)
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

# Elastic Net: combination of Ridge and Lasso
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X_train, y_train)

# Compare coefficients
print("Ridge coefficients:", ridge.coef_.round(2))
print("Lasso coefficients:", lasso.coef_.round(2))
print("Some Lasso coefficients may be exactly 0 (feature selection)")
```

## When Linear Regression Works Well

- The relationship between features and target is approximately linear
- You need an interpretable model (coefficients tell you exactly how each feature affects the prediction)
- You have a moderate number of features relative to observations
- You need a quick, reliable baseline model

## When It Fails

- Nonlinear relationships (use polynomial features or switch to a nonlinear model)
- Outliers heavily influence the fit (use robust regression)
- Too many features relative to observations (use regularization)
- Target variable is categorical (use logistic regression instead)

## Key Takeaways

1. Linear regression fits a line (or hyperplane) that minimizes squared errors
2. The Normal Equation gives a closed-form solution; gradient descent scales better
3. R-squared measures how much variance the model explains
4. Check assumptions: linearity, independence, homoscedasticity, normality, no multicollinearity
5. Use Ridge, Lasso, or Elastic Net for regularization when needed
6. Despite its simplicity, linear regression remains one of the most useful and interpretable models in ML

---

*Next: Logistic Regression — adapting the linear model for classification problems.*
