---
title: "Code Tutorial: A Simple Linear Regression in Scikit-Learn"
date: 2026-06-08T10:00:00+05:30
draft: false
description: "A step-by-step code tutorial for implementing linear regression in scikit-learn — from importing data to evaluating and visualizing your model."
tags: ["Scikit-Learn", "Linear Regression", "Tutorial", "Python", "Machine Learning"]
categories: ["Machine Learning"]
image: "/images/blogs/pool-ml/1.jpg"
keywords: ["scikit-learn linear regression", "sklearn tutorial", "python regression tutorial", "ML code tutorial"]
---

This is a hands-on, code-first tutorial. If you have been following this series, you already understand the theory behind linear regression. Now it is time to get your hands dirty with scikit-learn — the most popular machine learning library in Python.

By the end of this post, you will have a complete, working linear regression implementation that you can adapt to your own datasets.

## Prerequisites

```python
# Install if needed: pip install scikit-learn numpy pandas matplotlib seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

print("All imports successful!")
```

## Part 1: Simple Linear Regression (One Feature)

Let us start with the simplest case — predicting a target from a single feature.

```python
# Generate synthetic data: study hours -> exam score
np.random.seed(42)

hours_studied = np.random.uniform(1, 10, 100)
exam_score = 10 + 8 * hours_studied + np.random.normal(0, 5, 100)

# Create a DataFrame
df = pd.DataFrame({
    'hours_studied': hours_studied,
    'exam_score': exam_score
})

print(df.describe().round(2))
print(f"\nCorrelation: {df['hours_studied'].corr(df['exam_score']):.4f}")
```

### Visualize the Data

```python
plt.figure(figsize=(10, 6))
plt.scatter(df['hours_studied'], df['exam_score'], alpha=0.6, edgecolors='black')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Hours Studied vs. Exam Score')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_plot.png')
plt.show()
```

### Split the Data

```python
# Reshape X for sklearn (needs 2D array)
X = df[['hours_studied']]  # Double brackets keep it as DataFrame (2D)
y = df['exam_score']

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
```


![Visual representation of machine learning model training and optimization](/images/blogs/pool-ml/3.jpg)

### Fit the Model

```python
# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# The model has learned two things:
print(f"Intercept (b): {model.intercept_:.4f}")
print(f"Slope (w): {model.coef_[0]:.4f}")
print(f"\nEquation: score = {model.intercept_:.2f} + {model.coef_[0]:.2f} * hours")
```

### Make Predictions

```python
# Predict on test data
y_pred = model.predict(X_test)

# Look at some predictions
comparison = pd.DataFrame({
    'Hours': X_test['hours_studied'].values,
    'Actual Score': y_test.values.round(1),
    'Predicted Score': y_pred.round(1),
    'Error': (y_test.values - y_pred).round(1)
})
print(comparison.head(10).to_string(index=False))
```

### Evaluate the Model

```python
# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"  Mean Squared Error (MSE):    {mse:.2f}")
print(f"  Root Mean Squared Error:     {rmse:.2f}")
print(f"  Mean Absolute Error (MAE):   {mae:.2f}")
print(f"  R-Squared (R²):             {r2:.4f}")
print(f"\nInterpretation: The model explains {r2*100:.1f}% of the variance in exam scores.")
print(f"Average prediction error: +/- {mae:.1f} points")
```

### Visualize the Fit

```python
plt.figure(figsize=(10, 6))

# Plot all data points
plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training data')
plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Test data')

# Plot the regression line
x_line = np.linspace(0, 11, 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Fit: y = {model.intercept_:.1f} + {model.coef_[0]:.1f}x')

plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title(f'Linear Regression (R² = {r2:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regression_line.png')
plt.show()
```


![Data flowing through a machine learning pipeline illustration](/images/blogs/pool-ml/5.jpg)

## Part 2: Multiple Linear Regression

Now let us use multiple features:

```python
# Generate data with multiple features
np.random.seed(42)
n = 300

study_data = pd.DataFrame({
    'hours_studied': np.random.uniform(1, 10, n),
    'hours_sleep': np.random.uniform(4, 10, n),
    'previous_score': np.random.uniform(40, 100, n),
    'attendance_pct': np.random.uniform(50, 100, n),
})

# Generate exam score with known relationships
study_data['exam_score'] = (
    5 * study_data['hours_studied']
    + 3 * study_data['hours_sleep']
    + 0.4 * study_data['previous_score']
    + 0.2 * study_data['attendance_pct']
    + np.random.normal(0, 5, n)
)

# Split
features = ['hours_studied', 'hours_sleep', 'previous_score', 'attendance_pct']
X = study_data[features]
y = study_data['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# Evaluate
y_pred_multi = model_multi.predict(X_test)
r2_multi = r2_score(y_test, y_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_test, y_pred_multi))

print(f"Multiple Regression R²: {r2_multi:.4f}")
print(f"RMSE: {rmse_multi:.2f}")

# Coefficients tell us each feature's contribution
print("\nFeature Coefficients:")
for feature, coef in zip(features, model_multi.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  intercept: {model_multi.intercept_:.4f}")
```

## Part 3: Feature Scaling

```python
# Scale features for more meaningful coefficient comparison
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)

# Now coefficients show RELATIVE importance
print("\nStandardized Coefficients (relative importance):")
for feature, coef in sorted(zip(features, model_scaled.coef_),
                              key=lambda x: abs(x[1]), reverse=True):
    print(f"  {feature}: {coef:.4f}")
```

## Part 4: Polynomial Regression

When the relationship is not linear, we can add polynomial features:

```python
# Generate nonlinear data
np.random.seed(42)
X_nonlinear = np.sort(np.random.uniform(0, 10, 100)).reshape(-1, 1)
y_nonlinear = 3 + 2 * X_nonlinear.flatten() - 0.3 * X_nonlinear.flatten()**2 + np.random.normal(0, 2, 100)

X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(
    X_nonlinear, y_nonlinear, test_size=0.2, random_state=42
)

# Compare linear vs polynomial
# Linear
model_linear = LinearRegression()
model_linear.fit(X_train_nl, y_train_nl)
r2_linear = r2_score(y_test_nl, model_linear.predict(X_test_nl))

# Polynomial (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_nl)
X_test_poly = poly.transform(X_test_nl)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train_nl)
r2_poly = r2_score(y_test_nl, model_poly.predict(X_test_poly))

print(f"Linear R²: {r2_linear:.4f}")
print(f"Polynomial (degree 2) R²: {r2_poly:.4f}")

# Visualize
plt.figure(figsize=(12, 5))

x_range = np.linspace(0, 10, 200).reshape(-1, 1)

plt.subplot(1, 2, 1)
plt.scatter(X_nonlinear, y_nonlinear, alpha=0.5, s=20)
plt.plot(x_range, model_linear.predict(x_range), 'r-', linewidth=2, label=f'Linear (R²={r2_linear:.3f})')
plt.title('Linear Fit')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_nonlinear, y_nonlinear, alpha=0.5, s=20)
x_range_poly = poly.transform(x_range)
plt.plot(x_range, model_poly.predict(x_range_poly), 'r-', linewidth=2, label=f'Poly deg=2 (R²={r2_poly:.3f})')
plt.title('Polynomial Fit')
plt.legend()

plt.tight_layout()
plt.savefig('polynomial_comparison.png')
plt.show()
```

## Part 5: Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
cv_scores = cross_val_score(model_multi, X, y, cv=5, scoring='r2')
cv_rmse = cross_val_score(model_multi, X, y, cv=5,
                           scoring='neg_root_mean_squared_error')

print("5-Fold Cross-Validation:")
print(f"  R² scores: {cv_scores.round(4)}")
print(f"  R² mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"  RMSE mean: {-cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
```


![Visualization of algorithm performance and evaluation metrics](/images/blogs/pool-ml/7.jpg)

## Part 6: Making Predictions on New Data

```python
# Predict for a new student
new_student = pd.DataFrame({
    'hours_studied': [7.0],
    'hours_sleep': [8.0],
    'previous_score': [75.0],
    'attendance_pct': [90.0]
})

predicted_score = model_multi.predict(new_student)
print(f"\nNew Student Prediction:")
print(f"  Input: {new_student.to_dict('records')[0]}")
print(f"  Predicted Exam Score: {predicted_score[0]:.1f}")

# Predict for multiple students
new_students = pd.DataFrame({
    'hours_studied': [2, 5, 8, 10],
    'hours_sleep': [5, 7, 8, 6],
    'previous_score': [50, 65, 80, 90],
    'attendance_pct': [60, 75, 95, 85]
})

predictions = model_multi.predict(new_students)
new_students['predicted_score'] = predictions.round(1)
print("\nBatch Predictions:")
print(new_students.to_string(index=False))
```

## Part 7: Saving and Loading the Model

```python
import joblib

# Save the model
joblib.dump(model_multi, 'exam_predictor.joblib')
print("Model saved!")

# Load the model
loaded_model = joblib.load('exam_predictor.joblib')

# Verify it works
test_pred = loaded_model.predict(new_student)
print(f"Loaded model prediction: {test_pred[0]:.1f}")
```

## Complete Template

Here is a reusable template you can adapt for any regression problem:

```python
"""
Linear Regression Template
========================
Adapt this for your own regression problems.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. LOAD DATA
df = pd.read_csv('your_data.csv')

# 2. SELECT FEATURES AND TARGET
feature_cols = ['feature1', 'feature2', 'feature3']
target_col = 'target'
X = df[feature_cols]
y = df[target_col]

# 3. SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. SCALE (optional but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. TRAIN
model = LinearRegression()  # or Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# 6. EVALUATE
y_pred = model.predict(X_test_scaled)
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# 7. CROSS-VALIDATE
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 8. SAVE
joblib.dump({'model': model, 'scaler': scaler}, 'model_package.joblib')
```

## Key Takeaways

1. Scikit-learn's API is consistent: `fit()`, `predict()`, `score()` work the same across all models
2. Always split your data before any preprocessing
3. Use `StandardScaler` for meaningful coefficient comparison
4. `PolynomialFeatures` lets linear regression capture nonlinear relationships
5. Cross-validation gives a more reliable performance estimate than a single split
6. Save both the model and scaler together for deployment

This template and workflow will serve you well for any regression problem you encounter.

---

*Next: K-Nearest Neighbors — a fundamentally different approach to ML that makes predictions based on similarity.*
