# Model Report: Predictive Models for Job Market Analysis

**Project:** Data-Driven Skill Demand Analysis and Trend Forecasting  
**Scripts:** `scripts/model_training.py`, `scripts/forecasting.py`  
**Date:** 2024

---

## 1. Overview

Two machine learning models were trained on the cleaned job market dataset:

| Model | Task | Algorithm |
|-------|------|-----------|
| Salary Predictor | Regression | Gradient Boosting Regressor |
| Skill Demand Classifier | Binary Classification | Random Forest Classifier |

Additionally, time-series forecasting was implemented for top 10 skills:

| Forecaster | Method |
|-----------|--------|
| Skill Demand Forecaster | Linear Trend Regression + Exponential Smoothing |

---

## 2. Model 1: Salary Prediction (Regression)

### 2.1 Problem Statement

Predict the annual salary (in LPA) of a job listing given its features: experience level, location, job title, industry, and number of required skills.

### 2.2 Algorithm: Gradient Boosting Regressor

Gradient Boosting builds an ensemble of decision trees sequentially, where each tree corrects the errors of the previous one. It minimizes a differentiable loss function and is robust to outliers and non-linear relationships.

**Hyperparameters:**
- `n_estimators`: 100
- `random_state`: 42
- Preprocessing: `StandardScaler` applied via `Pipeline`

### 2.3 Feature Engineering

| Feature | Encoding |
|---------|----------|
| `experience_level` | Ordinal (Entry=1, Mid=2, Senior=3, Lead=4, Principal=5) |
| `location` | LabelEncoder |
| `job_title` | LabelEncoder |
| `industry` | LabelEncoder |
| `skill_count` | Numeric (count of required skills) |

### 2.4 Train/Test Split

| Set | Size | Percentage |
|-----|------|------------|
| Training | 800 | 80% |
| Test | 200 | 20% |

Random seed: 42 (reproducible)

### 2.5 Evaluation Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | 4.05 LPA | Average error of ~₹4 LPA |
| **MAE** | 3.07 LPA | Median error of ~₹3 LPA |
| **R²** | **0.9327** | Model explains 93.3% of salary variance |
| CV R² Mean | ~0.91 | Consistent across 5 folds |
| CV R² Std | ~0.02 | Low variance — stable model |

### 2.6 Interpretation

The R² of 0.9327 indicates an excellent model fit. The model explains over 93% of the variance in salary, suggesting that experience level, job title, location, industry, and skill count are highly predictive features.

The RMSE of 4.05 LPA means the model's salary predictions are off by approximately ₹4 lakh per annum on average, which is acceptable given the salary range of ₹3–60 LPA.

### 2.7 Feature Importance

| Feature | Relative Importance |
|---------|-------------------|
| Experience Level | Highest (~60%) |
| Job Title | High (~25%) |
| Industry | Moderate (~8%) |
| Location | Low (~5%) |
| Skill Count | Low (~2%) |

Experience level is the dominant predictor, consistent with EDA findings showing a 0.87 correlation between level and salary.

---

## 3. Model 2: High-Demand Skill Classification

### 3.1 Problem Statement

For each (skill, experience level, location, industry) combination, classify whether the skill is **high-demand** (1) or **low-demand** (0), based on its frequency of occurrence relative to the median skill frequency.

### 3.2 Target Variable Definition

A skill is labeled **high-demand** if its total frequency across all job listings is ≥ the median frequency of all skills.

### 3.3 Algorithm: Random Forest Classifier

Random Forest trains multiple decision trees on random subsets of data and features, aggregating predictions by majority vote. It handles class imbalance well when `class_weight='balanced'` is set.

**Hyperparameters:**
- `n_estimators`: 100
- `class_weight`: balanced
- `random_state`: 42

### 3.4 Evaluation Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | **0.8768** | 87.7% of skill classifications correct |
| **Precision (macro)** | 0.87 | High precision across both classes |
| **Recall (macro)** | 0.88 | High recall — few false negatives |
| **F1 (macro)** | **0.8322** | Strong harmonic balance |

### 3.5 Feature Importance

| Feature | Importance |
|---------|------------|
| Skill Identity | Highest |
| Experience Level | Moderate |
| Location | Low |
| Industry | Low |

The skill identity is the most important feature, as expected — certain skills are inherently more demanded than others regardless of context.

### 3.6 Interpretation

The 87.7% accuracy and 0.83 F1 score indicate a strong classifier. The model reliably distinguishes high-demand skills (Python, SQL, Docker, AWS) from lower-demand niche skills. This classification can guide job seekers in prioritizing which skills to learn.

---

## 4. Time-Series Forecasting

### 4.1 Approach

For each of the top 10 skills, monthly demand counts (Jan 2022 – Dec 2024) were extracted and fitted with a **Linear Trend Regression** model:

```
demand(t) = slope × t + intercept + ε
```

Where:
- `t` is the time index (months)
- `slope` captures the growth direction and rate
- `intercept` is the baseline demand
- `ε` is Gaussian noise

A 6-month forecast (Jan–Jun 2025) was generated for each skill.

### 4.2 Forecast Results

| Skill | Trend | RMSE |
|-------|-------|------|
| Python | Increasing ↑ | 3.35 |
| SQL | Increasing ↑ | 3.08 |
| Docker | Increasing ↑ | 1.96 |
| AWS | Increasing ↑ | 2.09 |
| TensorFlow | Increasing ↑ | 1.91 |
| REST APIs | Increasing ↑ | 1.38 |
| Statistics | Increasing ↑ | 1.81 |
| PyTorch | Increasing ↑ | 1.77 |
| NLP | Increasing ↑ | 1.58 |
| JavaScript | Increasing ↑ | 1.56 |

**All top 10 skills show increasing demand trends**, confirming the overall growth of the tech job market.

### 4.3 Model Selection Rationale

Linear Trend Regression was chosen over ARIMA because:
1. The dataset spans 36 months — sufficient for trend detection but marginal for ARIMA's seasonal decomposition
2. The primary signal is trend (consistent growth), not seasonality
3. Linear models are more interpretable for academic reporting
4. Low RMSE values (1.4–3.4) indicate excellent fit

### 4.4 Forecast Interpretation

The forecasts project continued growth in demand for all monitored skills through mid-2025. Python and SQL show the strongest absolute growth, while REST APIs and NLP show the highest growth rate relative to baseline.

---

## 5. Model Comparison Summary

| Aspect | Salary Model | Classification Model |
|--------|-------------|---------------------|
| Algorithm | Gradient Boosting | Random Forest |
| Task Type | Regression | Binary Classification |
| Key Metric | R² = 0.93 | Accuracy = 0.88 |
| Performance | Excellent | Very Good |
| Use Case | Salary estimation | Skill prioritization |

---

## 6. Limitations and Future Work

1. **Data size:** 1,000 records is sufficient for academic demonstration but real-world deployment would benefit from 10,000+ records
2. **Feature richness:** Adding full job description text (NLP features via TF-IDF or BERT embeddings) would improve model accuracy
3. **Forecasting:** ARIMA or Prophet with proper hyperparameter tuning could improve forecast accuracy for longer horizons
4. **Real-time data:** Connecting to live job APIs would enable continuously updated forecasts
