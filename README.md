# Ames Housing Price Prediction

End-to-end ML project on the Ames Housing dataset — covering exploratory data analysis, predictive modeling with three algorithms, and SHAP-based model explainability.

---

## Project Structure

```
├── 01_EDA.ipynb                  # Task 1: Exploratory Data Analysis
├── 02_Predictive_Modeling.ipynb  # Task 2: Modeling, Evaluation & SHAP Analysis
├── requirements.txt
└── README.md
```

---

## Overview

### Task 1 — EDA
- Missing value audit and targeted imputation (median, mode, domain-based "None"/0 fills)
- Distribution analysis with log transformation of the target (`SalePrice`)
- Outlier detection and removal (`Gr Liv Area > 4,000 sq ft`)
- Correlation heatmap identifying top 10 price drivers
- Neighborhood analysis: top 10 vs bottom 10 by median sale price
- Year-based trends in housing prices

### Task 2 — Predictive Modeling
Three regression models trained on log-transformed sale prices, evaluated with 5-Fold Cross-Validation:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 0.1053 | 0.0790 | 0.9322 |
| Random Forest | 0.1155 | 0.0834 | 0.9184 |
| **XGBoost (Tuned)** | **0.0938** | **0.0708** | **0.9462** |

XGBoost with `RandomizedSearchCV` hyperparameter tuning achieved the best performance — explaining ~94.6% of variance in housing sale prices.

SHAP analysis was applied to the final XGBoost model to explain both global feature importance and individual predictions.

---

## Key Findings

- **Overall Quality** is the single strongest predictor across all methods (correlation, Random Forest importance, and SHAP)
- **Living area, basement size, and garage features** are the next most impactful size-related drivers
- **Newer and recently renovated homes** consistently command higher prices
- **Log-transforming the target** was a critical step that improved all three models by normalizing the skewed sale price distribution

---

## Tech Stack

- **Python** — pandas, NumPy, Scikit-learn, XGBoost, SHAP
- **Visualization** — Matplotlib, Seaborn
- **Modeling** — LinearRegression, RandomForestRegressor, XGBRegressor, RandomizedSearchCV, KFold CV

---

## Setup

```bash
pip install -r requirements.txt
```

Then open either notebook in Jupyter or VS Code. Both notebooks are self-contained — Task 2 reproduces the cleaning pipeline from Task 1 so it can be run independently.

---

## Dataset

Ames Housing dataset (900-row sample). Place `housing_data_900.csv` in the root directory before running the notebooks.
