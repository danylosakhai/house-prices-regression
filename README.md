# House Prices — Regression Pipeline

A regression project built while learning data science: predicting home sale prices from structured tabular data, comparing multiple models and tuning hyperparameters.

## Problem

Predict the sale price of residential homes based on ~80 features (size, quality, location, amenities) — a regression task, unlike the binary classification in my [Titanic project](https://github.com/danylosakhai/titanic-survival-prediction).

## Dataset

[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) (Kaggle) — 1460 records, ~80 features, mix of numeric and categorical.

## What's inside

- **Missing value handling** — distinguishing *structural* absence (e.g. no pool, no garage → filled with `'None'`) from *genuinely unknown* values (e.g. `LotFrontage` → imputed with neighborhood median)
- **Target analysis** — log-transforming the right-skewed `SalePrice` for better model performance
- **EDA** — correlation analysis, multicollinearity checks, outlier detection and removal
- **Encoding** — ordinal encoding for quality ratings (preserving order: Poor → Excellent), one-hot encoding for nominal categories (e.g. `Neighborhood`)
- **Feature scaling** — standardizing continuous numeric features
- **Modeling** — Linear Regression, Random Forest, and XGBoost, compared
- **Hyperparameter tuning** — GridSearchCV with 5-fold cross-validation on XGBoost

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | $15,572 | $22,020 | 0.899 |
| Random Forest (default) | $16,416 | $23,842 | 0.876 |
| XGBoost (default) | $16,394 | $23,630 | 0.872 |
| **XGBoost (tuned)** | **$14,834** | **$20,632** | **0.909** |

**Key takeaway:** out-of-the-box tree models underperformed a simple linear baseline — a reminder that model complexity only pays off once properly tuned. After hyperparameter tuning, XGBoost became the best-performing model.

## Tech stack

Python · pandas · NumPy · matplotlib · seaborn · scikit-learn · XGBoost · Jupyter Notebook

## Project structure

```
├── train.csv
├── data_description.txt
├── House_Prices.ipynb
└── README.md
```

## Running locally

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
jupyter notebook House_Prices.ipynb
```

## Possible next steps

- Feature importance analysis for the tuned XGBoost model
- Tuning Random Forest hyperparameters
- Ensembling / stacking multiple models
