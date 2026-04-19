# Ola Driver Churn Prediction

**Live Demo:** https://ola-driver-churn-prediction.streamlit.app/  
**Model AUC-ROC:** 0.93  
**Tech Stack:** Python · XGBoost · SMOTE · scikit-learn · Streamlit

---

## Problem Statement

Ola faces significant revenue loss from driver churn. This project builds a machine learning model to predict which drivers are likely to leave the platform, enabling proactive retention interventions.

## What Makes This Project Different

**Caught and fixed critical data leakage** — the original approach used `LastWorkingDate` to compute `Tenure`, which directly encodes the target label. Fixed by computing tenure from the reporting period (`MMM-YY`) instead — reducing inflated AUC by ~15%.

**Feature engineering drove results** — 4 engineered features became the top predictors:
- `Rating_Change_mean` — #1 most important feature (16% importance)
- `Rating_Trend_Encoded_mean` — #2 most important (11% importance)
- `Income_Grade_Ratio` — income vs grade mismatch signal
- `Avg_Business_Per_Month` — normalised business activity

## Model Results

| Model | Test AUC-ROC | Accuracy |
|---|---|---|
| Logistic Regression | baseline | — |
| Random Forest (base) | 0.9095 | 86% |
| Random Forest (tuned) | 0.9156 | 87% |
| **XGBoost (base)** | **0.9318** | **89%** |

## Key Findings

- Drivers churn most in the **first 6 months** of joining
- **Rating trajectory** (improving vs declining) predicts churn better than absolute rating
- **Grade 1 drivers** churn at ~80% — need fast-track progression incentives
- **Gender and Education** carry zero predictive signal — retention programs should focus on performance metrics only

## Project Structure

```
ola-driver-churn/
├── Ola_Project_Submission.ipynb   # Full notebook
├── app.py                         # Streamlit app
├── requirements.txt               # Dependencies
├── ola_churn_model/
│   ├── ola_churn_xgb.pkl         # Trained XGBoost model
│   ├── label_encoder.pkl         # City encoder
│   ├── outlier_caps.pkl          # Training outlier caps
│   └── feature_columns.pkl       # Feature column order
└── README.md
```

## How to Run Locally

```bash
git clone https://github.com/Prashant-Chavan/ola-driver-churn
cd ola-driver-churn
pip install -r requirements.txt
streamlit run app.py
```
