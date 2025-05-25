
# COVID-19 Urgency Classification

--- 

This project is a machine learning pipeline for predicting the **urgency of hospitalization** for COVID-19 patients based on their symptoms and demographic information.

## Objective

The goal is to:
- Analyze the characteristics of COVID-19 patients with urgent hospitalization needs.
- Build and evaluate machine learning models (kNN and Logistic Regression) to classify patients by urgency.
- Visualize model performance and interpret results for different real-world policy scenarios.

---

## Dataset

The dataset (`covid.csv`) includes:
- Symptoms: `cough`, `fever`, `chills`, `sore_throat`, `headache`, `fatigue`
- Demographics: `age`, `gender`, etc.
- Target variable: `Urgency` (1 = high urgency, 0 = low urgency)

Missing values were handled using **KNN imputation**.

---

##  Exploratory Data Analysis

Visualizations include:
- Distribution of urgent cases by **age group**
- Most common **symptoms** in high-urgency patients
- Comparison of **cough frequency** by urgency level

--- 

##  Models

Two models were trained and compared:
- **k-Nearest Neighbors (kNN)** – with `k=7`
- **Logistic Regression** – with regularization (`C=0.1`)

Metrics evaluated:
- Accuracy
- Recall / Precision / F1-score
- Specificity
- ROC-AUC
- Confusion Matrix

---

##  ROC Curve & Thresholding

ROC curves were plotted for both models. Based on different real-world constraints, classifier choices were proposed:

| Scenario | Classifier & Threshold | Justification |
|----------|------------------------|---------------|
| **Brazil** | Logistic Regression with a **high threshold** | Prioritizes low **false positives** |
| **Germany** | Logistic Regression with a **low threshold** | Prioritizes high **true positives** |
| **India** | kNN with a **moderate threshold** | Accepts trade-off between TPR and FPR |

These were visualized using shaded regions on the ROC curve.

---

## How to Run

1. Install dependincies

``` bash
pip install -r requirements.txt
```

2. Run the script

```bash
python covid_model_analysis.py
```