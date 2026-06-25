# CreditWise Loan Approval System

A Machine Learning project that predicts whether a loan application should be **Approved** or **Rejected** using applicant financial, demographic, and credit information. The goal is to assist financial institutions in making faster, consistent, and data-driven lending decisions based on historical loan records.

---

## Project Overview

Traditional loan approval relies on manual verification of customer information, making the process slow and prone to inconsistencies. This project builds a classification model that learns patterns from previous loan applications and predicts loan approval for new applicants.

---

## Dataset

The dataset contains applicant information such as:

* Applicant & Co-applicant Income
* Employment Status
* Age
* Marital Status
* Number of Dependents
* Credit Score
* Existing Loans
* Debt-to-Income Ratio
* Savings
* Collateral Value
* Loan Amount
* Loan Term
* Loan Purpose
* Property Area
* Education Level
* Gender
* Employer Category

**Target Variable**

* **1** → Loan Approved
* **0** → Loan Rejected

---

## Project Workflow

```
Raw Dataset
      │
      ▼
Missing Value Imputation
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Categorical Encoding
      │
      ▼
Correlation Analysis
      │
      ▼
Train-Test Split
      │
      ▼
Feature Standardization
      │
      ▼
Gaussian Naive Bayes
      │
      ▼
Model Evaluation
```

---

# Mathematical Concepts

## Missing Value Imputation

### Mean Imputation

Missing numerical values are replaced by the arithmetic mean of the feature.

**x̄ = (1/n) Σxᵢ**

This preserves the central tendency of the data.

### Most Frequent Imputation

Missing categorical values are replaced by the most frequently occurring category (Mode).

---

## Correlation

Feature relationships are measured using the **Pearson Correlation Coefficient**.

**r = Cov(X,Y) / (σₓσᵧ)**

where

* **r = 1** → Perfect Positive Correlation
* **r = 0** → No Linear Correlation
* **r = -1** → Perfect Negative Correlation

Correlation analysis helps understand how strongly variables are related.

---

## Feature Standardization

Features are transformed to have zero mean and unit variance.

**z = (x − μ) / σ**

where

* **μ** = Mean
* **σ** = Standard Deviation

Standardization ensures all features contribute on a comparable scale.

---

## Gaussian Naive Bayes

The classifier is based on **Bayes' Theorem**.

**P(C│X) = [P(X│C) × P(C)] / P(X)**

where

* **P(C)** = Prior Probability
* **P(X│C)** = Likelihood
* **P(C│X)** = Posterior Probability

Naive Bayes assumes that all input features are conditionally independent given the target class.

For continuous features, the likelihood is modeled using the Gaussian distribution:

**P(x│C) = (1 / √(2πσ²)) · e^(-(x−μ)² / 2σ²)**

---

## Model Evaluation

The classifier is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

These metrics measure the overall predictive performance and classification quality of the model.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---


## Conclusion

This project demonstrates a complete machine learning pipeline for loan approval prediction, covering data preprocessing, statistical analysis, feature scaling, probabilistic classification using Gaussian Naive Bayes, and performance evaluation. It highlights how mathematical concepts and machine learning can be applied to automate financial decision-making.
