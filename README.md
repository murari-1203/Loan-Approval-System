# CreditWise Loan Approval System

A Machine Learning project that predicts whether a loan application should be **Approved** or **Rejected** using applicant financial, demographic, and credit information. The system learns patterns from historical loan records to assist financial institutions in making faster, consistent, and data-driven lending decisions.

---

# Project Overview

Traditional loan approval relies on manual verification of customer details, making the process time-consuming and susceptible to inconsistencies. This project automates the decision-making process by training a **Gaussian Naive Bayes** classifier on historical loan application data.

---

# Dataset

Each record represents a loan applicant with features including:

* Applicant & Co-applicant Income
* Employment Status
* Age
* Marital Status
* Number of Dependents
* Credit Score
* Existing Loans
* Debt-to-Income Ratio (DTI)
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

# Project Workflow

```text
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
Performance Evaluation
```

---

# Mathematical Foundations

## 1. Missing Value Imputation

### Mean Imputation

Missing numerical values are replaced with the arithmetic mean of the feature.

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

Each missing value is replaced as

$$
x_{\text{missing}}=\bar{x}
$$

### Most Frequent Imputation

Missing categorical values are replaced using the mode, i.e., the category with the highest frequency.

---

## 2. Correlation Analysis

Relationships between numerical features are measured using the **Pearson Correlation Coefficient**.

$$
r=
\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}
\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}
$$

where

* **r = 1** → Perfect Positive Correlation
* **r = 0** → No Linear Correlation
* **r = -1** → Perfect Negative Correlation

---

## 3. Feature Standardization

Features are transformed to have zero mean and unit variance.

$$
z=\frac{x-\mu}{\sigma}
$$

where

* **μ** = Mean
* **σ** = Standard Deviation

Standardization ensures all features are on a comparable scale before training.

---

## 4. Gaussian Naive Bayes

The classifier is based on **Bayes' Theorem**.

$$
P(C|X)=\frac{P(X|C)P(C)}{P(X)}
$$

where

* $P(C)$ = Prior Probability
* $P(X|C)$ = Likelihood
* $P(C|X)$ = Posterior Probability

### Conditional Independence

Naive Bayes assumes that features are conditionally independent given the target class.

$$
P(X|C)=\prod_{i=1}^{n}P(x_i|C)
$$

### Gaussian Distribution

For continuous features, the likelihood is modeled using a Gaussian (Normal) distribution.

$$
P(x|C)=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right)
$$

The class with the highest posterior probability is selected as the prediction.

---

# Model Evaluation

The trained classifier is evaluated using standard classification metrics.

### Accuracy

$$
\text{Accuracy}=
\frac{TP+TN}
{TP+TN+FP+FN}
$$

### Precision

$$
\text{Precision}=
\frac{TP}
{TP+FP}
$$

### Recall

$$
\text{Recall}=
\frac{TP}
{TP+FN}
$$

### F1-Score

$$
F_1=
2\cdot
\frac{\text{Precision}\times\text{Recall}}
{\text{Precision}+\text{Recall}}
$$

### Confusion Matrix

The confusion matrix summarizes the model's predictions into:

* **True Positives (TP)**
* **True Negatives (TN)**
* **False Positives (FP)**
* **False Negatives (FN)**

---

# Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---




# Conclusion

This project presents an end-to-end machine learning pipeline for loan approval prediction. From data preprocessing and statistical analysis to probabilistic classification using Gaussian Naive Bayes, it demonstrates how mathematical principles and machine learning techniques can be combined to automate loan approval decisions efficiently and consistently.
