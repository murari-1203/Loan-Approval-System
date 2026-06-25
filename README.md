# CreditWise Loan Approval System

A Machine Learning based loan approval system that predicts whether a loan application should be **Approved** or **Rejected** using historical customer information.

The objective of this project is to replace manual, time-consuming loan verification with a data-driven prediction system capable of making fast, consistent, and unbiased decisions.

---

# Project Overview

Financial institutions receive thousands of loan applications containing information such as income, employment status, credit score, existing loans, savings, collateral value, and requested loan amount.

Instead of manually evaluating every application, this project learns patterns from historical loan records and predicts whether a new applicant is likely to be approved.

The workflow consists of:

1. Data Preprocessing
2. Missing Value Treatment
3. Exploratory Data Analysis
4. Feature Encoding
5. Correlation Analysis
6. Train-Test Split
7. Feature Scaling
8. Gaussian Naive Bayes Training
9. Model Evaluation

---

# Dataset

Each record represents one loan applicant.

The target variable is:

**Loan_Approved**

- **1** → Approved
- **0** → Rejected

The dataset contains applicant demographic, financial, employment, and credit-related attributes used for prediction.

---

# Project Flow

```
Raw Dataset
      │
      ▼
Handle Missing Values
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Encode Categorical Features
      │
      ▼
Correlation Analysis
      │
      ▼
Train-Test Split
      │
      ▼
Standard Feature Scaling
      │
      ▼
Gaussian Naive Bayes
      │
      ▼
Prediction
      │
      ▼
Performance Evaluation
```

---

# Mathematical Foundations

## 1. Missing Value Imputation

Real-world datasets often contain missing observations. Machine learning algorithms require complete numerical inputs, so missing values are replaced using statistical estimates.

### Mean Imputation

For numerical features,

\[
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
\]

Every missing value is replaced with

\[
x_{missing}=\bar{x}
\]

This preserves the average value of the feature while keeping all samples available for training.

---

### Most Frequent Imputation

For categorical variables,

\[
x_{missing}=\operatorname{mode}(X)
\]

where the mode represents the category occurring most frequently.

---

# Exploratory Data Analysis (EDA)

EDA is performed to understand the statistical properties of the dataset before model training.

It helps identify

- class distribution
- feature distributions
- relationships between variables
- potential outliers
- feature interactions

Understanding these characteristics improves confidence that the data is suitable for modeling.

---

# Feature Encoding

Machine learning algorithms operate on numerical values.

Categorical attributes are transformed into numerical representations.

### Label Encoding

Each category is assigned an integer.

If

\[
C=\{c_1,c_2,\ldots,c_k\}
\]

then

\[
c_i \rightarrow i
\]

This produces one numerical value for each category.

---

### One-Hot Encoding

Suppose a feature contains

\[
k
\]

categories.

It is transformed into

\[
k
\]

binary variables.

For a category

\[
c_i
\]

its encoded vector becomes

\[
(0,\ldots,1,\ldots,0)
\]

where only one position equals 1.

This prevents the model from assuming any numerical ordering between categories.

---

# Correlation Analysis

Correlation measures the strength of the linear relationship between two variables.

The Pearson Correlation Coefficient is

\[
r=
\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum(x_i-\bar{x})^2}
\sqrt{\sum(y_i-\bar{y})^2}}
\]

Its value lies between

\[
-1 \le r \le 1
\]

where

- **1** → Perfect positive correlation
- **0** → No linear relationship
- **−1** → Perfect negative correlation

The correlation heatmap provides a visual representation of these relationships across all numerical variables.

---

# Train-Test Split

The dataset is divided into two independent subsets.

Training set

\[
80\%
\]

Testing set

\[
20\%
\]

The training data is used to learn model parameters, while the testing data measures the model's ability to generalize to unseen samples.

---

# Feature Scaling

Features often have different numerical ranges.

Standardization transforms every feature so that

- Mean = 0
- Standard Deviation = 1

The transformation is

\[
z=\frac{x-\mu}{\sigma}
\]

where

- \(x\) is the original value
- \(\mu\) is the feature mean
- \(\sigma\) is the standard deviation

Standardization prevents large-valued features from dominating smaller-valued ones and creates comparable feature distributions.

---

# Gaussian Naive Bayes

The classifier is based on **Bayes' Theorem**.

For a class \(C\) and feature vector \(X\),

\[
P(C|X)=
\frac{P(X|C)P(C)}
{P(X)}
\]

where

- \(P(C)\) is the prior probability
- \(P(X|C)\) is the likelihood
- \(P(C|X)\) is the posterior probability

---

## Naive Independence Assumption

Naive Bayes assumes every feature is conditionally independent given the class.

\[
P(X|C)
=
\prod_{i=1}^{n}
P(x_i|C)
\]

This assumption significantly simplifies probability estimation.

---

## Gaussian Distribution

Since numerical features are continuous, each feature is assumed to follow a Gaussian distribution.

\[
P(x|C)=
\frac{1}
{\sqrt{2\pi\sigma^2}}
\exp
\left(
-\frac{(x-\mu)^2}
{2\sigma^2}
\right)
\]

where

- \(\mu\) is the class mean
- \(\sigma^2\) is the class variance

The posterior probability is computed for every class, and the class with the highest probability is selected.

---

# Model Evaluation

The classifier is evaluated using several standard classification metrics.

---

## Accuracy

Overall proportion of correct predictions.

\[
Accuracy=
\frac{TP+TN}
{TP+TN+FP+FN}
\]

---

## Precision

Measures how many predicted positive approvals are actually correct.

\[
Precision=
\frac{TP}
{TP+FP}
\]

---

## Recall

Measures how many actual approved loans are successfully identified.

\[
Recall=
\frac{TP}
{TP+FN}
\]

---

## F1 Score

Balances Precision and Recall.

\[
F1=
2
\cdot
\frac{Precision \times Recall}
{Precision+Recall}
\]

---

## Confusion Matrix

The confusion matrix summarizes model predictions.

|                | Predicted Positive | Predicted Negative |
|----------------|-------------------|-------------------|
| Actual Positive | TP | FN |
| Actual Negative | FP | TN |

It provides a detailed breakdown of correct and incorrect classifications.

---

# Machine Learning Pipeline

```
Loan Applications
        │
        ▼
Missing Value Imputation
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Encoding
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
Loan Approval Prediction
        │
        ▼
Performance Evaluation
```

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

---

# Conclusion

This project demonstrates a complete supervised machine learning pipeline for loan approval prediction. Beginning with preprocessing and statistical analysis, the workflow transforms raw applicant data into standardized numerical features, trains a Gaussian Naive Bayes classifier using probabilistic principles, and evaluates its predictive performance through established classification metrics. The result is an interpretable and efficient system for automated loan approval prediction.
