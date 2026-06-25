# Data Preprocessing

## 1. Missing Value Imputation

Real-world datasets often contain missing values. Removing rows with missing values may lead to the loss of valuable information. Instead, missing values are replaced using **SimpleImputer**.

### Numerical Features

Missing numerical values are replaced using the **mean** of the corresponding feature.

### Formula

\[
\mu=\frac{1}{n}\sum_{i=1}^{n}x_i
\]

Every missing numerical value is replaced by

\[
x_{missing}=\mu
\]

---

### Categorical Features

Missing categorical values are replaced using the **most frequent value (mode)** of the corresponding feature.

---

# Feature Scaling

The numerical features are standardized using **StandardScaler** to ensure that all features contribute equally during model training.

Instead of using raw feature values, each numerical feature is transformed into a standard score.

## Formula

\[
z=\frac{x-\mu}{\sigma}
\]

where

- **x** = Original feature value
- **μ** = Mean of the feature
- **σ** = Standard deviation of the feature

After standardization:

- Mean = **0**
- Standard Deviation = **1**

Feature scaling prevents features with larger numerical ranges from dominating those with smaller ranges and often improves the performance and stability of machine learning algorithms.

---

# Machine Learning Model

## Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on **Bayes' Theorem**.

It assumes:

- Features are conditionally independent of one another.
- Continuous features follow a **Gaussian (Normal) distribution**.

---

## Bayes Theorem

\[
P(C|X)=\frac{P(X|C)\times P(C)}{P(X)}
\]

where

- **P(C|X)** → Posterior Probability
- **P(X|C)** → Likelihood
- **P(C)** → Prior Probability
- **P(X)** → Evidence

The classifier predicts the class with the highest posterior probability.

---

## Gaussian Probability Density Function

For every continuous feature, the likelihood is calculated using the Gaussian distribution:

\[
P(x_i|C)=
\frac{1}{\sqrt{2\pi\sigma_C^2}}
\exp\left(
-\frac{(x_i-\mu_C)^2}
{2\sigma_C^2}
\right)
\]

where

- **μ₍C₎** = Mean of the feature for class **C**
- **σ₍C₎** = Standard deviation of the feature for class **C**

---

## Naive Independence Assumption

Instead of computing one complex joint probability, Gaussian Naive Bayes assumes that all features are conditionally independent.

Therefore,

\[
P(X|C)=\prod_{i=1}^{n}P(x_i|C)
\]

This assumption greatly reduces computational complexity, making Gaussian Naive Bayes a fast and efficient algorithm while still delivering competitive performance on structured datasets.

---

# Model Evaluation

The model is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
