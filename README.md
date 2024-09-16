# Spam Detection : Logistic Regression vs. Random Forest

# Overview

This program is designed to predict whether an email is spam or not using two machine learning models: Logistic Regression and Random Forest Classifier. We will compare the performance of these models on a spam dataset from the UCI Machine Learning Library using the accuracy score as the metric.

#Key Steps:
Retrieve the data.
Predict which model (Logistic Regression or Random Forest) will perform better.
Preprocess the data (split and scale).
Train and evaluate both models.
Compare and analyze the results.
Dataset

The dataset contains various features derived from emails and includes a target variable (spam), which indicates whether an email is spam (1) or not spam (0).

# Dataset Source:
Spam Data CSV
Original Source: UCI Machine Learning Library
Program Requirements

# Dependencies:
pandas
scikit-learn
train_test_split
StandardScaler
LogisticRegression
RandomForestClassifier
accuracy_score
You can install the required libraries using the following commands:

```python
pip install pandas scikit-learn
```
# Instructions

1. Retrieve and Load the Data
We use pandas to load the dataset from the provided CSV file. The dataset includes various features that will be used to predict whether an email is spam or not.

```python
import pandas as pd
data = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv")
```
2. Split the Data
The dataset is split into two parts:

X (features) – all columns except the target (spam)
y (target) – the spam column
We then split the data into training and test sets to evaluate model performance on unseen data.

  ```python
from sklearn.model_selection import train_test_split

X = data.drop('spam', axis=1)
y = data['spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

```
3. Scale the Data
Standardization is applied using StandardScaler to ensure that features have a mean of 0 and a standard deviation of 1. We fit the scaler on the training data and use the same fit to transform the test data.

 ```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

```
4. Train the Models

### Logistic Regression

We create a logistic regression model, train it on the scaled training data, and evaluate its accuracy on the test data.

 ```python

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logistic_regression_model = LogisticRegression(random_state=1, max_iter=300)
lr_model = logistic_regression_model.fit(X_train_scaled, y_train)

# Model Accuracy
lr_train_score = lr_model.score(X_train_scaled, y_train)
lr_test_score = accuracy_score(y_test, lr_model.predict(X_test_scaled))
```

### Random Forest Classifier

We also create a random forest classifier, train it, and evaluate its accuracy on the test data.

 ```python

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=160, random_state=1)
rf_model.fit(X_train_scaled, y_train)


# Model Accuracy
rf_test_score = accuracy_score(y_test, rf_model.predict(X_test_scaled))

```
5. Compare Results
After training both models, we compare the accuracy scores of logistic regression and random forest on the test set.

 ```python

print(f"Logistic Regression Test Accuracy: {lr_test_score}")
print(f"Random Forest Test Accuracy: {rf_test_score}")

```

# Expected Results
Based on prior knowledge:

Random Forest is expected to outperform Logistic Regression due to its ability to handle non-linear interactions and its ensemble nature.

# Conclusion
You will record which model performed better and reflect on whether it matched your initial prediction. The expected output should show that Random Forest performs better, with higher accuracy than Logistic Regression.

# License

This project is licensed under the MIT License.

