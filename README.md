# Logistic-Regression-on-Students-Performance-Dataset-PyTorch-
A PyTorch implementation of logistic regression on the Students Performance dataset. The project includes data preprocessing, feature encoding, model training, and evaluation to classify student outcomes based on demographic, academic, and socio-economic factors.
This project implements a logistic regression model in PyTorch to analyze and classify student performance using the StudentsPerformance.csv dataset. The goal is to understand how demographic, academic, and socio-economic factors affect student outcomes.

Project Overview

Dataset: Student exam scores with features such as gender, parental education, lunch type, and test preparation course.

Target: Student performance classification (e.g., pass/fail or high/low performance).

Approach: Logistic regression model trained with PyTorch.

Workflow

Data Preprocessing

Converted categorical variables into numerical form with one-hot encoding.

Normalized numerical features.

Labeled the target variable into classes for classification.

Modeling

Logistic regression implemented using PyTorch tensors and nn.Module.

Loss function: Binary Cross Entropy / CrossEntropyLoss (depending on task).

Optimizer: Stochastic Gradient Descent (SGD) or Adam.

Evaluation

Accuracy, precision, recall, and F1-score.

Confusion matrix to analyze prediction results.

Visualization of loss curve and decision boundaries (if applicable).

Results & Insights

Test preparation course completion was a strong indicator of improved performance.

Socio-economic features such as lunch type and parental education level influenced student success.

Logistic regression provided a simple but interpretable baseline model for classification.

Tech Stack

Python: Pandas, NumPy, Matplotlib, Scikit-learn

PyTorch: Logistic regression model training and evaluation

Future Improvements

Compare logistic regression with more complex models (Random Forest, Neural Networks).

Apply regularization (L1/L2) to prevent overfitting.

Extend analysis to multi-class classification for subject-specific performance.
