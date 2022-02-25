# https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.05-Classifying-Iris-Species-Using-Logistic-Regression.ipynb

import numpy as np
import cv2

from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics

iris = datasets.load_iris()

print(f"Input data shape: {iris.data.shape}")
print(f"Input data feature names: {iris.feature_names}")
print(f"Target data shape: {iris.target.shape}")
print(f"Target data classes: {np.unique(iris.target)}")

# Set binary classification problem
idx = iris.target != 2
data = iris.data[idx].astype(np.float32)
target = iris.target[idx].astype(np.float32)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    data, target, test_size=0.1, random_state=42
)

print(f"Train data - input_shape:{X_train.shape}, output shape: {y_train.shape}")
print(f"Test data - input_shape:{X_test.shape}, output shape: {y_test.shape}")

# Initialize logistic regression
lr = cv2.ml.LogisticRegression_create()

lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(1)
lr.setIterations(100)

# Train logistic regression
lr.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

# Learned weights
print(f"Learned weights: {lr.get_learnt_thetas()}")

# Train accuracy
ret, y_pred = lr.predict(X_train)
print(f"Train accuracy: {metrics.accuracy_score(y_train, y_pred)}")

# Test accuracy
ret, y_pred = lr.predict(X_test)
print(f"Test accuracy: {metrics.accuracy_score(y_test, y_pred)}")
