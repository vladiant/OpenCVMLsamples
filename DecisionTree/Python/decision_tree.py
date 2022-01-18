import cv2

import numpy as np

import sklearn.model_selection as ms
from sklearn import metrics
from sklearn.datasets import load_breast_cancer

# Load dataset
data, target = load_breast_cancer(return_X_y=True)

data = data.astype(np.float32)

# Separate to train and test
X_train, X_test, y_train, y_test = ms.train_test_split(
    data, target, test_size=0.2, random_state=42
)

# Create
dtree = cv2.ml.DTrees_create()

# Values different than 0 and 1 cause crash!
dtree.setCVFolds(1)
# Default value is INT_MAX - causes crash!
dtree.setMaxDepth(10)

# Train
dtree.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

_, y_pred_train = dtree.predict(X_train)
_, y_pred_test = dtree.predict(X_test)

print(f"train accuracy: {metrics.accuracy_score(y_train, y_pred_train)}")
print(f"test accuracy: {metrics.accuracy_score(y_test, y_pred_test)}")
