# https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.03-Using-Random-Forests-for-Face-Recognition.ipynb

import cv2

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_olivetti_faces

# Olivetti face dataset
# http://cs.nyu.edu/~roweis/data/olivettifaces.mat
dataset = fetch_olivetti_faces()

X = dataset.data
y = dataset.target

# Same mean grayscale level
n_samples, n_features = X.shape
X -= X.mean(axis=0)

# Feature values of every data point (that is, a row in X) are centered around zero
X -= X.mean(axis=1).reshape(n_samples, -1)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=21)

# Initialize random forest
rtree = cv2.ml.RTrees_create()

# Initialize random forest parameters
num_trees = 50
eps = 0.01
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, num_trees, eps)

rtree.setTermCriteria(criteria)
rtree.setMaxCategories(len(np.unique(y)))
rtree.setMaxDepth(1000)

# Train model
rtree.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

print(f"Trained model depth: {rtree.getMaxDepth()}")

# Accuracy
_, y_hat_train = rtree.predict(X_train)
_, y_hat_test = rtree.predict(X_test)

print(f"Train accuracy: {accuracy_score(y_train, y_hat_train)}")
print(f"Test accuracy: {accuracy_score(y_test, y_hat_test)}")
