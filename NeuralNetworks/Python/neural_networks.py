# https://stackoverflow.com/questions/37500713/opencv-image-recognition-setting-up-ann-mlp
import cv2

import numpy as np

from sklearn.metrics import accuracy_score

# XOR
data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
target = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)

# Prepare data
target = target.astype(np.float32)
data = data.astype(np.float32)

# Create
nnetwork = cv2.ml.ANN_MLP_create()

# Set term criteria
criteria = (
    cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS,
    10000,
    0.00001,
)
nnetwork.setTermCriteria(criteria)

# Set NN params
nnetwork.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
nnetwork.setLayerSizes(np.array([data.shape[1], 4, 1], dtype=np.int))

# Set after the layers are set
# Only ANN_MLP_SIGMOID_SYM is supported
nnetwork.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)

# Train
# train_data = cv2.ml.TrainData_create(data, cv2.ml.ROW_SAMPLE, target)
# nnetwork.train(train_data)
nnetwork.train(data, cv2.ml.ROW_SAMPLE, target)

print(f"Is trained: {nnetwork.isTrained()}")

# Accuracy
for i in range(data.shape[0]):
    print(nnetwork.predict(np.array([data[i]], dtype=np.float32)), target[i])
