# https://github.com/methylDragon/opencv-python-reference/blob/master/04%20OpenCV%20Machine%20Learning%20and%20AI%20Detectors.md
# Source: https://docs.opencv.org/3.4.4/d5/d26/tutorial_py_knn_understanding.html

# help(cv.FUNCTION_YOU_NEED_HELP_WITH)

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("5000_digits.png")

if img is None:
    raise Exception("we need the 5000_digits.png image from samples/data here !")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)

# Now we prepare train_data and test_data.
train = x[:, :50].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)
test = x[:, 50:100].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k, 160)[:, np.newaxis]
test_labels = train_labels.copy()

# You can decide to save the data if you wish
# np.savez('digit_dataset.npz', train=train, test=test, train_labels=train_labels, test_labels=test_labels)

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv.ml.KNearest_create()

knn.train(train, cv.ml.ROW_SAMPLE, train_labels)

# At this point you could potentially pickle the knn classifier

ret, result, neighbours, dist = knn.findNearest(test, k=5)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size

print(f"accuracy: {accuracy}")
