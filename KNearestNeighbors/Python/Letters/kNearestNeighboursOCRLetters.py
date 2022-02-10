# https://github.com/methylDragon/opencv-python-reference/blob/master/04%20OpenCV%20Machine%20Learning%20and%20AI%20Detectors.md
# Source: https://docs.opencv.org/3.4.4/d5/d26/tutorial_py_knn_understanding.html

# help(cv.FUNCTION_YOU_NEED_HELP_WITH)

import cv2 as cv
import numpy as np

# Load the data, converters convert the letter to a number
data = np.loadtxt(
    "letter-recognition.data.txt",
    dtype="float32",
    delimiter=",",
    converters={0: lambda ch: ord(ch) - ord("A")},
)

# split the data to two, 10000 each for train and test
train, test = np.vsplit(data, 2)

# split trainData and testData to features and responses
responses, trainData = np.hsplit(train, [1])
labels, testData = np.hsplit(test, [1])

# Initiate the kNN, classify, measure accuracy.
knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)

ret, result, neighbours, dist = knn.findNearest(testData, k=5)
correct = np.count_nonzero(result == labels)

accuracy = correct * 100.0 / 10000
print(f"accuracy: {accuracy}")

