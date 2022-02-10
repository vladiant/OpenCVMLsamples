# https://github.com/methylDragon/opencv-python-reference/blob/master/04%20OpenCV%20Machine%20Learning%20and%20AI%20Detectors.md
# Source: https://docs.opencv.org/3.4.4/d5/d26/tutorial_py_knn_understanding.html

# help(cv.FUNCTION_YOU_NEED_HELP_WITH)

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Feature set containing (x,y) values of 25 known/training data
trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)

# Labels each one either Red or Blue with numbers 0 and 1
responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)

# Take Red families and plot them
red = trainData[responses.ravel() == 0]
plt.scatter(red[:, 0], red[:, 1], 80, "r", "^")

# Take Blue families and plot them
blue = trainData[responses.ravel() == 1]
plt.scatter(blue[:, 0], blue[:, 1], 80, "b", "s")
plt.show()

newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32)
plt.scatter(newcomer[:, 0], newcomer[:, 1], 80, "g", "o")

# HERE WE GO!
knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)

# You can change the algorithm type if you wish
# knn.setAlgorithmType(cv.ml.KNearest_BRUTE_FORCE)
# knn.setAlgorithmType(cv.ml.KNearest_KDTREE)

ret, results, neighbours, dist = knn.findNearest(newcomer, 3)

print(f"result:  {results}\n")  # Index of the label that was assigned
print(f"neighbours:  {neighbours}\n")
print(f"distance:  {dist}\n")

plt.show()
