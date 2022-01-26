# https://github.com/methylDragon/opencv-python-reference/blob/master/04%20OpenCV%20Machine%20Learning%20and%20AI%20Detectors.md
# Source: https://docs.opencv.org/3.4.4/d5/d26/tutorial_py_knn_understanding.html

# help(cv.FUNCTION_YOU_NEED_HELP_WITH)

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

X = np.random.randint(25, 50, (25, 2))
Y = np.random.randint(60, 85, (25, 2))
Z = np.vstack((X, Y))

# convert to np.float32
Z = np.float32(Z)

# define criteria and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv.kmeans(Z, 2, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# Now separate the data, Note the flatten()
A = Z[label.ravel() == 0]
B = Z[label.ravel() == 1]

# Plot the data
plt.scatter(A[:, 0], A[:, 1])
plt.scatter(B[:, 0], B[:, 1], c="r")
plt.scatter(center[:, 0], center[:, 1], s=80, c="y", marker="s")
plt.xlabel("Height"), plt.ylabel("Weight")

plt.show()
