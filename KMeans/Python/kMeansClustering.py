# https://github.com/methylDragon/opencv-python-reference/blob/master/04%20OpenCV%20Machine%20Learning%20and%20AI%20Detectors.md
# Source: https://docs.opencv.org/3.4.4/d5/d26/tutorial_py_knn_understanding.html

# help(cv.FUNCTION_YOU_NEED_HELP_WITH)

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

x = np.random.randint(25, 100, 25)
y = np.random.randint(175, 255, 25)

z = np.hstack((x, y))
z = z.reshape((50, 1))
z = np.float32(z)

plt.hist(z, 256, [0, 256]), plt.show()

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags (Just to avoid line break in the code)
flags = cv.KMEANS_RANDOM_CENTERS

# Apply KMeans
compactness, labels, centers = cv.kmeans(z, 2, None, criteria, 10, flags)

A = z[labels == 0]
B = z[labels == 1]

# Now plot 'A' in red, 'B' in blue, 'centers' in yellow
plt.hist(A, 256, [0, 256], color="r")
plt.hist(B, 256, [0, 256], color="b")
plt.hist(centers, 32, [0, 256], color="y")

plt.show()
