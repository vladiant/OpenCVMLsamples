import cv2

import numpy as np

trainingData = np.array(
    [
        [40, 55],
        [35, 35],
        [55, 15],
        [45, 25],
        [10, 10],
        [15, 15],
        [40, 10],
        [30, 15],
        [30, 50],
        [100, 20],
        [45, 65],
        [20, 35],
        [80, 20],
        [90, 5],
        [95, 35],
        [80, 65],
        [15, 55],
        [25, 65],
        [85, 35],
        [85, 55],
        [95, 70],
        [105, 50],
        [115, 65],
        [110, 25],
        [120, 45],
        [15, 45],
        [55, 30],
        [60, 65],
        [95, 60],
        [25, 40],
        [75, 45],
        [105, 35],
        [65, 10],
        [50, 50],
        [40, 35],
        [70, 55],
        [80, 30],
        [95, 45],
        [60, 20],
        [70, 30],
        [65, 45],
        [85, 40],
    ],
    dtype=np.float32,
)

responses = [
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "R",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
]

responsesMat = np.array([ord(i) for i in responses], dtype=np.int32)

priors = np.array([1, 1], dtype=np.float32)

boost = cv2.ml.Boost_create()

boost.setBoostType(cv2.ml.Boost_REAL)
boost.setWeakCount(10)
boost.setWeightTrimRate(0.95)
boost.setUseSurrogates(False)
boost.setPriors(priors)

boost.train(trainingData, cv2.ml.ROW_SAMPLE, responsesMat)

myData = np.array([55, 25], dtype=np.float32)

r, _ = boost.predict(myData)

print(f"Result: {chr(int(r))}")
