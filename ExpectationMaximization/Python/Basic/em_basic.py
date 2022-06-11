import numpy as np
import cv2 as cv

img = np.zeros(shape=(500, 500, 3), dtype=np.uint8)

colorTab = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
]

np.random.seed(12345)

numCluster = np.random.randint(2, 5)
print(f"number of clusters: {numCluster}")

sampleCount = np.random.randint(5, 1000)
points = np.zeros(shape=(sampleCount, 2), dtype=np.float32)

# Generate random numbers
for k in range(numCluster):
    center_x = np.random.randint(0, img.shape[1])
    center_y = np.random.randint(0, img.shape[0])
    startrow = int(k * sampleCount / numCluster)
    if k == (numCluster - 1):
        endrow = sampleCount
    else:
        endrow = int((k + 1) * sampleCount / numCluster)

    points[startrow:endrow, :] = np.column_stack(
        (
            np.random.normal(
                loc=center_x, scale=0.05 * img.shape[1], size=endrow - startrow
            ),
            np.random.normal(
                loc=center_y, scale=0.05 * img.shape[0], size=endrow - startrow
            ),
        )
    )

np.random.shuffle(points)

# Initialize the EM model
em_model = cv.ml.EM_create()
em_model.setClustersNumber(numCluster)
em_model.setCovarianceMatrixType(cv.ml.EM_COV_MAT_SPHERICAL)
em_model.setTermCriteria((cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 100, 0.1))

# retval, logLikelihoods, labels, probs
_, _, labels, _ = em_model.trainEM(points)

# Process each pixel
for row in range(img.shape[0]):
    for col in range(img.shape[1]):
        sample = np.zeros(shape=(1, 2), dtype=np.float32)
        sample[0, 0] = col
        sample[0, 1] = row
        response, _ = em_model.predict(sample)
        c = colorTab[int(response)]
        img[row, col] = (c[0] * 3 // 4, c[1] * 3 // 4, c[2] * 3 // 4)
        # cv.circle(img, (col, row),1, (c[0] * 3 //4, c[1] * 3 //4, c[2] * 3 //4), -1)

# Plot sample data
for i in range(sampleCount):
    c = colorTab[int(labels[i])]
    cv.circle(img, (int(points[i, 0]), int(points[i, 1])), 1, c, -1)

cv.imshow("GMM-EM Demo", img)
cv.waitKey(0)
