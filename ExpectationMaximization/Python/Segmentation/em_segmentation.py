import numpy as np
import cv2 as cv

source = cv.imread("test-example_em.jpg")

# ouput images
meanImg = np.zeros(shape=(source.shape[0], source.shape[1], 3), dtype=np.float32)
fgImg = np.zeros(shape=(source.shape[0], source.shape[1], 3), dtype=np.float32)
bgImg = np.zeros(shape=(source.shape[0], source.shape[1], 3), dtype=np.float32)

# convert the input image to float
floatSource = source.astype(np.float32)

# now convert the float image to column vector
samples = floatSource.reshape(source.shape[0] * source.shape[1], -1)

em = cv.ml.EM_create()

# we need just 2 clusters
em.setClustersNumber(2)

# train the classifier
em.trainEM(samples)

# the two dominating colors
means = em.getMeans()

# the weights of the two dominant colors
weights = em.getWeights()

# we define the foreground as the dominant color with the largest weight
if weights[0, 0] > weights[0, 1]:
    fgId = 0
else:
    fgId = 1

# now classify each of the source pixels
idx = 0
for row in range(source.shape[0]):
    for col in range(source.shape[1]):
        # classify
        result, _ = em.predict(np.array(samples[idx]).reshape(1, 3))
        idx = idx + 1
        # get the according mean (dominant color)
        ps = means[int(result)]
        # set the according mean value to the mean image
        # float images need to be in [0..1] range
        meanImg[row, col] = (ps[0] / 255.0, ps[1] / 255.0, ps[2] / 255.0)

        # set either foreground or background
        if result == fgId:
            fgImg[row, col] = source[row, col] / 255.0
        else:
            bgImg[row, col] = source[row, col] / 255.0

cv.imshow("Means", meanImg)
cv.imshow("Foreground", fgImg)
cv.imshow("Background", bgImg)

cv.waitKey(0)
