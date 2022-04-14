# https://notebook.community/mbeyeler/opencv-machine-learning/notebooks/07.01-Implementing-Our-First-Bayesian-Classifier

from sklearn import datasets, model_selection, metrics
import numpy as np
import cv2
import matplotlib.pyplot as plt

# plt.style.use("ggplot")


def plot_decision_boundary(model, X_test, y_test):
    h = 0.02
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    X_hypo = np.column_stack(
        (xx.ravel().astype(np.float32), yy.ravel().astype(np.float32))
    )
    ret = model.predict(X_hypo)

    if isinstance(ret, tuple):
        zz = ret[1]
    else:
        zz = ret
    zz = zz.reshape(xx.shape)

    plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=200)


X, y = datasets.make_blobs(100, 2, centers=2, random_state=1701, cluster_std=2)

# plt.figure(figsize=(10,6))
# plt.scatter(X[:,0], X[:,1], c=y, s=50)
# plt.show()

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X.astype(np.float32), y, test_size=0.1
)

model_norm = cv2.ml.NormalBayesClassifier_create()

model_norm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

_, y_pred = model_norm.predict(X_test)

print("Accuracy score:", metrics.accuracy_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
plot_decision_boundary(model_norm, X, y)
plt.show()
