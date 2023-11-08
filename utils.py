import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(pred_func, X, y):
  h = 0.01
  x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
  y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

  predictions = pred_func(np.c_[xx.ravel(), yy.ravel()])
  z = predictions.reshape(xx.shape)
  plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
  plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
