import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt 
from model import Model
from utils import plot_decision_boundary

np.random.seed(0)

X, y = make_moons(200, noise=0.2)

plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
plt.title('Training Data')
plt.show()

layer_dimensions = [2, 3, 2]

model = Model(layer_dimensions)
model.train(X, y, print_loss=True)

plot_decision_boundary(lambda x: model.predict(x), X, y)
plt.title('Decision Boundary for hidden layer size 3')
plt.show()
