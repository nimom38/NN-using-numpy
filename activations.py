import numpy as np

class Tanh:
  def forward(self, X):
    return np.tanh(X)
  
  def backward(self, X, top_diff):
    return (1 - np.square(np.tanh(X))) * top_diff


  
class Softmax:
  def forward(self, X):
    return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
  
  def backward(self, X, y):
    num_examples = X.shape[0]
    X[range(num_examples), y] -= 1
    return X