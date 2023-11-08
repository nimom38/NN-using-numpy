import numpy as np

class MultiplyGate:
  def forward(self, X, W):
    return np.dot(X, W)
  
  def backward(self, X, W, top_diff):
    dw = np.dot(np.transpose(X), top_diff)
    dtanh = np.dot(top_diff, np.transpose(W))
    return dw, dtanh
  

class AddGate:
  def forward(self, X, b):
    return X+b
  
  def backward(self, top_diff):
    dmul = top_diff
    db = np.dot(np.ones((1, top_diff.shape[0])), top_diff)

    return dmul, db
  