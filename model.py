import numpy as np
from gate import MultiplyGate, AddGate
from activations import Tanh, Softmax

class Model:
  def __init__(self, layer_dimensions):
    self.layer_dimensions = layer_dimensions
    self.W = []
    self.b = []
    for i in range(len(layer_dimensions)-1):
      self.W.append(np.random.randn(layer_dimensions[i], layer_dimensions[i+1]))
      self.b.append(np.random.randn(layer_dimensions[i+1]).reshape(1, layer_dimensions[i+1]))

  def calculate_loss(self, X, y):
    mulGate = MultiplyGate()
    addGate = AddGate()
    tanhActivation = Tanh()
    softmaxActivation = Softmax()
    input = X
    for i in range(len(self.W)):
      mul = mulGate.forward(input, self.W[i])
      add = addGate.forward(mul, self.b[i])
      layer = tanhActivation.forward(add)
      input = layer

    softmax_output = softmaxActivation.forward(input)
    num_examples = X.shape[0]
    probs = softmax_output
    correct_probs = probs[range(num_examples), y]
    correct_log_probs = -np.log(correct_probs)
    loss = np.sum(correct_log_probs)
    return (1./num_examples) * loss
  
  def predict(self, X):
    mulGate = MultiplyGate()
    addGate = AddGate()
    tanhActivation = Tanh()
    softmaxActivation = Softmax()
    input = X
    for i in range(len(self.W)):
      mul = mulGate.forward(input, self.W[i])
      add = addGate.forward(mul, self.b[i])
      layer = tanhActivation.forward(add)
      input = layer

    softmax_output = softmaxActivation.forward(input)
    probs = softmax_output
    return np.argmax(probs, axis=1)
    
  def train(self, X, y, learning_rate=0.01, regularization=0.01, epoch_count=20000, print_loss = False):
    mulGate = MultiplyGate()
    addGate = AddGate()
    tanhActivation = Tanh()
    softmaxActivation = Softmax()
    for epoch in range(epoch_count):
      # forward pass
      input = X
      forward = [(None, None, input)]
      for i in range(len(self.W)):
        mul = mulGate.forward(input, self.W[i])
        add = addGate.forward(mul, self.b[i])
        layer = tanhActivation.forward(add)
        forward.append((mul, add, layer))
        input = layer

      softmax_output = softmaxActivation.forward(forward[-1][2])
      dtanh = softmaxActivation.backward(softmax_output, y)

      # backward pass
      for i in range(len(forward)-1, 0, -1):
        dadd = tanhActivation.backward(forward[i][1], dtanh)
        dmul, db = addGate.backward(dadd)
        dw, dtanh = mulGate.backward(forward[i-1][2], self.W[i-1], dmul)
        
        dw += regularization * self.W[i-1]
        db += regularization * self.b[i-1]

        self.W[i-1] -= learning_rate * dw
        self.b[i-1] -= learning_rate * db

      if print_loss and epoch % 1000 == 0:
        print(f'Loss after iteration {epoch}: {self.calculate_loss(X, y)}')




  