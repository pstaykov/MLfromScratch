"""
this was written by me
This has hardcoded 2 input neurons but the number of hidden neurons can be changed. The output neuron is also hardcoded.
"""

import numpy as np

class Relu:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dL_dy):
        return dL_dy * (self.x > 0) # chain Rule. Multiply with previous layer's gradient'

class Sigmoid:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = 1 / (1 + np.exp(-x))
        return self.x

    def backward(self, dL_dy):
        return dL_dy * self.x * (1 - self.x)

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.b = float(np.zeros_like(output_dim))

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dL_dy):
        self.dL_dW = np.outer(self.x, dL_dy)
        self.dL_db = dL_dy
        return np.dot(dL_dy, self.W.T)

    def update(self, lr):
        self.W -= lr * self.dL_dW
        self.b -= lr * self.dL_db

data = np.array([[1.0, 0.5], [1.3, 2.4], [0.8, 1.2], [0.5, 1.7], [1.1, 1.5]])
labels = np.array([a * b for a, b in data])

class NeuralNetwork():
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.hidden_layer = LinearLayer(n_inputs, n_hidden)
        self.relu = Relu()
        self.output_layer = LinearLayer(n_hidden, n_outputs)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        hidden_output = self.hidden_layer.forward(x)
        relu_hidden_output = self.relu.forward(hidden_output)
        output = self.output_layer.forward(relu_hidden_output)
        sigmoid_output = self.sigmoid.forward(output)
        return output

    def backward(self, dL_dyhat):
        grad = self.sigmoid.backward(dL_dyhat)
        grad = self.output_layer.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.hidden_layer.backward(grad)
        return grad

    def update(self, lr):
        self.hidden_layer.update(lr)
        self.output_layer.update(lr)

data   = np.array([[1.0, 0.5], [1.3, 2.4], [0.8, 1.2], [0.5, 1.7], [1.1, 1.5]])
labels = np.array([a * b for a, b in data])

net = NeuralNetwork(n_inputs=2, n_hidden=4, n_outputs=1)
lr  = 0.01

for epoch in range(300):
    total_loss = 0

    for x, y in zip(data, labels):
        pred = net.forward(x).squeeze()

        loss = (pred - y) ** 2
        dL_dyhat = 2 * (pred - y)
        total_loss += loss

        net.backward(dL_dyhat)
        net.update(lr)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(data):.4f}")