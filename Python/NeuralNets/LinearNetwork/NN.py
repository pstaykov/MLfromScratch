"""
this was written by me
"""

import numpy as np

def relu(x):
    return max(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


data = np.array([[1.0, 0.5], [1.3, 2.4], [0.8, 1.2], [0.5, 1.7], [1.1, 1.5]])
labels = np.array([a * b for a, b in data])

# 3 neuron hidden layer according to 2 inputs (3 by 2 matrix)
W1 = np.array([[0.2, -0.5], [0.4, -0.9], [0.1, 0.9]])
b1 = np.array([0.5, -0.3, 0.2])

# 1 neuron output layer according to 3 inputs (1 by 3 matrix)
W2 = np.array([0.1, -0.3, 0.5])
b2 = np.array([0.3])

learning_rate = 0.01

for i in range(2000):
    preds = []

    # Accumulate gradients across all samples
    dL_dW1_total = np.zeros_like(W1)
    dL_db1_total = np.zeros_like(b1)
    dL_dW2_total = np.zeros_like(W2)
    dL_db2_total = np.zeros_like(b2)

    for sample, true_label in zip(data, labels):
        # Forward propagation
        z1 = np.dot(W1, sample) + b1
        a1 = np.array([relu(val) for val in z1])
        z2 = np.dot(W2, a1) + b2
        pred = sigmoid(z2)
        preds.append(pred)

        # Backward propagation (assuming MSE loss function)
        dL_dyhat = 2 * (pred - true_label)
        dsigmoid = sigmoid(z2) * (1 - sigmoid(z2))
        delta2 = dL_dyhat * dsigmoid  # scalar

        dL_dW2 = delta2 * a1  # scalar * vector → shape (3,)
        dL_db2 = delta2  # scalar

        dL_da1 = delta2 * W2  # scalar × vector → shape (3,)

        delta1 = dL_da1 * (z1 > 0)  # element-wise, shape (3,)

        dL_dW1 = np.outer(delta1, sample)  # outer product → shape (3, 2)
        dL_db1 = delta1  # shape (3,)

        dL_dW1_total += dL_dW1
        dL_db1_total += dL_db1
        dL_dW2_total += dL_dW2
        dL_db2_total += dL_db2

    preds = np.array(preds)
    loss = np.mean((preds - labels) ** 2)
    print(f"Loss after iteration {i+1}: {loss}")

    # Update weights and biases (averaged over samples)
    n = len(data)
    W1 = W1 - learning_rate * dL_dW1_total / n
    b1 = b1 - learning_rate * dL_db1_total / n
    W2 = W2 - learning_rate * dL_dW2_total / n
    b2 = b2 - learning_rate * dL_db2_total / n