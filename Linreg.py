import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 10, 10)
y = X + np.random.randint(-3, 3, size=X.shape)

plt.scatter(X, y)

weight = 0.0
bias = 0.0
epochs = 1000
lr = 0.01
n = len(X)

# Vectorized batch gradient descent
for epoch in range(epochs):
    y_pred = weight * X + bias
    # Mean squared error gradients
    dw = (2 / n) * np.sum((y_pred - y) * X)
    db = (2 / n) * np.sum(y_pred - y)
    weight -= lr * dw
    bias -= lr * db

# Final predictions and plot
y_pred = weight * X + bias
plt.plot(X, y_pred, color="red")
plt.title(f"Linear regression: weight={weight:.3f}, bias={bias:.3f}, MSE = {np.mean((y_pred - y) ** 2):.3f}")
plt.show()
