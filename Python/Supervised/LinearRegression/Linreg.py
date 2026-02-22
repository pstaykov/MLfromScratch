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

# L2 regularization parameter
lambda_reg = 0.1  # set to 0.0 to disable L2
save_fig = True

# Vectorized batch gradient descent with L2 regularization
for epoch in range(epochs):
    y_pred = weight * X + bias
    # Mean squared error gradients
    dw = (2 / n) * np.sum((y_pred - y) * X) + lambda_reg * weight
    db = (2 / n) * np.sum(y_pred - y)
    weight -= lr * dw
    bias -= lr * db

# Final predictions and plot
y_pred = weight * X + bias
plt.plot(X, y_pred, color="red")
mse = np.mean((y_pred - y) ** 2)
plt.title(f"Linear regression: weight={weight:.3f}, bias={bias:.3f}, MSE = {mse:.3f}, lambda={lambda_reg}")
if save_fig:
    out = 'linreg.png'
    plt.savefig(out)
    print(f"Saved plot to {out}")
else:
    plt.show()
