import numpy as np

def reLu(x):
    return np.maximum(0, x)

inputs = [0, 1, 2, 3, 4]
outputs = [0, 1, 2, 3, 4]

Wxh = np.random.randn(5, 5)
Whh = np.random.randn(5, 5)
Why = np.random.randn(5, 5)
bh = np.random.randn(5)
by = np.random.randn(5)

def forward(x):
    h = np.zeros((5, 1))
    for t in range(len(x)):
        h = reLu(np.dot(Wxh, x[t]) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    return y

print(forward(inputs))
