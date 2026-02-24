import numpy as np

from Python.Supervised.CART.RegressionTree.CART_Regression import regressorTree

X = np.array([1, 2, 3, 4, 5])
y = np.array([10, 20, 30, 40, 50])

n_trees = 100
learning_rate = 0.1

# Initialize with the mean
f_0 = np.mean(y)

# This list will store "weak learners"
trees = []


# Initialize predictions once
y_pred = np.full(len(X), f_0)

for i in range(n_trees):
    residuals = y - y_pred

    # Fit new tree
    tree = regressorTree(max_depth=3)
    tree.fit(X.reshape(-1, 1), residuals)
    trees.append(tree)

    # Update running predictions for the next iteration
    y_pred += learning_rate * tree.predict(X.reshape(-1, 1))

    # Track loss reduction
    loss = np.mean(0.5 * (y - y_pred) ** 2)
    if i % 10 == 0:
        print(f"Tree {i} | MSE: {loss:.4f}")