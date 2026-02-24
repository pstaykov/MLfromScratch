import numpy as np

class LeafNode:
    """
    Leaf node of the decision tree, which stores the predicted value for that leaf.
    """
    def __init__(self, value):
        self.value = value

class InternalNode:
    """
    Internal node of the decision tree, which stores the feature index and split value/categories
    for that node, as well as references to the left and right child nodes.

    For numerical features:
        split_type = "numerical", split_value = threshold
        Left child: feature <= threshold, Right child: feature > threshold

    For categorical features:
        split_type = "categorical", split_value = set of categories for the left branch
        Left child: feature in split_value, Right child: feature not in split_value
    """
    def __init__(self, feature_idx, split_value, left_child, right_child, split_type="numerical"):
        self.feature_idx = feature_idx
        self.split_value = split_value
        self.left_child = left_child
        self.right_child = right_child
        self.split_type = split_type

def create_leaf_node(value):
    """
    Create a leaf node with the given value.
    """
    return LeafNode(value)

def create_internal_node(feature_idx, split_value, left_child, right_child, split_type="numerical"):
    """
    Create an internal node with the given feature index, split value, and child nodes.
    """
    return InternalNode(feature_idx, split_value, left_child, right_child, split_type)


def variance_two_groups(s1, s2):
    """
    calculate the weighted average of variance for two groups S1 and S2, where the last column of each group is the target value.
    """
    n1, n2 = len(s1), len(s2)
    total = n1 + n2

    # Calculate variance for each group (last column is the target value)
    var1 = np.var(s1[:, -1]) if n1 > 0 else 0
    var2 = np.var(s2[:, -1]) if n2 > 0 else 0

    # Return weighted average of variance
    return (n1 / total) * var1 + (n2 / total) * var2

def _is_numeric(column):
    """
    Check if the array dtype is a number type (int, float)
    """
    return np.issubdtype(column.dtype, np.number)


def find_best_split(data):
    """
    Find the best split for the data, supporting both numerical and categorical features.

    For numerical features, candidate splits use thresholds (<=, >).
    For categorical features, candidate splits consider all binary partitions of the
    unique category values.

    :param data: numpy array where the last column contains the values
    :return: (best_gain, best_feature_index, best_split_info)
             best_split_info is a dict with keys:
               - "value": threshold (numerical) or frozenset of left-branch categories (categorical)
               - "feature": feature index
               - "type": "numerical" or "categorical"
    """
    best_gain = float("inf")
    best_feature = -1
    best_split = None

    for feature in range(data.shape[1] - 1):
        feature_values = np.unique(data[:, feature])

        if _is_numeric(data[:, feature]):
            # Numerical feature: threshold-based split
            for val in feature_values:
                s1 = data[data[:, feature] <= val]
                s2 = data[data[:, feature] > val]
                if len(s1) == 0 or len(s2) == 0:
                    continue
                gain = variance_two_groups(s1, s2)
                if gain < best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_split = {"value": val, "feature": feature, "type": "numerical"}
        else:
            # Categorical feature: optimized split for regression
            # Sort categories by their mean target value, then try contiguous splits
            # This reduces complexity from O(2^N) to O(N log N)
            category_means = {}
            for val in feature_values:
                category_means[val] = np.mean(data[data[:, feature] == val, -1])
            sorted_categories = sorted(feature_values, key=lambda v: category_means[v])

            for i in range(1, len(sorted_categories)):
                left_set = set(sorted_categories[:i])
                mask = np.array([v in left_set for v in data[:, feature]])
                s1 = data[mask]
                s2 = data[~mask]
                if len(s1) == 0 or len(s2) == 0:
                    continue
                gain = variance_two_groups(s1, s2)
                if gain < best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_split = {"value": frozenset(sorted_categories[:i]), "feature": feature, "type": "categorical"}

    return best_gain, best_feature, best_split


def build_tree(data, current_depth, max_depth):
    """
    Build a decision tree using the CART algorithm for classification.
    :param data: The dataset as a numpy array, where the last column contains the values
    :param current_depth: The current depth of the tree being built
    :param max_depth: The maximum depth allowed for the tree
    :return: The root node of the decision tree
    """
    # 1. Base Case: If all values are the same or max depth reached
    unique_values = np.unique(data[:, -1])
    if len(unique_values) == 1 or current_depth == max_depth:
        return create_leaf_node(np.mean(data[:, -1]))  # Return the most common value as leaf node

    # 2. Find the best split using existing functions
    best_gain, best_feature_idx, best_split_info = find_best_split(data)

    current_variance = np.var(data[:, -1])

    # 3. Handle cases where no further split improves impurity
    # Stop if no split was found OR if the best split doesn't actually reduce variance
    if best_split_info is None or best_gain >= current_variance:
        return create_leaf_node(np.mean(data[:, -1]))

    best_split_value = best_split_info["value"]
    split_type = best_split_info["type"]

    # 4. Split the data into left and right branches
    if split_type == "numerical":
        left_data = data[data[:, best_feature_idx] <= best_split_value]
        right_data = data[data[:, best_feature_idx] > best_split_value]
    else:  # categorical
        mask = np.array([v in best_split_value for v in data[:, best_feature_idx]])
        left_data = data[mask]
        right_data = data[~mask]

    if len(left_data) == 0 or len(right_data) == 0:
        return create_leaf_node(np.mean(data[:, -1]))

    # 5. Recursively build the subtrees
    left_child = build_tree(left_data, current_depth + 1, max_depth)
    right_child = build_tree(right_data, current_depth + 1, max_depth)

    return create_internal_node(best_feature_idx, best_split_value, left_child, right_child, split_type)


def predict_node(sample, node):
    """
    Predict the value for a given single sample using the decision tree node.
    (Renamed from `predict` to `predict_node` so the class method can be named `predict`.)
    :param sample: A single data point (numpy array) for which we want to predict
    :param node: The current node in the decision tree (can be an internal node or a leaf node)
    :return: The predicted value for the sample
    """
    # If we reached a leaf, return the stored value
    if isinstance(node, LeafNode):
        return node.value

    # Otherwise, traverse down the tree based on the split
    if node.split_type == "categorical":
        if sample[node.feature_idx] in node.split_value:
            return predict_node(sample, node.left_child)
        else:
            return predict_node(sample, node.right_child)
    else:  # numerical
        if sample[node.feature_idx] <= node.split_value:
            return predict_node(sample, node.left_child)
        else:
            return predict_node(sample, node.right_child)


class regressorTree:
    """
    Simple CART regression tree wrapper with fit and predict methods.

    Usage:
        tree = regressorTree(max_depth=5)
        tree.fit(X, y)
        preds = tree.predict(X_new)
    """

    def __init__(self, max_depth=None):
        self.max_depth = int(max_depth) if max_depth is not None else int(1e9)
        self.root = None

    def fit(self, X, y):
        """
        Fit the regression tree to data.
        X: array-like shape (n_samples, n_features)
        y: array-like shape (n_samples,)
        Returns self for chaining.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim != 1:
            y = y.ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        data = np.hstack([X, y.reshape(-1, 1)])
        self.root = build_tree(data, 0, self.max_depth)
        return self

    def predict(self, X):
        """
        Predict target values for X.
        If X is a single sample (1D array), returns a scalar. Otherwise returns a 1D numpy array.
        """
        X = np.asarray(X)
        single = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            single = True

        preds = np.array([predict_node(row, self.root) for row in X])
        return preds[0] if single else preds

