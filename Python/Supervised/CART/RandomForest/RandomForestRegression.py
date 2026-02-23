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


def find_best_split(data, n_features=None):
    """
    Find the best split for the data among a RANDOM subset of features.

    :param data: numpy array where the last column contains the target values
    :param n_features: Number of random features to consider (The 'm' in Random Forest)
    """
    best_gain = float("inf")
    best_feature = -1
    best_split = None

    # Get all available feature indices
    all_indices = np.arange(data.shape[1] - 1)

    # Select random subset of features to evaluate
    if n_features is not None and n_features < len(all_indices):
        features_to_check = np.random.choice(all_indices, size=n_features, replace=False)
    else:
        features_to_check = all_indices

    # Only loop through the randomly selected features
    for feature in features_to_check:
        feature_values = np.unique(data[:, feature])

        if _is_numeric(data[:, feature]):
            # Numerical feature: threshold-based split
            for val in feature_values:
                mask = data[:, feature] <= val
                s1 = data[mask]
                s2 = data[~mask]

                if len(s1) == 0 or len(s2) == 0:
                    continue

                gain = variance_two_groups(s1, s2)
                if gain < best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_split = {"value": val, "feature": feature, "type": "numerical"}
        else:
            # Categorical feature logic
            category_means = {val: np.mean(data[data[:, feature] == val, -1]) for val in feature_values}
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
                    best_split = {"value": frozenset(left_set), "feature": feature, "type": "categorical"}

    return best_gain, best_feature, best_split

def build_tree(data, current_depth, max_depth, n_features):
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
    best_gain, best_feature_idx, best_split_info = find_best_split(data, n_features)

    # 3. If no valid split is found or the gain is not better than current variance, create a leaf node
    current_variance = np.var(data[:, -1])
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
    left_child = build_tree(left_data, current_depth + 1, max_depth, n_features)
    right_child = build_tree(right_data, current_depth + 1, max_depth, n_features)

    return create_internal_node(best_feature_idx, best_split_value, left_child, right_child, split_type)


def predict(sample, node):
    """
    Predict the value for a given sample using the decision tree.
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
            return predict(sample, node.left_child)
        else:
            return predict(sample, node.right_child)
    else:  # numerical
        if sample[node.feature_idx] <= node.split_value:
            return predict(sample, node.left_child)
        else:
            return predict(sample, node.right_child)

def bootstrap_samples(data, n_samples):
    """
    Generate bootstrap samples from the original dataset.
    :param data: The original dataset as a numpy array
    :param n_samples: The number of samples to generate for each bootstrap sample
    :return: A list of bootstrap samples
    """
    bootstrap_samples = []
    for _ in range(n_samples):
        indices = np.random.choice(len(data), size=len(data), replace=True)
        bootstrap_samples.append(data[indices])
    return bootstrap_samples


def build_random_forest(data, n_trees, max_depth):
    """
    Build a random forest by creating multiple decision trees using bootstrap samples of the data and random feature selection.
    :param data: The original dataset as a numpy array, where the last column contains the target values
    :param n_trees: The number of trees to build in the random forest
    :param max_depth: The maximum depth for each decision tree
    :return: A list of decision trees that make up the random forest
    """
    trees = []
    # Calculate the number of features to consider once
    n_total_features = data.shape[1] - 1
    n_features = int(np.sqrt(n_total_features))

    # Get all bootstrap samples
    samples = bootstrap_samples(data, n_trees)

    for sample in samples:
        # NOTICE: We do NOT slice the sample columns here anymore.
        # We pass the full sample and let build_tree handle the randomness.
        tree = build_tree(sample, current_depth=0, max_depth=max_depth, n_features=n_features)
        trees.append(tree)

    return trees

def predict_forest(sample, forest):
    """
    Get a prediction from every tree and return the average.
    """
    predictions = [predict(sample, tree) for tree in forest]
    return np.mean(predictions)