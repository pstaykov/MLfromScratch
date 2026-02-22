import numpy as np
from itertools import combinations

class LeafNode:
    """
    Leaf node of the decision tree, which stores the predicted label for that leaf.
    """
    def __init__(self, label):
        self.label = label

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

def create_leaf_node(label):
    """
    Create a leaf node with the given label.
    """
    return LeafNode(label)

def create_internal_node(feature_idx, split_value, left_child, right_child, split_type="numerical"):
    """
    Create an internal node with the given feature index, split value, and child nodes.
    """
    return InternalNode(feature_idx, split_value, left_child, right_child, split_type)

def gini(s):
    """
    :param s: List of (id, label) pairs representing a subset of the data
    :return: gini impurity for S in data
    """
    # create dictionary of label count
    labels = {}
    for i in s:
        label = i[1]
        labels[label] = labels.get(label, 0) + 1

    # calculate gini impurity
    impurity = 1
    for label in labels:
        prob_of_label = labels[label] / len(s)
        impurity -= prob_of_label ** 2
    return impurity

def gini_two_groups(s1, s2):
    """
    :param s1: List of (id, label) pairs representing the first subset of the data
    :param s2: List of (id, label) pairs representing the second subset of the data
    :return: gini impurity for the two groups (split point)
    """
    return len(s1)/(len(s1)+len(s2)) * gini(s1) + len(s2)/(len(s1)+len(s2)) * gini(s2)

def _is_numeric(column):
    """
    Check whether a column (numpy array) contains numeric data.
    Returns True if all values can be interpreted as numbers.
    """
    try:
        column.astype(float)
        return True
    except (ValueError, TypeError):
        return False


def find_best_split(data):
    """
    Find the best split for the data, supporting both numerical and categorical features.

    For numerical features, candidate splits use thresholds (<=, >).
    For categorical features, candidate splits consider all binary partitions of the
    unique category values.

    :param data: numpy array where the last column contains the labels
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
                gain = gini_two_groups(s1, s2)
                if gain < best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_split = {"value": val, "feature": feature, "type": "numerical"}
        else:
            # Categorical feature: subset-based split
            # Try all binary partitions of the unique values
            values_list = list(feature_values)
            for r in range(1, len(values_list)):
                for left_subset in combinations(values_list, r):
                    left_set = set(left_subset)
                    mask = np.array([v in left_set for v in data[:, feature]])
                    s1 = data[mask]
                    s2 = data[~mask]
                    if len(s1) == 0 or len(s2) == 0:
                        continue
                    gain = gini_two_groups(s1, s2)
                    if gain < best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_split = {"value": frozenset(left_subset), "feature": feature, "type": "categorical"}

    return best_gain, best_feature, best_split


def build_tree(data, current_depth, max_depth):
    """
    Build a decision tree using the CART algorithm for classification.
    :param data: The dataset as a numpy array, where the last column contains the labels
    :param current_depth: The current depth of the tree being built
    :param max_depth: The maximum depth allowed for the tree
    :return: The root node of the decision tree
    """
    # 1. Base Case: If all labels are the same or max depth reached
    labels = list(data[:, -1])
    unique_labels = np.unique(data[:, -1])
    most_common_label = max(unique_labels, key=lambda l: labels.count(l))
    if len(unique_labels) == 1 or current_depth == max_depth:
        return create_leaf_node(most_common_label)  # Return the most common label as leaf node

    # 2. Find the best split using existing functions
    best_gain, best_feature_idx, best_split_info = find_best_split(data)

    # 3. Handle cases where no further split improves impurity
    if best_split_info is None or best_gain == 0:
        return create_leaf_node(most_common_label)

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
        return create_leaf_node(most_common_label)

    # 5. Recursively build the subtrees
    left_child = build_tree(left_data, current_depth + 1, max_depth)
    right_child = build_tree(right_data, current_depth + 1, max_depth)

    return create_internal_node(best_feature_idx, best_split_value, left_child, right_child, split_type)


def predict(sample, node):
    """
    Predict the label for a given sample using the decision tree.
    :param sample: A single data point (numpy array) for which we want to predict
    :param node: The current node in the decision tree (can be an internal node or a leaf node)
    :return: The predicted label for the sample
    """
    # If we reached a leaf, return the stored label
    if isinstance(node, LeafNode):
        return node.label

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
