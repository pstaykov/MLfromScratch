import numpy as np

def gini(S):
    """
    :param S: List of (id, label) pairs representing a subset of the data
    :return: gini impurity for S in data
    """
    # create dictionary of label -> count
    labels = {}
    for i in S:
        label = i[1]
        labels[label] = labels.get(label, 0) + 1

    # calculate gini impurity
    gini = 1
    for label in labels:
        prob_of_label = labels[label] / len(S)
        gini -= prob_of_label ** 2
    return gini

def gini_two_groups(S1, S2):
    """
    :param S1: List of (id, label) pairs representing the first subset of the data
    :param S2: List of (id, label) pairs representing the second subset of the data
    :return: gini impurity for the two groups (split point)
    """
    return len(S1)/(len(S1)+len(S2)) * gini(S1) + len(S2)/(len(S1)+len(S2)) * gini(S2)

def find_best_split(data):
    '''
    find the best split for a given column(feature) of data
    :param data: Dataframe
    :return: best gain, best feature index, best split point (value, feature index)
    '''
    best_gain = 0
    best_feature = -1
    best_split = None
    for feature in range(data.shape[1]-1):
        feature_values = np.unique(data[:, feature])
        for val in feature_values:
            S1 = data[data[:, feature] <= val]
            S2 = data[data[:, feature] > val]
            gain = gini_two_groups(S1, S2)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_split = (val, feature)

    return best_gain, best_feature, best_split








