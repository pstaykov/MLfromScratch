//
// Created by pstay on 2/24/26.
//

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float value;
}LeafNode;

typedef struct TreeNode {
    int feature; // index of feature
    float split; // splitpoint
    struct TreeNode *left;
    struct TreeNode *right;
    int split_type; // 0 for numerical, 1 for categorical
}TreeNode;

LeafNode* leafNode(float value) {
    LeafNode* leaf = malloc(sizeof(LeafNode));
    leaf->value = value;
    return leaf;
}

typedef struct {
    int num_feature;
    int cat_feature;
    int label;
}data;


TreeNode* treeNode(int feature, float split, TreeNode* left, TreeNode* right, int split_type) {
    TreeNode* node = malloc(sizeof(TreeNode));
    node->feature = feature;
    node->split = split;
    node->left = left;
    node->right = right;
    node->split_type = split_type;
    return node;
}

float variance(float* values, int n) {
    // param: float* values: array of values of some feature
    // param: int n: length of the array of features
    // returns: float variance of the values
    float mean = 0.0;
    for (int i = 0; i < n; i++) {
        mean += values[i];
    }
    mean /= n;

    float var = 0.0;
    for (int i = 0; i < n; i++) {
        var += (values[i] - mean) * (values[i] - mean); // squared error
    }
    return var / n;
}

void sort_floats(float* values, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (values[i] > values[j]) {
                float temp = values[i];
                values[i] = values[j];
                values[j] = temp;
            }
        }
    }
}

float get_feat_value(data d, char* feat) {
    if (strcmp(feat, "num_feature") == 0) return d.price;
    if (strcmp(feat, "cat_feature") == 0) return d.weight;
    return 0.0f;
}

float numerical_splitpoint(data* dataset, int n, char feat[50]) {
    float* values = malloc(n * sizeof(float));

    // Extract values
    for (int i = 0; i < n; i++) {
        values[i] = get_feat_value(dataset[i], feat);
    }

    // sort values


    float best_gain = INFINITY;
    float best_split = 0.0;

    for (int i = 1; i < n; i++) {
        float gain = variance(values, i) + variance(&values[i], n - i);

        if (gain < best_gain) {
            best_gain = gain;
            best_split = values[i];
        }
    }

    free(values);
    return best_split;
}

