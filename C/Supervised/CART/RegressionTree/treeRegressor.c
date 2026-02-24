//
// Created by pstay on 2/24/26.
//

#include <stdlib.h>

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




