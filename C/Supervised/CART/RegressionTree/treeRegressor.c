//
// Created by pstay on 2/24/26.
//

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct TreeNode {
    int is_leaf;             // 1 if this node is a leaf, 0 if it's an internal split
    float value;             // Used only if is_leaf == 1 (the predicted value/class)

    char feature[50];        // Name of feature to split on
    float split;             // Threshold value for the split
    int split_type;          // 0 for numerical, 1 for categorical

    struct TreeNode *left;   // Left child
    struct TreeNode *right;  // Right child
} TreeNode;

typedef struct {
    float num_feature;
    int cat_feature;
    int label;
}data;

typedef struct {
    float gain;
    int split_point;
}split;


TreeNode* leafNode(float value) {
    TreeNode* leaf = (TreeNode*)malloc(sizeof(TreeNode));
    if (leaf == NULL) return NULL;

    leaf->is_leaf = 1;
    leaf->value = value;

    leaf->left = NULL;
    leaf->right = NULL;

    return leaf;
}

TreeNode* treeNode(const char* feature, float split, TreeNode* left, TreeNode* right, int split_type) {
    TreeNode* node = (TreeNode*)malloc(sizeof(TreeNode));
    if (node == NULL) return NULL;

    node->is_leaf = 0;

    strncpy(node->feature, feature, 49);
    node->feature[49] = '\0';

    node->split = split;
    node->split_type = split_type;
    node->left = left;
    node->right = right;

    return node;
}

float variance(float* values, int n) {
    // param: float* values: array of labels of some feature split
    // param: int n: length of the array of features
    // returns: float variance of the values
    float mean = 0.0;

    if (values == NULL || n == 0) return 0.0;

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

int compare(const void *a, const void *b) {
    const float x = *(float*)a;
    const float y = *(float*)b;
    return x - y;
}

float get_feat_value(data d, char* feat) {
    if (strcmp(feat, "num_feature") == 0) return d.num_feature;
    if (strcmp(feat, "cat_feature") == 0) return d.cat_feature;
    return 0.0f;
}

float* split_left(float* values, int n, float split) {
    int lensplit = 0;

    for (int i = 0; i < n; i++) {
        if (values[i] <= split) {
            lensplit++;
        }
    }

    float* left = malloc(lensplit * sizeof(float));
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (values[i] <= split) {
            left[j] = values[i];
            j++;
        }
    }
    return left;

}

float* split_right(float* values, int n, float split) {
    int lensplit = 0;

    for (int i = 0; i < n; i++) {
        if (values[i] > split) {
            lensplit++;
        }
    }

    float* right = malloc(lensplit * sizeof(float));
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (values[i] > split) {
            right[j] = values[i];
            j++;
        }
    }
    return right;

}

float* get_feat_label(data* data, float* values, int n, char feat[50]) {
    // get array of labels for array of values that need to match to features
    int len = 0;

    if (values == NULL || n == 0) return NULL;

    for (int i = 0; i < n; i++) {
        if (get_feat_value(data[i], feat) == values[i]) {
            len++;
        }
    }

    float* labels = malloc(len * sizeof(float));
    int j = 0;
    for (int i = 0; i < n; i++) {
        if (get_feat_value(data[i], feat) == values[i]) {
            labels[j] = data[i].label;
            j++;
        }
    }
    return labels;
}

float* unique(float* values, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i+1; j < n; j++) {
            if (values[i] == values[j]) {
                n--;
                for (int k = j; k < n; k++) {
                    values[k] = values[k+1];
                }
            }
        }
    }
    return values;
}

split splitpoint(data* dataset, int n, char feat[50]) {
    // extract values
    float* values = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        values[i] = get_feat_value(dataset[i], feat);
    }

    qsort(values, n, sizeof(float), compare);

    // only keep unique values
    unique(values, n);

    float best_gain = INFINITY;
    float best_split = 0.0;

    for (int i = 1; i < n; i++) {
        float split_val = values[i];

        // split values
        float* lsplit = split_left(values, n, split_val);
        float* rsplit = split_right(values, n, split_val);

        int l_count = 0; for(int k=0; k<n; k++) if(values[k] <= split_val) l_count++;
        int r_count = n - l_count;

        // map features to labels
        float* left_labels = get_feat_label(dataset, lsplit, l_count, feat);
        float* right_labels = get_feat_label(dataset, rsplit, r_count, feat);

        float gain = variance(left_labels, l_count) + variance(right_labels, r_count);

        if (gain < best_gain) {
            best_gain = gain;
            best_split = split_val;
        }

        free(lsplit);
        free(rsplit);
        free(left_labels);
        free(right_labels);
    }

    free(values);
    split result;
    result.gain = best_gain;
    result.split_point = best_split;

    return result;
}

TreeNode* build_tree(data* dataset, int n, int depth) {
    // Empty dataset
    if (n == 0) return NULL;

    // Check if all labels are the same
    int all_same = 1;
    for (int i = 1; i < n; i++) {
        if (dataset[i].label != dataset[0].label) {
            all_same = 0;
            break;
        }
    }
    if (all_same) {
        return createLeafNode((float)dataset[0].label);
    }


    // find best split
    const split num_split = splitpoint(dataset, n, "num_feature");
    const split cat_split = splitpoint(dataset, n, "cat_feature");

    // find best gain
    int is_num_best = (num_split.gain >= cat_split.gain);
    float best_gain = is_num_best ? num_split.gain : cat_split.gain;
    float best_split_point = is_num_best ? num_split.split_point : cat_split.split_point;
    const char* best_feature = is_num_best ? "num_feature" : "cat_feature";

    // No gain possible
    if (best_gain <= 0.0) {
        return createLeafNode((float)dataset[0].label);
    }

    // partition data
    data* left_data = (data*)malloc(n * sizeof(data));
    data* right_data = (data*)malloc(n * sizeof(data));
    int left_count = 0;
    int right_count = 0;

    for (int i = 0; i < n; i++) {
        int go_left = 0;
        if (is_num_best) {
            go_left = (dataset[i].num_feature <= best_split_point);
        } else {
            go_left = (dataset[i].cat_feature == (int)best_split_point);
        }

        if (go_left) {
            left_data[left_count++] = dataset[i];
        } else {
            right_data[right_count++] = dataset[i];
        }
    }

    // Build Subtrees recursively using the new partitioned sizes
    TreeNode* left = build_tree(left_data, left_count, depth + 1);
    TreeNode* right = build_tree(right_data, right_count, depth + 1);

    free(left_data);
    free(right_data);

    return createInternalNode(best_feature, best_split_point, left, right, is_num_best ? 0 : 1);
}
