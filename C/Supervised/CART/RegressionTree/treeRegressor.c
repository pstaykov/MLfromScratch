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
    float num_feature;
    int cat_feature;
    int label;
}data;

typedef struct {
    int gain;
    int split_point;
}split;


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

split splitpoint(data* dataset, int n, char feat[50]) {
    // extract values
    float* values = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        values[i] = get_feat_value(dataset[i], feat);
    }

    qsort(values, n, sizeof(float), compare);

    // only keep unique values
    for (int i = 1; i < n; i++) {
        if (values[i] == values[i-1]) {
            n--;
            for (int j = i; j < n; j++) {
                values[j] = values[j+1];
            }
        }
    }

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


