//
// Created by pstay on 2/24/26.
//

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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
} data;

typedef struct {
    float gain;
    float split_point;
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

float get_feat_value(data d, const char* feat) {
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

float* get_feat_label(data* data, float* values, int n, const char* feat) {
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

int unique(float* values, int n) {
    int unique_count = n;
    for (int i = 0; i < unique_count; i++) {
        for (int j = i+1; j < unique_count; j++) {
            if (values[i] == values[j]) {
                unique_count--;
                for (int k = j; k < unique_count; k++) {
                    values[k] = values[k+1];
                }
                j--; // Check the same position again since we shifted
            }
        }
    }
    return unique_count;
}

split splitpoint(data* dataset, int n, const char* feat) {
    // extract values
    float* values = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        values[i] = get_feat_value(dataset[i], feat);
    }

    qsort(values, n, sizeof(float), compare);

    // only keep unique values
    int unique_count = unique(values, n);

    float best_gain = INFINITY;
    float best_split = 0.0;

    for (int i = 1; i < unique_count; i++) {
        float split_val = values[i];

        // split values
        float* lsplit = split_left(values, unique_count, split_val);
        float* rsplit = split_right(values, unique_count, split_val);

        int l_count = 0;
        for(int k=0; k<unique_count; k++) {
            if(values[k] <= split_val) l_count++;
        }
        int r_count = unique_count - l_count;

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
        return leafNode((float)dataset[0].label);
    }

    // find best split
    const split num_split = splitpoint(dataset, n, "num_feature");
    const split cat_split = splitpoint(dataset, n, "cat_feature");

    // find best gain (choose the one with lower variance/gain)
    int is_num_best = (num_split.gain <= cat_split.gain);
    float best_gain = is_num_best ? num_split.gain : cat_split.gain;
    float best_split_point = is_num_best ? num_split.split_point : cat_split.split_point;
    const char* best_feature = is_num_best ? "num_feature" : "cat_feature";

    // No valid split found (gain is still INFINITY) - create leaf with mean value
    if (best_gain >= INFINITY) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += dataset[i].label;
        }
        return leafNode(sum / n);
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

    // If one side is empty, create a leaf node with the mean
    if (left_count == 0 || right_count == 0) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += dataset[i].label;
        }
        free(left_data);
        free(right_data);
        return leafNode(sum / n);
    }

    // Build Subtrees recursively using the new partitioned sizes
    TreeNode* left = build_tree(left_data, left_count, depth + 1);
    TreeNode* right = build_tree(right_data, right_count, depth + 1);

    free(left_data);
    free(right_data);

    return treeNode(best_feature, best_split_point, left, right, is_num_best ? 0 : 1);
}

float predict(TreeNode* node, data d) {
    if (node->is_leaf) {
        return node->value;
    }

    float feat_value = get_feat_value(d, node->feature);
    int go_left = 0;

    if (node->split_type == 0) { // numerical
        go_left = (feat_value <= node->split);
    } else { // categorical
        go_left = (feat_value == node->split);
    }

    if (go_left) {
        return predict(node->left, d);
    } else {
        return predict(node->right, d);
    }
}

void free_tree(TreeNode* node) {
    if (node == NULL) return;

    if (!node->is_leaf) {
        free_tree(node->left);
        free_tree(node->right);
    }

    free(node);
}

data* parse_csv(const char* filename, int* count) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file '%s'\n", filename);
        return NULL;
    }

    // Read header line
    char line[256];
    if (fgets(line, sizeof(line), file) == NULL) {
        fprintf(stderr, "Error: Empty file\n");
        fclose(file);
        return NULL;
    }

    // Count number of data rows
    int num_rows = 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        num_rows++;
    }

    // Allocate memory for data
    data* dataset = (data*)malloc(num_rows * sizeof(data));
    if (dataset == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    // Reset file pointer to read data
    rewind(file);
    fgets(line, sizeof(line), file); // Skip header

    // Read data rows
    int i = 0;
    while (fgets(line, sizeof(line), file) != NULL && i < num_rows) {
        // Check if this is categorical data (3 columns) or numerical data (2 columns)
        int num_commas = 0;
        for (char* p = line; *p; p++) {
            if (*p == ',') num_commas++;
        }

        if (num_commas == 2) {
            // Categorical data: num_feature,cat_feature,label
            if (sscanf(line, "%f,%d,%d", &dataset[i].num_feature, &dataset[i].cat_feature, &dataset[i].label) == 3) {
                i++;
            }
        } else if (num_commas == 1) {
            // Numerical data: num_feature,label
            if (sscanf(line, "%f,%d", &dataset[i].num_feature, &dataset[i].label) == 2) {
                dataset[i].cat_feature = 0; // Default value for missing categorical feature
                i++;
            }
        }
    }

    fclose(file);
    *count = i;
    return dataset;
}

void free_data(data* dataset) {
    free(dataset);
}

int main() {
    // Parse CSV file
    int count = 0;
    data* dataset = parse_csv("data_categorical.csv", &count);

    if (dataset == NULL) {
        fprintf(stderr, "Failed to parse CSV file\n");
        return 1;
    }

    printf("Successfully parsed %d rows from CSV\n", count);

    // Display first few rows
    printf("\nFirst 5 rows of parsed data:\n");
    printf("num_feature, cat_feature, label\n");
    for (int i = 0; i < 5 && i < count; i++) {
        printf("%.2f, %d, %d\n", dataset[i].num_feature, dataset[i].cat_feature, dataset[i].label);
    }

    // Build regression tree
    printf("\nBuilding regression tree...\n");
    TreeNode* tree = build_tree(dataset, count, 0);

    if (tree == NULL) {
        fprintf(stderr, "Failed to build tree\n");
        free_data(dataset);
        return 1;
    }

    printf("Tree built successfully!\n");

    // Test predictions on first 10 samples
    printf("\nTesting predictions on first 10 samples:\n");
    printf("Actual -> Predicted\n");
    for (int i = 0; i < 10 && i < count; i++) {
        float prediction = predict(tree, dataset[i]);
        printf("%d -> %.2f\n", dataset[i].label, prediction);
    }

    // Clean up
    free_tree(tree);
    free_data(dataset);

    return 0;
}
