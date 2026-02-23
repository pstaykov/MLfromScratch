#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>

#define DATA_SIZE 10

typedef struct {
    int values[2];
    int label;
} datum;

typedef struct {
    float x, y;
} centroid;

float Distance(datum d, centroid c) {
    float dx = (float)d.values[0] - c.x;
    float dy = (float)d.values[1] - c.y;
    return sqrtf(dx * dx + dy * dy);
}

int find_min_label(float dists[], int k) {
    float min_val = FLT_MAX;
    int min_idx = 0;
    for (int i = 0; i < k; i++) {
        if (dists[i] < min_val) {
            min_val = dists[i];
            min_idx = i;
        }
    }
    return min_idx;
}

void Kmeans(datum* data, int n, int k) {
    centroid centroids[k];
    float distances[k];
    int changed = 1;

    // 1. Initialize centroids randomly from data
    for (int i = 0; i < k; i++) {
        int idx = rand() % n;
        centroids[i].x = (float)data[idx].values[0];
        centroids[i].y = (float)data[idx].values[1];
    }

    while (changed) {
        changed = 0;

        // Distance Calculation and Label Assignment
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                distances[j] = Distance(data[i], centroids[j]);
            }
            int new_label = find_min_label(distances, k);
            if (data[i].label != new_label) {
                data[i].label = new_label;
                changed = 1; // Keep looping if even one point moved
            }
        }

        // Update Centroids (Calculate Mean)
        for (int i = 0; i < k; i++) {
            float sum_x = 0, sum_y = 0;
            int count = 0;
            for (int j = 0; j < n; j++) {
                if (data[j].label == i) {
                    sum_x += (float)data[j].values[0];
                    sum_y += (float)data[j].values[1];
                    count++;
                }
            }
            if (count > 0) {
                centroids[i].x = sum_x / (float)count;
                centroids[i].y = sum_y / (float)count;
            }
        }
    }
}

int main() {
    datum myData[DATA_SIZE] = {
        {{1, 2}, -1}, {{2, 3}, -1}, {{3, 1}, -1},
        {{6, 5}, -1}, {{7, 8}, -1}, {{8, 6}, -1},
        {{1, 0}, -1}, {{2, 1}, -1}, {{5, 7}, -1},
        {{6, 9}, -1}
    };

    int k = 2;

    // function directly modifies data
    Kmeans(myData, DATA_SIZE, k);

    // Print results
    printf("Results:\n");
    for (int i = 0; i < DATA_SIZE; i++) {
        printf("Point (%d, %d) -> Cluster %d\n",
                myData[i].values[0], myData[i].values[1], myData[i].label);
    }

    return 0;
}