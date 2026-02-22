//
// Created by pstay on 2/22/26.
//
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define DATA_SIZE 10

typedef struct {
    int values[2];
    int label;
} datum;

typedef struct {
    float distance;
    int label;
} neighbor;

int compareNeighbors(const void *a, const void *b) {
    float da = ((const neighbor*)a)->distance;
    float db = ((const neighbor*)b)->distance;
    return (da > db) - (da < db);
}

float Distance(datum c1, datum c2) {
    float dif_x = c1.values[0] - c2.values[0];
    float dif_y = c1.values[1] - c2.values[1];
    return sqrt(dif_x * dif_x + dif_y * dif_y);
}

int KNN(datum datapoint, int k)
{
    datum data[DATA_SIZE] = {
        {{1, 2}, 0},
        {{2, 3}, 0},
        {{3, 1}, 0},
        {{6, 5}, 1},
        {{7, 8}, 1},
        {{8, 6}, 1},
        {{1, 0}, 0},
        {{2, 1}, 0},
        {{5, 7}, 1},
        {{6, 9}, 1}
    };

    neighbor neighbors[DATA_SIZE];

    for (int i = 0; i < DATA_SIZE; i++) {
        neighbors[i].distance = Distance(datapoint, data[i]);
        neighbors[i].label = data[i].label;
    }

    qsort(neighbors, DATA_SIZE, sizeof(neighbor), compareNeighbors);

    // Take first k neighbors and perform majority vote
    int votes[2] = {0, 0};
    for (int i = 0; i < k; i++) {
        votes[neighbors[i].label]++;
    }

    return (votes[1] > votes[0]) ? 1 : 0;
}

int main() {
    datum query = {{4, 5}, -1};
    int k = 3;
    int prediction = KNN(query, k);
    printf("Predicted label for (%d, %d) with k=%d: %d\n",
           query.values[0], query.values[1], k, prediction);
    return 0;
}