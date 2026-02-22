//
// Created by pstay on 2/22/26.
//
#include <stdio.h>

float LinearRegression(int x)
{
    float X[5] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    float y[5] = { 10.0, 20.0, 30.0, 40.0, 50.0 };

    float weight = 0.0;
    float bias = 0.0;
    const int epochs = 100;
    const float learning_rate = 0.01;

    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < 5; j++) {
            float prediction = X[j] * weight + bias;
            float error = y[j] - prediction;
            weight += learning_rate * error * X[j];
            bias += learning_rate * error;
        }
    }

    return x * weight + bias;
}

int main(){
    printf("%f", LinearRegression(10));
}