"""
this was written by me
2d CNN with stride 1 padding 0
"""

import numpy as np

def forward(input, kernel):
    output_height = input.shape[1] - kernel.shape[1] + 1
    output_width = input.shape[2] - kernel.shape[2] + 1

    outputs = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output = 0
            for ki in range(kernel.shape[1]):
                for kj in range(kernel.shape[2]):
                    output += input[0][i + ki][j + kj] * kernel[0][ki][kj]
            outputs[i][j] = output

    return outputs

def gradients(input, kernel, output, target):
    output_height = input.shape[1] - kernel.shape[1] + 1
    output_width = input.shape[2] - kernel.shape[2] + 1

    dl_dz = 2 * (output - target.reshape(output_height, output_width))

    dl_dW = np.zeros((1, kernel.shape[1], kernel.shape[2]))

    for ki in range(kernel.shape[1]):
        for kj in range(kernel.shape[2]):
            for i in range(output_height):
                for j in range(output_width):
                    dl_dW[0][ki][kj] += dl_dz[i][j] * input[0][i + ki][j + kj]

    return dl_dW

def loss(output, target):
    return np.sum((output - target)**2)

input = np.random.randn(1, 3, 3)
kernel = np.random.randn(1, 2, 2)
target = np.array([1,1,1,1])

for epoch in range(20):
    output = forward(input, kernel)
    output.flatten()
    dl_dW = gradients(input, kernel, output, target)
    kernel -= 0.01 * dl_dW
    print(f"Epoch {epoch+1}, Loss: {loss(output.flatten(), target)}")



