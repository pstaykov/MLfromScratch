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

input = np.random.randn(1, 3, 3)
print(input)

kernel = np.random.randn(1, 2, 2)
print(kernel)

print(forward(input, kernel))