import numpy as np

def causal_pad(input, kernel):
    # Left-pad with (kernel_size - 1) zeros so each output only sees current and past inputs.
    pad = kernel.shape[1] - 1
    return np.pad(input, ((0, 0), (pad, 0)))

def forward(input, kernel):
    padded = causal_pad(input, kernel)
    output_length = input.shape[1]

    outputs = np.zeros(output_length)

    for i in range(output_length):
        output = 0
        for ki in range(kernel.shape[1]):
            output += padded[0][i + ki] * kernel[0][ki]
        outputs[i] = output

    return outputs

def gradients(input, kernel, output, target):
    padded = causal_pad(input, kernel)
    output_length = input.shape[1]

    dl_dz = 2 * (output - target)
    dl_dW = np.zeros(kernel.shape)

    for i in range(output_length):
        for ki in range(kernel.shape[1]):
            dl_dW[0][ki] += dl_dz[i] * padded[0][i + ki]

    return dl_dW

def loss(output, target):
    return np.sum((output - target)**2)

input = np.random.randn(1, 4)
kernel = np.random.randn(1, 2)
target = np.array([1,1,1,1])

for epoch in range(20):
    output = forward(input, kernel)
    dl_dW = gradients(input, kernel, output, target)
    kernel -= 0.01 * dl_dW
    print(f"Epoch {epoch+1}, Loss: {loss(output, target)}")

