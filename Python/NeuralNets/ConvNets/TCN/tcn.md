# TCN - Temporal Convolutional Networks

## 1D CNN
In order to motivate the usage and functionality of TCNs, we will first look at 1D CNNs. TCNs aim to capture
temporal dependencies in the data. Time series data is often represented as a Vector in $\mathbb{R}^n$ where
$n$ is the number of features. Intuitively, we would make the kernel of the convolutional layer a vector of
arbitrary length, say $k$. The convolution operation would then be performed on the input vector. The problem is,
that at position $t$ of the kernel, we would only capture information from $t-\frac{k-1}{2}$ to $t+\frac{k-1}{2}$. Not only is this a short timeframe,
but most importantly, it captures information from the future, which is not desirable in time series forecasting (though 
it could be useful in other applications like classification). 

## Temporal Convolution
This is where TCNs come in, or more precisely, casual convolution.
It changes the position the kernel is applied to, from $t \pm \frac{k-1}{2}$ to $t-k$ to $t$. It follows, that the kernel
is now not seing the future, but only the past, which is what we want.

## Dilation
It remains a problem, that the kernel is not able to capture temporal dependencies far away from the current position.
This is where dilation comes in. Dilation spaces out the kernels look back points/ positions.

**Example:**
- Kernel size = 3
- Dilation = 1 → [t-2, t-1, t]
- Dilation = 2 → [t-4, t-2, t]
- Dilation = 4 → [t-8, t-4, t]

chaining multiple layers with increasing dilation allows the network to capture long-term dependencies efficiently, without the need for a large kernel size.

## Architecture
A TCN is basically:
1. Causal convolutions → prevent future leakage.
2. Dilations → allow long-term memory efficiently.
3. Residual blocks → make deep networks trainable.

### Residual Blocks
Oftenly Each block contains:
- Two causal dilated convolution layers (same dilation in the block).
- Normalization + activation + dropout between them.
- A skip connection from input to output.

Why residuals?
- Prevent loss of important signals.
- Help gradients flow backward for easier training.