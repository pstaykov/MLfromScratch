# Convolutional Neural Networks (CNNs)

Convolutional Neural Networks are a class of deep learning models designed to recognize features and patterns in images.

While a standard neural network can handle relatively simple tasks, such as classifying handwritten digits, it struggles with more 
complex visual recognition. Consider detecting a cat in a photo: a standard network may correctly label the image, but it cannot reliably *localize* the cat, 
and it is highly sensitive to position. A model trained on cats centered in frame will likely fail when the cat appears in a corner.

CNNs solve this through a mechanism called **convolution**, which allows them to scan across an entire image for relevant features regardless of where those features appear.

## Key Properties

- **Translation invariance** — CNNs recognize objects regardless of their position in the image, since the convolution operation scans the full input rather than expecting features at fixed locations.
- **Locality** — CNNs focus on spatially local regions of the image at a time, capturing fine-grained structure without being distracted by unrelated areas.

## How It Works

### The Kernel (Filter)

The core of a CNN is the **convolution layer**, which applies a small matrix of learnable weights called a **kernel** (or filter) to the input.

The operation works by sliding the kernel across every possible position in the input, computing the element-wise product between the kernel and the overlapping region, 
summing the results, and writing the output to the corresponding position in a new matrix. This output is called a **feature map**, and it highlights where in the image a particular pattern was detected.

here you can see an example of a convolution operation with a 3 by 3 kernel applied to a 6 by 6 input image.

![picture](https://upload.wikimedia.org/wikipedia/commons/1/19/2D_Convolution_Animation.gif)

### Padding and Strides
The kernel is applied to the input image by sliding it across the image, and the way it moves is determined by two parameters: **padding** and **strides**.
- **Padding** is the amount of blank space around the image that is added to ensure that the kernel can fully cover the entire image. Usually, this is done by adding zeros to the edges of the image.
- **Strides** control how far the kernel moves across the image at each step.

for example here you can see a 3 by 3 kernel going across a 5 by 5 input with a padding of 1 and a stride of 2.

![picture](https://towardsdatascience.com/wp-content/uploads/2024/11/0SUjDc9LYZbEHTWAC.gif)
 
### Pooling
Pooling is a technique that reduces the size of the feature maps by applying a **max pooling** or **average pooling** operation.
- **Max pooling** selects the maximum value in each region of the feature map,
- **Average pooling** averages the values in each region.
- Both operations reduce the size of the feature map

It works very similarly to convolution, but instead of sliding the kernel across the image, it slides a pooling window across the feature map and applies the pooling operation to each region.

## Loss Function
The loss function for a CNN is typically the same as for a standard neural network, such as cross-entropy loss for classification tasks. 

## Gradient calculation
To calculate the gradient of the loss function with respect to the weights of the convolutional layer, 
we use the chain rule.
We set $L$ to the loss, $W_i$ to the weight of the i-th convolutional layer and $z_j$ to the output of the j-th convolution.

$$ \frac{\partial L}{\partial W_i} = \sum_{j=1}^n \frac{\partial L}{\partial z_j} \frac{\partial z_j}{\partial W_i}$$

In other words, the gradient of the loss with respect to the weights of the convolutional layer is the sum 
of the gradients of the loss with respect to the outputs of each convolutional layer times the gradients of 
the outputs with respect to the weights of the convolutional layer.

### example
I will use an example to illustrate the gradient calculation.

Say we have a 3×3 input and a 2×2 filter:

$$X = \begin{bmatrix}x_1 & x_2 & x_3 \\ x_4 & x_5 & x_6 \\ x_7 & x_8 & x_9\end{bmatrix}, \quad W = \begin{bmatrix}w_1 & w_2 \\ w_3 & w_4\end{bmatrix}$$

The 4 output positions are:

$$Z_1 = w_1 x_1 + w_2 x_2 + w_3 x_4 + w_4 x_5$$
$$Z_2 = w_1 x_2 + w_2 x_3 + w_3 x_5 + w_4 x_6$$
$$Z_3 = w_1 x_4 + w_2 x_5 + w_3 x_7 + w_4 x_8$$
$$Z_4 = w_1 x_5 + w_2 x_6 + w_3 x_8 + w_4 x_9$$

$w_1$ appears in all four outputs, so:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial Z_1} x_1 + \frac{\partial L}{\partial Z_2} x_2 + \frac{\partial L}{\partial Z_3} x_4 + \frac{\partial L}{\partial Z_4} x_5$$

$$= \delta_1 x_1 + \delta_2 x_2 + \delta_3 x_4 + \delta_4 x_5$$

The same logic applies to every weight — $w_i$ picks up the input patch element it multiplied at each output position, summed over all positions it contributed to.




