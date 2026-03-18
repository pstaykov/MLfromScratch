# Neural Network

## What is a neural Network?

At its core, a neural network is a function that takes many inputs and 
produces outputs. Its inspired by the way the brain works but its really just chained
linear algebra.

Neurons are weightes sums. Each neuron takes its inputs and produces an output using 
the assigned weights to each input and a bias and finally passes the output through a 
non-linear activation function.

Mathematically a singe neuron is:

$$ z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$

$$ a = \sigma(z)$$

where $w$ is the weights, $x$ is the inputs, 
$b$ is the bias, $\sigma$ is the activation function 
and $a$ is the output of the neuron.

### Neural Network Architecture
We can chain many layers of neurons together to create a neural network. One layer excluding
the input and output layer is called a hidden layer. The Input layer represent our given data
and the output layer is the result of the neural network.

Usually we represent the input and the output layer as a vector. The weights between the layers
for each neuron are represented as a matrix including the bias.

## Stochastic Gradient Descent (SGD)
We imagine a multi dimensional space representing the expected loss. We then move towards the 
minimum by taking small steps in the direction of the steepest descent by calculating the 
gradient(the multi dimentional derivative) of the loss function at each point.

The update rule for every parameter θ (weight or bias):
$$ \theta \leftarrow \theta - \eta \cdot \frac{\partial L}{\partial \theta}$$

- $\theta$ — any parameter (a weight w or bias b)
- $\eta$ (eta) — the learning rate: how big a step to take. Too large $\rightarrow$ you overshoot. Too small $\rightarrow$ training is slow.
- $\frac{\delta L}{\delta \theta}$ — the gradient: "if I increase $\theta$ by a tiny bit, how much does the loss increase?" If this is positive, we subtract (step downhill). If negative, we add.

Stochastic means, that we take a single sample or a small batch each step instead of the whole datset.
This is noiser but much faster and helps escape local minima.

## Forward Pass 
In the forward pass we calculate the output of each neuron. We input our data $x$
and out network transforms it into the prediction $\hat{y}$

We illustrate this with a simple linear network with 2 hidden layers and 1 neuron each.

### Stage 1: $W_1x + b_1 = z_1$
Every input value gets multiplied by a weight, they all get added together, 
and a bias is added to specify when the neuron should get activated. 
The result $z_1$ is called the pre-activation because it hasn't been "activated" yet. 
The problem with stopping here is that stacking multiple linear layers is mathematically equivalent to just one linear layer 
That's why we need the next step.

### Stage 2: $ReLU(z_1) = a_1$
The relu function is a non-linear activation function. It is defined as $ReLU(x) = max(0, x)$.
Negative values become zero, positive values pass through unchanged. 
This nonlinearity is what allows the network to learn curved, complex decision boundaries instead of just straight lines. 
The output $a_1$ is called the hidden layer activation. 

### Stage 3: $W_2·a_1 + b_2 → z_2$
Same idea as Stage 1, but now the inputs are the hidden activations a₁ instead of the raw data. 
This second weight matrix learns to combine the hidden layer's features into output-sized values.

### Stage 4: $\sigma{z_2} → \hat{y}$
Sigmoid squishes any number into the range (0, 1) using $\frac{1}{1 + e^-z_2}$. 
This is used at the output because our targets are between 0 and 1 (probabilities or binary labels). 
The result $$\hat{y}$$ is the network's prediction.

### Stage 5: $L = \frac{1}{m} \sum_{i=1}^m (y_i - \hat{y}_i)^2$
You may recall this loss function from Linear Regression.
We compare $$\hat{y}$$ to the true label $y$ using Mean Squared Error: $L = ($\hat{y}$ - y)^2$. 
This single number tells us how wrong the network is. The entire goal of training is to make this number as small as possible.

## Backward Pass
The backward pass is the opposite of the forward pass. We want to find the gradient of the loss function with respect to each parameter.
In other words, we want to know: "which weights caused the most error, and in which
direction should we change them to reduce the error?" The chain rule from calculus tells us that the gradient is the product 
of the partial derivatives of the loss function with respect to each parameter.

$\frac{\delta L}{\delta z_2} · \sigma'$ is the Output layer delta. 

Starting from the loss, we ask: "how does the loss change if we change $z_2$ by a tiny bit?" 
This requires two things multiplied together via the chain rule:
- the derivative of the loss with respect to $z_2$: $\frac{\delta L}{\delta \hat{y}} = 2(\hat{y} - y)$
- the derivative of the sigmoid function with respect to $z_2$: $\sigma'(z_2) = \sigma(z)\times(1-\sigma(z))$
