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
and our network transforms it into the prediction $\hat{y}$.

We illustrate this with a network with **2 inputs → 1 hidden layer of 3 neurons (ReLU) → 1 output neuron (sigmoid)**.

### Stage 1: $W_1x + b_1 = z_1$
Every input value gets multiplied by a weight, they all get added together, 
and a bias is added to specify when the neuron should get activated. 
$W_1$ is a $3 \times 2$ matrix (3 hidden neurons, 2 inputs), so $z_1$ is a vector of 3 pre-activations.
The result $z_1$ is called the pre-activation because it hasn't been "activated" yet. 
The problem with stopping here is that stacking multiple linear layers is mathematically equivalent to just one linear layer. 
That's why we need the next step.

### Stage 2: $\text{ReLU}(z_1) = a_1$
The relu function is a non-linear activation function. It is defined as $\text{ReLU}(x) = \max(0, x)$.
Negative values become zero, positive values pass through unchanged. 
This nonlinearity is what allows the network to learn curved, complex decision boundaries instead of just straight lines. 
The output $a_1$ is a vector of 3 hidden layer activations.

### Stage 3: $W_2 \cdot a_1 + b_2 = z_2$
Same idea as Stage 1, but now the inputs are the hidden activations $a_1$ instead of the raw data. 
$W_2$ is a $1 \times 3$ row vector (1 output neuron, 3 hidden neurons), producing a single scalar $z_2$.

### Stage 4: $\sigma(z_2) = \hat{y}$
Sigmoid squishes any number into the range (0, 1) using $\frac{1}{1 + e^{-z_2}}$. 
This is used at the output because our targets are between 0 and 1 (probabilities or binary labels). 
The result $\hat{y}$ is the network's prediction.

### Stage 5: $L = (\hat{y} - y)^2$
We compare $\hat{y}$ to the true label $y$ using Mean Squared Error.
This single number tells us how wrong the network is. The entire goal of training is to make this number as small as possible.

## Backward Pass
The backward pass is the opposite of the forward pass. We want to find the gradient of the loss function with respect to each parameter. 
In other words, we want to know: "which weights caused the most error, and in which direction should we change them to reduce the error?" 
The chain rule from calculus tells us that the gradient of a composed function is the product of the partial derivatives at each step.

**Example network setup** — 2 inputs, 3 hidden neurons (ReLU), 1 output neuron (sigmoid), MSE loss:

| Symbol | Value |
|--------|-------|
| $x$ | $[1.0,\ 0.5]$ |
| $y$ (true label) | $0.0$ |
| $z_1$ | $[0.4,\ {-0.1},\ 0.8]$ |
| $a_1 = \text{ReLU}(z_1)$ | $[0.4,\ 0.0,\ 0.8]$ |
| $W_2$ | $[0.3,\ 0.5,\ {-0.2}]$ |
| $z_2 = W_2 \cdot a_1 + b_2$ | $0.52$ |
| $\hat{y} = \sigma(z_2)$ | $0.627$ |
| $L = (\hat{y} - y)^2$ | $0.393$ |

---

### Step 1: Output layer delta: $\delta_2$

Starting from the loss, we ask: "how does the loss change if we change $z_2$ by a tiny bit?" This requires two things multiplied together via the chain rule:

- The derivative of the loss with respect to the prediction $\hat{y}$:

$$\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)$$

- The derivative of the sigmoid with respect to its input $z_2$:

$$\sigma'(z_2) = \sigma(z_2)(1 - \sigma(z_2))$$

Multiplied together, this gives us the output layer delta $\delta_2$ — the "blame signal" for the output layer:

$$\delta_2 = \frac{\partial L}{\partial \hat{y}} \cdot \sigma'(z_2) = 2(\hat{y} - y) \cdot \sigma(z_2)(1-\sigma(z_2))$$

Since there is only one output neuron, $\delta_2$ is a **scalar**.

From $\delta_2$ we can immediately compute the gradients for $W_2$ and $b_2$:

$$\frac{\partial L}{\partial W_2} = \delta_2 \cdot a_1 \qquad \frac{\partial L}{\partial b_2} = \delta_2$$

Because $\delta_2$ is a scalar and $a_1$ is a vector, this is simply scalar multiplication — each element of $a_1$ is scaled by $\delta_2$.

> **Example:**
> $$\frac{\partial L}{\partial \hat{y}} = 2(0.627 - 0.0) = 1.254$$
> $$\sigma'(z_2) = 0.627 \times (1 - 0.627) = 0.627 \times 0.373 = 0.234$$
> $$\delta_2 = 1.254 \times 0.234 = \mathbf{0.293}$$
>
> Gradients for the output layer weights and bias:
> $$\frac{\partial L}{\partial W_2} = 0.293 \times [0.4,\ 0.0,\ 0.8] = [0.117,\ 0.000,\ 0.235]$$
> $$\frac{\partial L}{\partial b_2} = 0.293$$

---

### Step 2 — Propagate error back to the hidden layer

We now ask: "how much did each hidden neuron contribute to the output error?" We route $\delta_2$ backwards through $W_2$:

$$\frac{\partial L}{\partial a_1} = \delta_2 \cdot W_2$$

Because $\delta_2$ is a scalar and $W_2$ is a row vector $[w_1, w_2, w_3]$, this is again scalar multiplication. Each hidden neuron receives a share of the blame proportional to the magnitude of its weight connecting it to the output.

> **Example:**
>
> $W_2 = [0.3,\ 0.5,\ {-0.2}]$, $\delta_2 = 0.293$:
> $$\frac{\partial L}{\partial a_1} = 0.293 \times [0.3,\ 0.5,\ {-0.2}] = \mathbf{[0.088,\ 0.147,\ {-0.059}]}$$
>
> Neuron 1 (weight 0.3) gets a small positive blame, neuron 2 (weight 0.5) gets more, and neuron 3 (weight −0.2) gets a small negative signal — meaning increasing its activation would have *reduced* the loss.

---

### Step 3 — Hidden layer delta: $\delta_1$

The hidden activations $a_1$ came through a ReLU. ReLU's derivative is a simple gate: 1 where the neuron was active during the forward pass, 0 where it was off. Neurons that output zero contributed nothing to the error, so they receive no gradient:

$$\text{ReLU}'(z_1) = \begin{cases} 1 & \text{if } z_1 > 0 \\ 0 & \text{if } z_1 \leq 0 \end{cases}$$

Multiplying element-wise gives us $\delta_1$:

$$\delta_1 = \frac{\partial L}{\partial a_1} \odot \text{ReLU}'(z_1) = (\delta_2 \cdot W_2) \odot \text{ReLU}'(z_1)$$

where $\odot$ denotes element-wise multiplication.

> **Example:**
>
> $z_1 = [0.4,\ {-0.1},\ 0.8]$, so the ReLU gate is:
> $$\text{ReLU}'(z_1) = [1,\ 0,\ 1]$$
>
> Neuron 2 was inactive ($z_1 = -0.1 \leq 0$), so it is fully blocked — it gets no gradient regardless of how much blame was routed to it:
> $$\delta_1 = [0.088,\ 0.147,\ {-0.059}] \odot [1,\ 0,\ 1] = \mathbf{[0.088,\ 0.000,\ {-0.059}]}$$

---

### Step 4 — Gradients for $W_1$ and $b_1$

With $\delta_1$ in hand, the gradients for the first layer follow the same pattern as Step 1.
$\delta_1$ is a vector of shape $(3,)$ and $x$ is a vector of shape $(2,)$, so their outer product gives a $(3 \times 2)$ gradient matrix — one row per hidden neuron, one column per input:

$$\frac{\partial L}{\partial W_1} = \delta_1 \otimes x \qquad \frac{\partial L}{\partial b_1} = \delta_1$$

> **Example:**
>
> $\delta_1 = [0.088,\ 0.000,\ {-0.059}]$, $x = [1.0,\ 0.5]$
>
> $$\frac{\partial L}{\partial W_1} = \begin{bmatrix} 0.088 \times 1.0 & 0.088 \times 0.5 \\ 0.000 \times 1.0 & 0.000 \times 0.5 \\ {-0.059} \times 1.0 & {-0.059} \times 0.5 \end{bmatrix} = \begin{bmatrix} 0.088 & 0.044 \\ 0.000 & 0.000 \\ {-0.059} & {-0.030} \end{bmatrix}$$
>
> $$\frac{\partial L}{\partial b_1} = [0.088,\ 0.000,\ {-0.059}]$$
>
> Neuron 2's entire row is zero — a dead neuron produces no weight update whatsoever.

---

### Step 5 — SGD parameter update

With all gradients computed, every parameter is nudged in the direction that reduces the loss. For learning rate $\eta$:

$$\theta \leftarrow \theta - \eta \cdot \frac{\partial L}{\partial \theta}$$

Applied to each parameter:

$$W_1 \leftarrow W_1 - \eta \cdot \frac{\partial L}{\partial W_1}, \quad b_1 \leftarrow b_1 - \eta \cdot \frac{\partial L}{\partial b_1}$$

$$W_2 \leftarrow W_2 - \eta \cdot \frac{\partial L}{\partial W_2}, \quad b_2 \leftarrow b_2 - \eta \cdot \frac{\partial L}{\partial b_2}$$

We subtract because we want to move *downhill* on the loss surface — opposite to the direction the gradient points.

> **Example** with $\eta = 0.1$:
>
> $$W_2 \leftarrow [0.300,\ 0.500,\ {-0.200}] - 0.1 \times [0.117,\ 0.000,\ 0.235]$$
> $$W_2 \leftarrow \mathbf{[0.288,\ 0.500,\ {-0.224}]}$$
>
> $$b_2 \leftarrow 0.0 - 0.1 \times 0.293 = \mathbf{-0.029}$$
>
> For the hidden layer (first row of $W_1$ shown):
> $$W_1[0] \leftarrow [w_{11},\ w_{12}] - 0.1 \times [0.088,\ 0.044] \quad \Rightarrow \quad \text{each weight shifts slightly negative}$$
> $$W_1[2] \leftarrow [w_{31},\ w_{32}] - 0.1 \times [{-0.059},\ {-0.030}] \quad \Rightarrow \quad \text{each weight shifts slightly positive}$$
>
> After this update, $\hat{y}$ will be slightly closer to $0.0$ on the next forward pass.





### notes on AI usage
Ai was used for research, example notation and visualisation.