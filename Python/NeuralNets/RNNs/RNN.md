## Recurrent Neural Networks
Or RNNs are a type of Neural Network that can remember information from previous time steps and is used
to model sequential data. RNNs can be used for many tasks such as speech recognition, language modeling, and time series prediction such as 
stock prices.

## Functionality

### Forward Propagation
The forward propagation of an RNN is done by iterating over the time steps of the input sequence.
The output of the previous time step is used as the input of the next time step as well as the hidden state.
The output of the last time step is the final output of the RNN. The hidden state is used to store information about the sequence.

#### Algebraically
Algebraically, we can describe the forward propagation of an RNN as follows:
$$H_t = f(W_{xh}X_t + W_{hh}H_{t-1} + b_h)$$
$$Y_t = W_{hy}H_t + b_y$$
Where:
- $H_t$ is the hidden state at time step $t$ (vector of arbitrary length)
- $X_t$ is the input at time step $t$
- $Y_t$ is the output at time step $t$
- $f$ is the activation function (usually a non-linear function such as ReLU or Tanh)
- $W_{xh}$ is the weight matrix for the input to hidden layer
- $W_{hh}$ is the weight matrix for the hidden to hidden layer
- $W_{hy}$ is the weight matrix for the hidden to output layer
- $b_h$ is the bias for the hidden layer
- $b_y$ is the bias for the output layer

Intuitevly this means that the hidden state at time step $t$ is a functions of a weighted sum of the input plus a weighted sum of the hidden state at time step $t-1$.
The output at time step $t$ is a weighted sum of the hidden state at time step $t$.

here you can see an example of the RNN structure:
![RNN](https://www.researchgate.net/profile/Weijiang-Feng/publication/318332317/figure/fig1/AS:614309562437664@1523474221928/The-standard-RNN-and-unfolded-RNN.png)
Where:
- W is the $W_{hh}$ matrix 
- U is the $W_{xh}$ matrix
- V is the $W_{hy}$ matrix. 

### Backpropagation Through Time (BPTT)

The required gradients are:
- $\frac{\partial L}{\partial W_{hh}}$ loss w.r.t. the hidden-to-hidden weights
- $\frac{\partial L}{\partial W_{xh}}$ loss w.r.t. the input-to-hidden weights
- $\frac{\partial L}{\partial W_{hy}}$ loss w.r.t. the hidden-to-output weights

---

#### Loss w.r.t. hidden-to-hidden weights: $\frac{\partial L}{\partial W_{hh}}$

Each timestep $t$ contributes a gradient that must be chained **back through all previous hidden states**:

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial H_t} \left(\prod_{k=t}^{1} \frac{\partial H_k}{\partial H_{k-1}}\right) \frac{\partial H_t}{\partial W_{hh}}$$

#### Example with $T = 3$ timesteps:

$$\frac{\partial L}{\partial W_{hh}} =
\frac{\partial L}{\partial Y} \frac{\partial Y}{\partial H_3} \frac{\partial H_3}{\partial W_{hh}} + \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial H_3} \frac{\partial H_3}{\partial H_2} \frac{\partial H_2}{\partial W_{hh}} + \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial H_3} \frac{\partial H_3}{\partial H_2} \frac{\partial H_2}{\partial H_1} \frac{\partial H_1}{\partial W_{hh}}$$

> Each earlier timestep needs a longer chain because its contribution
> to the loss traveled through more hidden states to reach the output.

---

#### Loss w.r.t. input-to-hidden weights: $\frac{\partial L}{\partial W_{xh}}$

Each timestep contributes independently through its own input $X_t$:

$$\frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial H_t} \frac{\partial H_t}{\partial W_{xh}}$$

#### Example with $T = 3$ timesteps:

$$\frac{\partial L}{\partial W_{xh}} = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial H_3} \frac{\partial H_3}{\partial W_{xh}}+ \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial H_2} \frac{\partial H_2}{\partial W_{xh}}+ \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial H_1} \frac{\partial H_1}{\partial W_{xh}}$$

> No cross-timestep chaining here — each $X_t$ only directly affects $H_t$.

---

#### Loss w.r.t. output weights: $\frac{\partial L}{\partial W_{hy}}$

The output only connects to the **final** hidden state $H_T$, so there is a single term:

$$\frac{\partial L}{\partial W_{hy}} = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial W_{hy}}$$

### Concrete Example: BPTT with $T = 3$ Timesteps

**Setup:**
- Input sequence: $x_1 = 1,\ x_2 = 2,\ x_3 = 3$
- True target: $y = 10$
- All weights initialized to $0.5$: $W_{xh} = 0.5,\ W_{hh} = 0.5,\ W_{hy} = 0.5$
- All biases zero, activation $\text{ReLU}$, output is linear
- Loss: MSE $= \frac{1}{2}(y - \hat{y})^2$

---

#### Forward Pass

$$H_0 = 0$$

$$H_1 = \text{ReLU}(W_{xh} \cdot x_1 + W_{hh} \cdot H_0) = \text{ReLU}(0.5 \cdot 1 + 0.5 \cdot 0) = \text{ReLU}(0.5) = 0.5$$

$$H_2 = \text{ReLU}(W_{xh} \cdot x_2 + W_{hh} \cdot H_1) = \text{ReLU}(0.5 \cdot 2 + 0.5 \cdot 0.5) = \text{ReLU}(1.25) = 1.25$$

$$H_3 = \text{ReLU}(W_{xh} \cdot x_3 + W_{hh} \cdot H_2) = \text{ReLU}(0.5 \cdot 3 + 0.5 \cdot 1.25) = \text{ReLU}(2.125) = 2.125$$

$$\hat{y} = W_{hy} \cdot H_3 = 0.5 \cdot 2.125 = 1.0625$$

$$L = \frac{1}{2}(10 - 1.0625)^2 = \frac{1}{2}(8.9375)^2 \approx 39.94$$

---

#### Backward Pass

**Step 1 — gradient of loss into the network:**

$$\frac{\partial L}{\partial \hat{y}} = -(y - \hat{y}) = -(10 - 1.0625) = -8.9375$$

---

**Step 2 — gradient w.r.t. $W_{hy}$:**

$$\frac{\partial L}{\partial W_{hy}} = \frac{\partial L}{\partial \hat{y}} \cdot H_3 = -8.9375 \cdot 2.125 \approx -18.992$$

---

**Step 3 — gradient into $H_3$:**

$$\frac{\partial L}{\partial H_3} = \frac{\partial L}{\partial \hat{y}} \cdot W_{hy} = -8.9375 \cdot 0.5 = -4.469$$

---

**Step 4 — backprop through ReLU at each timestep:**

Recall: $\frac{\partial}{\partial z}\text{ReLU}(z) = \mathbb{1}[z > 0]$ — gradient is $1$ if the pre-activation was positive, $0$ otherwise.

All pre-activations were positive $(0.5,\ 1.25,\ 2.125)$, so the ReLU derivative is $1$ everywhere.

$$\delta_3 = \frac{\partial L}{\partial H_3} \cdot 1 = -4.469$$

$$\delta_2 = \delta_3 \cdot W_{hh} \cdot 1 = -4.469 \cdot 0.5 = -2.234$$

$$\delta_1 = \delta_2 \cdot W_{hh} \cdot 1 = -2.234 \cdot 0.5 = -1.117$$

> With ReLU the derivative is just $1$ — no squashing. The gradient halves
> at each step only because of $W_{hh} = 0.5$, not the activation.
> With $W_{hh} \geq 1$ the gradient would stay constant or **explode**.

---

**Step 5 — accumulate gradients for $W_{hh}$ and $W_{xh}$:**

$$\frac{\partial L}{\partial W_{hh}} = \delta_3 \cdot H_2 + \delta_2 \cdot H_1 + \delta_1 \cdot H_0$$
$$= (-4.469)(1.25) + (-2.234)(0.5) + (-1.117)(0)$$
$$= -5.586 - 1.117 + 0 = -6.703$$

$$\frac{\partial L}{\partial W_{xh}} = \delta_3 \cdot x_3 + \delta_2 \cdot x_2 + \delta_1 \cdot x_1$$
$$= (-4.469)(3) + (-2.234)(2) + (-1.117)(1)$$
$$= -13.407 - 4.468 - 1.117 = -18.992$$

---

#### Summary of Gradients

| Weight | Gradient |
|--------|----------|
| $W_{hy}$ | $-18.992$ |
| $W_{hh}$ | $-6.703$ |
| $W_{xh}$ | $-18.992$ |

With learning rate $\eta = 0.01$, the weight update is $W \leftarrow W - \eta \cdot \frac{\partial L}{\partial W}$:

| Weight | Old value | Update | New value |
|--------|-----------|--------|-----------|
| $W_{hy}$ | $0.500$ | $+0.190$ | $0.690$ |
| $W_{hh}$ | $0.500$ | $+0.067$ | $0.567$ |
| $W_{xh}$ | $0.500$ | $+0.190$ | $0.690$ |