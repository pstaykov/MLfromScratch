# Gradient Boosting

Gradient Boosting is considered the most powerful algorithm in the world of machine learning. 
It is an algorithm that combines multiple weak learners to create a strong learner. 
The idea behind Gradient Boosting is to build the model in a stage-wise fashion, where each stage tries to correct 
the errors of the previous stage.

## The "Correction" Strategy

### 1. The Starting Point (Initialization)
You can't calculate a residual without a first guess. We start with a constant value $F_0(x)$—usually the mean of all your target values $y$.

$$F_0(x) = \text{mean}(y)$$

### 2. The Iterative Loop
For $m = 1$ to $M$ trees:

* **A. Compute the "Gradients" (Pseudo-residuals):**
  For every single data point, calculate how far off you are.
  $$r_{im} = y_i - F_{m-1}(x_i)$$
  (If using MSE, this is just your negative gradient).

* **B. Fit a "Weak Learner" (The Tree):**
  Train a simple decision tree (usually very shallow, like 3-5 levels deep) to predict those residuals ($r_{im}$) instead of the actual values ($y_i$).

* **C. Calculate the Step Size (Output Value):**
  For each leaf in this new tree, determine the best value to add to the model to minimize loss. (In simple regression, this is usually just the average residual in that leaf).

* **D. Update the Model:**
  Add this new tree to your existing model, but multiply it by a small Learning Rate ($\nu$).
  $$F_m(x) = F_{m-1}(x) + \nu \cdot \text{Tree}_m(x)$$

## The Gradient
The term "Gradient" comes from Gradient Descent. In mathematics, a gradient tells 
you the direction and rate of fastest increase of a function.In boosting, we define a 
Loss Function (which measures how far off our predictions are). 
We use the gradient of that loss function to tell the next tree exactly how to adjust the predictions to 
reduce the error as quickly as possible.+1Mathematically, if we are predicting $y$ using a model $F(x)$, 
we want to minimize a loss function $L(y, F(x))$. Each step adds a new estimator $h(x)$ such that:
$$F_{m+1}(x) = F_m(x) + \gamma h_m(x)$$
Where:
$F_m(x)$ is the current model.
$h_m(x)$ is the new weak learner.
$\gamma$ is the Learning Rate.

