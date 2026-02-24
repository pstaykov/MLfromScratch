### Gradient Boosting (MSE Regression)

**1. Initialization**
Initialize the model with a constant value (the mean of the targets):
$$F_0(x) = \text{mean}(y)$$

**2. Iterative Optimization (for $m = 1 \dots M$)**

* **Compute Negative Gradient (Pseudo-Residuals):**
  For a Loss Function $L(y, F(x)) = \frac{1}{2}(y - F(x))^2$, the negative gradient is:
  $$r_{im} = - \left[ \frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)} \right] = y_i - F_{m-1}(x_i)$$

* **Train Weak Learner:**
  Fit a decision tree $h_m(x)$ using the original features $X$ to predict the current residuals $r_{im}$:
  $$h_m = \text{fit}(X, r_{im})$$

* **Update Model:**
  Add the new tree to the ensemble, scaled by the learning rate $\nu$:
  $$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$

**3. Final Prediction**
$$y_{pred} = F_M(x)$$