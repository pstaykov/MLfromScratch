# Linear Regression

The linear regression model is, as the name suggests, 
a regression model that fits a linear relationship between the independent 
variable and the dependent variable. It is used to predict numeric values. 
The formula for linear regression is y = mx + c. 
We implement it by iterating a certain number of times. For each iteration, 
we calculate the mean squared error and update the weights accordingly. The formula for the MSE is the following:
loss = (predicted - actual)^2 / 2m
where predicted is the value predicted by the model, actual is the true value, and m is the number of samples.
The value by which the weights are updated is called the gradient. It is calculated by the following formula:
gradientW = loss * x
gradientB = loss

Where predicted is the value predicted by the model, actual is the true value, and x is the input feature.

In L2 regularization, we add a penalty term to the loss function that penalizes large weights.
The formula for the penalty term is the following:
penalty = lambda * sum(w^2)

Thus the following GradientW formula is:
gradientW = loss * x + lambda * w
Where lambda is the regularization parameter, and w is the weight vector.