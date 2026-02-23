# Gradient Boosting

Gradient Boosting is considered the most powerful algorithm in the world of machine learning. 
It is an algorithm that combines multiple weak learners to create a strong learner. 
The idea behind Gradient Boosting is to build the model in a stage-wise fashion, where each stage tries to correct 
the errors of the previous stage.

## The "Correction" Strategy
Unlike traditional models that try to find the perfect answer in one go, Gradient Boosting builds the solution incrementally.
1. Start with a Rough Guess: The model starts with a very simple baseline—usually just the average value of the target you're trying to predict.
2. Calculate the Error (The Residuals): The model looks at where it was wrong. These errors are called "residuals."
3. Train a Weak Learner: A new, small decision tree is trained. However, it isn't trained to predict the final answer; it is trained specifically to predict the residuals (the mistakes of the previous stage).
4. Update the Model: The prediction from this new tree is added to the previous prediction.
5. Repeat: This process repeats hundreds or thousands of times. Each new tree focuses on the "leftover" error from the combined ensemble of all previous trees.

## The "Gradient" Aspect
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
$\gamma$ is the Learning Rate (a small multiplier to prevent overshooting).

