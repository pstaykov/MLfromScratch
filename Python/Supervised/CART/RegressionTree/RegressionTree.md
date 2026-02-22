# Regression Tree
The CART algorithm can also be used for regression tasks. In this case, the algorithm 
stays the same, but instead of using Gini Impurity, it uses Mean Squared Error (MSE) to determine the best split point. 
The algorithm works as follows:
1. Choose the best feature and split point.
2. Recursively build the left and right subtrees.
3. Calculate the MSE for each subtree.
4. Choose the feature and split point that results in the lowest MSE.
5. The prediction for a leaf node is the mean of the target values of the samples in that node.

# MSE
the MSE is calculated as follows:
MSE = (1/n) * sum((y_i - y_pred)^2)

where:
- n is the number of samples in the node
- y_i is the true target value of the i-th sample
- y_pred is the predicted target value for the node, which is the mean of the target values of the samples in that node.

