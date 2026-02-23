# Random Forest

The normal CART algorithm is a powerful algorithm but it can be prone to overfitting. 

In order to combat this problem, the random forest algorithm uses the following 2 techniques:

## Bagging
Instead of giving every tree the entire dataset, each tree is trained on a 
"Bootstrap" sample ie. a random subset of the data (with replacement). 
This means some rows are repeated, and some are left out.

## Random feature selection
In a normal CART, the model considers every feature for a split. 
In a Random Forest, each tree is only allowed to see a random subset of features at each split.

in summary the random forest creates a lot of different trees by using different subsets of the data and different subsets of the features.
in order to make a decision, the random forest takes the majority vote of all the trees for classification
or the mean of all predicted values for Regression.

This type of algorithm is called an ensemble learning algorithm because it 
combines the predictions of multiple models to make a final prediction.