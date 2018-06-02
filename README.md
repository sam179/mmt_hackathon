# make_my_trip_hackathon

This solution obtained a prediction accuracy of 89.86%. 

## Problem

Binary Classification of the given dataset. 


## Dataset

- The data contains 16 columns (excluding the index), with 9 columns of categorical and 6 columns of real valued features. 
- Problem becomes tricky because of the insufficient number of training samples, ie ~550.

## Preprocessing

- Firstly, I replaced all the missing categorical data with the most frequently occuring data of their respective individual feature. Looking at it retrospectively, substituting the data with a more neutral value may have benefitted, as the lack of training data might have added some inaccurate bias. 

- Next is the most obvious step of replacing missing real valued data with the mean of their respective feature.

- One hot encoding all the categorical variables and separating training data and testing data.

## Training the model and cross-validation

- With the irregularly low amount of training samples, I ran a variation of Xgboost. 
- This was followed by a K fold cross validation to determine the approximate performance of the model. This was accomplished by exploration of several values. 
- Finally optimised the hyper-parameters of xgboost using GridSearch algorithm. Hyperparameters were optimised in the order max_depth, min_child_weight, gamma, reg_alpha, n_estimators, and seed value to add some bias.





