
# Importing libraries
import numpy as np
import pandas as pd

# Importing the dataset
training_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
y = training_data.iloc[:, 16].values
training_data = training_data.iloc[:, :16]


# Preprocessing the data
data_list = []
data_list.append(training_data)
data_list.append(test_data)
training_data = pd.concat(data_list)


# replacing missing categorical data with the most frequent data
training_data['A'] = training_data['A'].fillna(training_data['A'].value_counts().index[0]) 
training_data['D'] = training_data['D'].fillna(training_data['D'].value_counts().index[0])
training_data['E'] = training_data['E'].fillna(training_data['E'].value_counts().index[0])
training_data['F'] = training_data['F'].fillna(training_data['F'].value_counts().index[0])
training_data['G'] = training_data['G'].fillna(training_data['G'].value_counts().index[0])
training_data['I'] = training_data['I'].fillna(training_data['I'].value_counts().index[0])
training_data['J'] = training_data['J'].fillna(training_data['J'].value_counts().index[0])
training_data['L'] = training_data['L'].fillna(training_data['L'].value_counts().index[0])
training_data['M'] = training_data['M'].fillna(training_data['M'].value_counts().index[0])
X = training_data.iloc[:, 1:16].values
X_num = X[:, [1, 2, 7, 10, 13, 14]]



# replacing missing numerical values with mean values of each columns
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_num)
X[:, [1, 2, 7, 10, 13, 14]] = imputer.transform(X_num)


# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:, [1, 2, 7, 10, 13, 14]] = sc.fit_transform(X[:, [1, 2, 7, 10, 13, 14]])


# One-hot encoding categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoders = [LabelEncoder() for i in range(9)]
X[:, 0] = label_encoders[0].fit_transform(X[:, 0])
X[:, 3] = label_encoders[1].fit_transform(X[:, 3])
X[:, 4] = label_encoders[2].fit_transform(X[:, 4])
X[:, 5] = label_encoders[3].fit_transform(X[:, 5])
X[:, 6] = label_encoders[4].fit_transform(X[:, 6])
X[:, 8] = label_encoders[5].fit_transform(X[:, 8])
X[:, 9] = label_encoders[6].fit_transform(X[:, 9])
X[:, 11] = label_encoders[7].fit_transform(X[:, 11])
X[:, 12] = label_encoders[8].fit_transform(X[:, 12])
onehotencoder = OneHotEncoder(categorical_features = [0, 3, 4, 5, 6, 8, 9, 11, 12])
X = onehotencoder.fit_transform(X).toarray()


# separating training and testing data
X_train = X[:552, :]
X_test = X[552:, :]


# Creating an xgboost model and finding the prediction
from xgboost import XGBClassifier
classifier = XGBClassifier(max_depth=3, min_child_weight=5, gamma = 0.0, subsample=0.8, colsample_bytree=0.6, reg_alpha=0.005, n_estimators=256, seed = 27)
classifier.fit(X_train, y)
y_pred = classifier.predict(X_test)


# K-cross-validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y, cv = 20)
print (accuracies.mean())
print (accuracies.std())
print (y_pred)


# Appying grid search on each parameter of model in turns to find the optimal values of those parameters
from sklearn.model_selection import GridSearchCV
param_test6 = {
 'reg_lambda':[1e-2,0,0.0985, 0.09851, 0.099]
}
gsearch1 = GridSearchCV(estimator = XGBClassifier(reg_alpha = 0.005, max_depth=3,
 min_child_weight=5, gamma=0.0, subsample=0.8, colsample_bytree=0.6,
 objective= 'binary:logistic', scale_pos_weight=1,seed=27), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=-1,iid=False, cv=5)
gsearch1 = gsearch1.fit(X_train, y)
best_val = gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
print (best_val)


# Creating the submission
file = open("submission.csv","w")
file.write("id,P\n")
for i in range(0,y_pred.size):
        file.write(str(i+553))
        file.write(',')
        file.write(str(y_pred[i])+'\n')

