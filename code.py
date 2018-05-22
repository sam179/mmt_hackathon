

import numpy as np
import pandas as pd


training_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#training_data[0, 3, 4, 5, 6, 8, 9, 11, 12] = training_data[0, 3, 4, 5, 6, 8, 9, 11, 12].fillna(training_data[0, 3, 4, 5, 6, 8, 9, 11, 12].value_counts.index.values[0])
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
y = training_data.iloc[:, 16].values
X_num = X[:, [1, 2, 7, 10, 13, 14]]
#print (X[35])

#np.set_printoptions(precision=3, suppress=True)
#for i in range(0, X_num.shape[1]-1):
#   print(X_num[i])
#   print(X_num[i].shape)
#   np.genfromtxt(np.array(X_num[:,i]))

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_num)
X[:, [1, 2, 7, 10, 13, 14]] = imputer.transform(X_num)

print (X[56])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:, [1, 2, 7, 10, 13, 14]] = sc.fit_transform(X[:, [1, 2, 7, 10, 13, 14]])

print (X[56])
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

print (X.shape)
