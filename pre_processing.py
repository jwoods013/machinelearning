#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# setup the independent and dependent variables
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print values
print(x)
print(y)

# replace missing values with mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# print values after replacing missing values
print(x)

#encoding independant using onehotencoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

print(x)

#encoding dependant variable

le = LabelEncoder()
y = le.fit_transform(y)

print(y)