#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# setup the independent and dependent variables
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print values
print(x)
print(y)

# setup sklearn
from sklearn.impute import SimpleImputer

# replace missing values with mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# print values after replacing missing values
print(x)