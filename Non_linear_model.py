# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 10:13:55 2018

@author: Rishabh Sharma
"""

import pandas as pd #Pands is the library for loading and working with datafiles.
import numpy as np # To perform matrix operations
import sys
import matplotlib.pyplot as plt # for ploting data
import seaborn as sns # for ploting data with various flexible functions/ uses matplotlib at back
plt.rcParams["figure.figsize"] = (12,12)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression


filename = 'Dataset.csv'
data = pd.read_csv(filename)

x = data.iloc[:,0:-1]
y = data.iloc[:,-1]


x = np.asarray(x)
y = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

poly = PolynomialFeatures(degree = 2)
X = poly.fit_transform(X_train)

x_test = poly.fit_transform(X_test)

clf1 = LinearRegression()
clf1.fit(X, y_train)
preds = clf1.predict(x_test)


print("The coeffecients/weight matrix for the non linear model with degree 2:", clf1.coef_)

print("Mean squared error: %.4f" % np.mean((preds - y_test) ** 2))

print('Variance score: %.2f' % clf1.score(x_test, y_test))



poly = PolynomialFeatures(degree = 4)
X = poly.fit_transform(X_train)

x_test = poly.fit_transform(X_test)

clf = LinearRegression()
clf.fit(X, y_train)
preds = clf.predict(x_test)


print("The coeffecients/weight matrix for the non linear model with degree 4:", clf.coef_)

print("Mean squared error: %.4f" % np.mean((preds - y_test) ** 2))

print('Variance score: %.2f' % clf.score(x_test, y_test))

