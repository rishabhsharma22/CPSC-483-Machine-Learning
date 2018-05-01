
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 09:48:12 2018

@author: Rishabh Sharma
"""

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,12)


file_name = "Dataset.csv"

data = pd.read_csv(file_name)


X = data.iloc[:,0:-1]
y = data.iloc[:, -1]


from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

lr = LinearRegression()
lr.fit(X_train,y_train)

preds = lr.predict(X_test)

print("The coeffecients/weight matrix for the linear model:", lr.coef_)

print("Mean squared error: %.2f" % np.mean((lr.predict(X_test) - y_test) ** 2))

print('Variance score: %.2f' % lr.score(X_test, y_test))

    

