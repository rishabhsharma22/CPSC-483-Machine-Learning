# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 19:48:34 2018

@author: Rishabh Sharma
"""

# CPSC 483_Machine_Learning. Neural Network Implementation.
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import pandas as pd
import numpy as np

dataset = pd.read_csv('file:///C:/CSUF/CPSC483/Assignments/Dataset.csv')

X = dataset.iloc[:,:4].values
Y = dataset.iloc[:,4:].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

classifier = Sequential()

classifier.add(Dense(output_dim = 4,init = 'uniform', activation = 'linear', input_dim = 4))
#classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(output_dim = 2,init = 'uniform', activation = 'linear'))

#classifier.add(Dense(output_dim = 2,init = 'uniform', activation = 'linear'))
classifier.add(Dense(output_dim = 1,init = 'uniform', activation = 'linear'))
classifier.compile(optimizer = 'RMSprop', loss = 'mean_squared_error', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 1000)

y_pred = classifier.predict(X_test)

print("Mean squared error: %.4f" % np.mean((y_pred - y_test) ** 2))