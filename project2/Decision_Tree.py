# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 21:31:43 2018

@author: Rishabh Sharma
"""

import os
import numpy as np
import pandas as pd
import pdb

import matplotlib.pyplot as plt
import seaborn as sns
sns.axes_style("darkgrid")

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv("Dataset.csv")


X = data.iloc[:,0:-1]
y = data.iloc[:, -1]




