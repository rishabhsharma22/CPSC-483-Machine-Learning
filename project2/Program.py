# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 01:05:19 2018

@author: Rishabh Sharma
"""

import numpy as np
from sklearn import tree 
import pandas as pd
import os
import io
import numpy as np
import pandas as pd
from pandas import DataFrame
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('Dataset.csv', header = None, skipinitialspace=True)
df.columns = ["age", "class", "education", "years", "married", "occupation", "relationship", "gender","gain", "loss","hours", "income"]

wc = {'Private' : 0, 'Self-emp-not-inc' : 1, 'Self-emp-inc' : 2, 'Federal-gov' : 3, 'Local-gov' : 4, 'State-gov' : 5, 'Without-pay' : 6, 'Never-worked' : 7}
df['class'] = df['class'].map(wc)

el = {'Bachelors' : 0, 'Some-college' : 1, '11th' : 2, 'HS-grad' : 3, 'Prof-school' : 4, 'Assoc-acdm' : 5, 'Assoc-voc' : 6, '9th' : 7, '7th-8th' : 8, '12th' : 9, 'Masters' : 10, '1st-4th' : 11, '10th' : 12, 'Doctorate' : 13, '5th-6th' : 14, 'Preschool' : 15}
df['education'] = df['education'].map(el)

ms = {'Married-civ-spouse' : 0, 'Divorced' : 1, 'Never-married' : 2, 'Separated' : 3, 'Widowed' : 4, 'Married-spouse-absent' : 5, 'Married-AF-spouse' : 6}
df['married'] = df['married'].map(ms)

oc = {'Tech-support' : 0, 'Craft-repair' : 1, 'Other-service' : 2, 'Sales' : 3, 'Exec-managerial' : 4, 'Prof-specialty' : 5, 'Handlers-cleaners' : 6, 'Machine-op-inspct' : 7, 'Adm-clerical' : 8, 'Farm-fishing' : 9, 'Transport-moving' : 10, 'Priv-house-serv' : 11, 'Protective-serv' : 12, 'Armed-Forces' : 13}
df['occupation'] = df['occupation'].map(oc)

rp = {'Wife' : 0, 'Own-child' : 1, 'Husband' : 2, 'Not-in-family' : 3, 'Other-relative' : 4, 'Unmarried' : 5}
df['relationship'] = df['relationship'].map(rp)

gd = {'Female' : 0, 'Male' : 1}
df['gender'] = df['gender'].map(gd)

ic = {'>50K' : 0, '<=50K' : 1}
df['income'] = df['income'].map(ic)

df = df.dropna()

df.head()

features = list(df.columns[:11])

y = df['income']
X = df[features]

pd.DataFrame(X).fillna(X.mean(), inplace = True)
pd.DataFrame(y).fillna(y.mean(), inplace = True)
clf = tree.DecisionTreeClassifier(random_state = 50,max_depth=3, min_samples_leaf=5)
clf = clf.fit(X,y)

from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf, out_file = dot_data, feature_names = features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
