#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 19:26:36 2020

@author: manpreetsaluja
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

data=pd.read_csv('/Users/manpreetsaluja/Downloads/Credit_Card-Risk-assessment-master/Credit_default_dataset.csv')
data.head(100)
data.info()

#dont need id column
data.iloc[:1]
data=data.drop(["ID"],axis=1)

#correcting the column name making it sequenctial
data.info()
data.rename(columns={'PAY_0':'PAY_1'},inplace=True)
data.info()

#mapping with values 
data["EDUCATION"].value_counts()
data["EDUCATION"]=data["EDUCATION"].map({0:4,1:1,2:2,3:3,4:4,5:4,6:4})
data["EDUCATION"].value_counts()

data["MARRIAGE"]=data["MARRIAGE"].map({0:3,1:1,2:2,3:3})
data["MARRIAGE"].value_counts()

#scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=data.drop(["default.payment.next.month"],axis=1)
X=sc.fit_transform(X)

X

y=data["default.payment.next.month"]

#willbe using xgboost for this problem statement 

#hyper parameter 
param={
       "learning_rate":[0.01,0.05,0.10,0.15,0.20,0.25,0.30],
           "max_depth":[2,4,6,8,10,12,14],
               "min_child_weight":[1,3,5,7,9],
                   "gamma":[0.0,0.1,0.2,0.3,0.4],
                       "colsample_bytree":[0.4,0.5,0.6,0.7,0.8]
       }

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import xgboost
classifier=xgboost.XGBClassifier()

random_search=RandomizedSearch(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,verbose=3)

random_search.best_estimator_

random_search.best_params_

classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.4, gamma=0.1, learning_rate=0.25,
       max_delta_step=0, max_depth=3, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)

from sklearn.model_selection import cross_val_score

score=cross_val_score(classifier,X,y,cv=10)


score


score.mean()