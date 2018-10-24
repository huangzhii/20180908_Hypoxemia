#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 00:06:08 2018

@author: zhihuan
"""


import logging, time, os, argparse
import pandas as pd
import numpy as np
from sklearn import preprocessing, datasets, linear_model
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn import preprocessing

#data = pd.read_csv("/home/zhihuan/Downloads/Hypoxemia - LSTM/PO2data/PO2数据/expanded.all.data.merged.imputed.calculated.shrinked.csv", header=0)
#data = data.drop(columns=['INTIME', 'OUTTIME', 'CURR_TIME'])
min_max_scaler = preprocessing.MinMaxScaler()
n_splits = 5
kf = KFold(n_splits=n_splits, random_state=None, shuffle=True)
data.columns.values

data.ix[data.ix[:,"GENDER"] == "M", "GENDER"] = 0
data.ix[data.ix[:,"GENDER"] == "F", "GENDER"] = 1
data.ix[data.ix[:,"HYPOXEMIA_CLASS"] == "Normal", "HYPOXEMIA_CLASS"] = 0
data.ix[data.ix[:,"HYPOXEMIA_CLASS"] == "Mild", "HYPOXEMIA_CLASS"] = 1
data.ix[data.ix[:,"HYPOXEMIA_CLASS"] == "Morderate", "HYPOXEMIA_CLASS"] = 2
data.ix[data.ix[:,"HYPOXEMIA_CLASS"] == "Severe", "HYPOXEMIA_CLASS"] = 3
coef_all = np.zeros((n_splits, 66))

regr = linear_model.LogisticRegression(penalty='l1', C=10)

idx = -1
for train_index, test_index in kf.split(data):
    idx += 1
    X_train, X_test = data.iloc[train_index, ], data.iloc[test_index, ]
    X_train = X_train.ix[:, data.columns.difference(["HYPOXEMIA_CLASS"])]
    X_test = X_test.ix[:, data.columns.difference(["HYPOXEMIA_CLASS"])]
    colnames = X_train.columns.values
    X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train.values))
    X_test = pd.DataFrame(min_max_scaler.fit_transform(X_test.values))

    
    
    
    
    y_train, y_test = data.iloc[train_index, 36], data.iloc[test_index, 36]
    
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    y_pred_proba = regr.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = auc(fpr, tpr)
    f1 = f1_score(y_test, y_pred, average="macro")
#    coef_all[idx, :] = regr.coef_
    print("here")
    
    
    
coef_mean = np.mean(regr.coef_, axis = 0)
sort_index = np.argsort(coef_mean).reshape(-1)# + 1
sort_index_nonzero = sort_index[np.sort(coef_mean) != 0]

param_nonzero_values = [coef_mean[index] for index in sort_index_nonzero]
param_nonzero_names = [colnames[index] for index in sort_index_nonzero]
param = np.transpose(np.matrix([param_nonzero_values, param_nonzero_names]))
param_df = pd.DataFrame(param)
param_df.to_csv('LogisticRegression_L1_parameters.csv', index = False, header = False)

    
    
    
    
    
    