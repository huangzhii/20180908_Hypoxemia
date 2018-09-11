#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 11:03:04 2018

@author: zhihuan
"""


import logging, time, os, argparse
import pandas as pd
import numpy as np
from sklearn import preprocessing, datasets, linear_model
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier




data = pd.read_csv("/home/zhihuan/Documents/20180907_Cong_Feng/20180908_Hypoxemia/data/vitalsign.all.csv", header=0, index_col=0)
#colnames = list(data.columns.values)

colnames = list(data.columns.values)


X, y = data.iloc[:,1:8], data.iloc[:,-1]

pca = PCA(n_components=2).fit_transform(X)

plt.figure(dpi=120)
plt.scatter(pca[y.values==0,0], pca[y.values==0,1], alpha=0.5, label='No hypoxemia', s=16, color='navy')
plt.scatter(pca[y.values==1,0], pca[y.values==1,1], alpha=0.5, label='Has hypoxemia', s=16, color='darkorange')
plt.legend()
plt.title('PCA Result: 7 features in 2D view and their labels')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.gca().set_aspect('equal')
plt.show()




#regr = RandomForestClassifier(n_estimators=10)
regr = linear_model.LogisticRegression(penalty='l1', C=10)
#regr = linear_model.LogisticRegression(penalty='l2', C=10)
# regr = svm.SVC(kernel='linear', C=1)# Line
        
n_splits = 5
kf = KFold(n_splits=n_splits, random_state=123, shuffle=True)
roc_auc = []
f1_all = []
average_precision = []
average_recall = []
coef_all = np.zeros((n_splits, 7))
idx = -1
for train_index, test_index in kf.split(data):
    idx += 1
    X_train, X_test = data.iloc[train_index,1:8], data.iloc[test_index,1:8]
    y_train, y_test = data.iloc[train_index,-1], data.iloc[test_index,-1]
#    y2_train, y2_test = data.iloc[train_index,-2], data.iloc[test_index,-2]
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    coef_all[idx,:] = regr.coef_
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    #https://towardsdatascience.com/what-metrics-should-we-use-on-imbalanced-data-set-precision-recall-roc-e2e79252aeba
    #With a large number of negative samples — precision is probably better
    average_precision.append(precision_score(y_test, y_pred))
    average_recall.append(recall_score(y_test, y_pred))
    
    f1 = f1_score(y_test, y_pred)
    f1_all.append(f1)
    
    # =============================================================================
    # ROC
    # =============================================================================
    y_pred_proba = regr.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc.append(auc(fpr, tpr))
    
    # =============================================================================
    # Precision Recall Curve
    # =============================================================================
#    plt.step(recall, precision, color='b', alpha=0.2,
#             where='post')
#    plt.fill_between(recall, precision, step='post', alpha=0.2,
#                     color='b')
#    
#    plt.xlabel('Recall')
#    plt.ylabel('Precision')
#    plt.ylim([0.0, 1.05])
#    plt.xlim([0.0, 1.0])
#    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
#              precision_score(y_test, y_pred)))
#    
    # =============================================================================
    # ROC curve
    # =============================================================================
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    


roc_auc_averaged = np.mean(roc_auc)
f1_averaged = np.mean(f1_all)
print('Average AUC: %.4f' % roc_auc_averaged)
print('Average F1: %.4f' % f1_averaged)
print('Average precision score: {0:0.2f}'.format(np.mean(average_precision)))
print('Average recall score: {0:0.2f}'.format(np.mean(average_recall)))
        
coef_mean = np.mean(coef_all,0)
print(colnames[1:8])