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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from itertools import repeat
import pickle
from tqdm import tqdm
from sklearn import preprocessing

TIMESTRING  = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())

writing_dir = '/home/zhihuan/Documents/20180907_Cong_Feng/20180908_Hypoxemia/Results'
data = pd.read_csv("/home/zhihuan/Documents/20180907_Cong_Feng/20180908_Hypoxemia/data/data_20180925.csv", header=0, index_col=0)
#colnames = list(data.columns.values)

colnames = list(data.columns.values)


X, y = data.iloc[:, np.r_[0:4, 25:28, 46:49]], data.iloc[:,-1]
#X, y = data.iloc[:, np.r_[0:4, 46:49]], data.iloc[:,-1]
variableNames = list(X.columns.values)
# find rows without NaN values
rowsToKeep = np.invert(pd.isnull(X).any(axis=1).values)

X = X.loc[rowsToKeep, ]
y = y.loc[rowsToKeep, ]
y.loc[y == 1, ] = 0
y.loc[y == 2, ] = 1

# Normalize data
X = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(X_scaled)

# =============================================================================
# PCA
# =============================================================================
#pca = PCA(n_components=2).fit_transform(X)
#
#plt.figure(dpi=120)
#plt.scatter(pca[y.values==0,0], pca[y.values==0,1], alpha=0.5, label='No hypoxemia', s=16, color='navy')
#plt.scatter(pca[y.values==1,0], pca[y.values==1,1], alpha=0.5, label='Has hypoxemia', s=16, color='darkorange')
#plt.legend()
#plt.title('PCA Result: 7 features in 2D view and their labels')
#plt.xlabel('PC1')
#plt.ylabel('PC2')
#plt.gca().set_aspect('equal')
#plt.show()




methods = ['logit_l1', 'logit_l2', 'SVM_linear', 'RF', 'AdaBoost', 'GBM', 'NN_l2']

roc_auc_10_times = np.zeros([len(methods),10])
f1_10_times = np.zeros([len(methods),10])
average_precision_10_times = np.zeros([len(methods),10])
average_recall_10_times = np.zeros([len(methods),10])
coef_all_10_times = np.zeros((10, X.shape[1]))

fivefold_indices = [[] for x in repeat(None, 51)]
fivefold_indices[0] = ['experiment', 'fold'] + list(X.index)
y_test_all = [[] for x in repeat(None, 10*len(methods))]
y_pred_proba_all = [[] for x in repeat(None, 10*len(methods))]


for i in tqdm(range(10)):
    for mtd_idx, mtd in enumerate(methods):
#        print("***Current using method:", mtd)
        if mtd == "logit_l1":
            regr = linear_model.LogisticRegression(penalty='l1', C=26)
        if mtd == "logit_l2":
            regr = linear_model.LogisticRegression(penalty='l2', C=10)
        if mtd == "SVM_linear":
            regr = svm.SVC(kernel='linear', C=1, probability = True)# Linear
        if mtd == "RF":
            regr = RandomForestClassifier(n_estimators=300)
        if mtd == "AdaBoost":
            regr = AdaBoostClassifier(n_estimators=100)
        if mtd == "GBM":
            regr = GradientBoostingClassifier(n_estimators=100)
        if mtd == "NN_l2":
            regr = MLPClassifier(solver='adam', alpha=1e-5, # alpha is L2 reg
                                 hidden_layer_sizes=(8), random_state=1)
        #regr = svm.SVC(kernel='rbf', C=1)# RBF
        #regr = svm.SVC(kernel='poly', C=1, degree=2)
        #regr = svm.SVC(kernel='poly', C=1, degree=3)
        
        n_splits = 5
        kf = KFold(n_splits=n_splits, random_state=None, shuffle=True)
        roc_auc = []
        f1_all = []
        average_precision = []
        average_recall = []
        coef_all = np.zeros((n_splits, X.shape[1]))
        idx = -1
        
        y_test_all[mtd_idx*10 + i] = [mtd]
        y_pred_proba_all[mtd_idx*10 + i] = [mtd]
        
        
        for train_index, test_index in kf.split(X):
            idx += 1
            
#            fold_detail = ['train'] * len(data)
#            for j in test_index:
#                fold_detail[j] = 'test'
#            fivefold_indices[i*5+idx +1] = [i+1, idx+1] + fold_detail
            
            X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
            y_train, y_test = y.iloc[train_index,], y.iloc[test_index,]
        #    y2_train, y2_test = data.iloc[train_index,-2], data.iloc[test_index,-2]
            regr.fit(X_train, y_train)
            y_pred = regr.predict(X_test)
            y_pred_proba = regr.predict_proba(X_test)[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            y_test_all[mtd_idx*10 + i].append(list(y_test))
            y_pred_proba_all[mtd_idx*10 + i].append(list(y_pred_proba))
            
            
            roc_auc.append(auc(fpr, tpr))
            
            if mtd == "logit_l1":
                coef_all[idx,:] = regr.coef_
            
            
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
            #https://towardsdatascience.com/what-metrics-should-we-use-on-imbalanced-data-set-precision-recall-roc-e2e79252aeba
            #With a large number of negative samples — precision is probably better
            average_precision.append(precision_score(y_test, y_pred))
            average_recall.append(recall_score(y_test, y_pred))
            
            f1 = f1_score(y_test, y_pred)
            f1_all.append(f1)
        
        roc_auc_averaged = np.mean(roc_auc)
        f1_averaged = np.mean(f1_all)
#        print('Average AUC: %.4f' % roc_auc_averaged)
#        print('Average F1: %.4f' % f1_averaged)
#        print('Average precision score: {0:0.2f}'.format(np.mean(average_precision)))
#        print('Average recall score: {0:0.2f}'.format(np.mean(average_recall)))
        
        if mtd == "logit_l1":
            coef_mean = np.mean(coef_all, axis = 0)
            coef_all_10_times[i,:] = coef_mean
        #np.sort(coef_mean)
    
        roc_auc_10_times[mtd_idx, i] = roc_auc_averaged
        f1_10_times[mtd_idx, i] = f1_averaged
        average_precision_10_times[mtd_idx, i] = np.mean(average_precision)
        average_recall_10_times[mtd_idx, i] = np.mean(average_recall)
        
        
final_roc_auc = np.nanmean(roc_auc_10_times, axis = 1) # ignore nan
final_f1 = np.nanmean(f1_10_times, axis = 1) # ignore nan
final_ave_precision = np.nanmean(average_precision_10_times, axis = 1) # ignore nan
final_ave_recall = np.nanmean(average_recall_10_times, axis = 1) # ignore nan


print(methods)
print("AUC:", ["%.3f" % v for v in final_roc_auc])
print("F1: ", ["%.3f" % v for v in final_f1])
print("P:  ", ["%.3f" % v for v in final_ave_precision])
print("R:  ", ["%.3f" % v for v in final_ave_recall])

coef_mean = np.mean(coef_all_10_times, axis = 0)
sort_index = np.argsort(coef_mean).reshape(-1)# + 1
sort_index_nonzero = sort_index[np.sort(coef_mean) != 0]

param_nonzero_values = [coef_mean[index] for index in sort_index_nonzero]
param_nonzero_names = [colnames[index] for index in sort_index_nonzero]
param = np.transpose(np.matrix([param_nonzero_values, param_nonzero_names]))
param_df = pd.DataFrame(param)
param_df.to_csv(writing_dir + '/LogisticRegression_L1_parameters.csv', index = False, header = False)

# Save data
pickle.dump( y_test_all, open(writing_dir + '/y_test_logit_RF_SVM.pickle', "wb" ) )
pickle.dump( y_pred_proba_all, open(writing_dir + '/y_pred_proba_logit_RF_SVM.pickle', "wb" ) )


result = [list(final_roc_auc), list(final_f1), list(final_ave_precision), list(final_ave_recall)]
result = pd.DataFrame(result)
result = result.transpose()
result = pd.concat([pd.DataFrame(['AUC', 'F1', 'P', 'R']).T, result])
result.insert(loc=0, column='Methods', value=([''] + methods))
result.to_csv(writing_dir + '/results.csv', index = False, header = False)
