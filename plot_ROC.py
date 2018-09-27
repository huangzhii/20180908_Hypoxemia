#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 22:22:37 2018

@author: zhihuan
"""


import pickle
from itertools import chain
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt 


writing_dir = '/home/zhihuan/Documents/20180907_Cong_Feng/20180908_Hypoxemia/Results'

y_pred_proba_logit_RF_SVM = pickle.load( open( writing_dir+"/y_pred_proba_logit_RF_SVM.pickle", "rb" ) )
y_test_logit_RF_SVM = pickle.load( open( writing_dir+"/y_test_logit_RF_SVM.pickle", "rb" ) )
#y_pred_proba_NN = pickle.load( open( writing_dir+"/y_pred_proba_NN.pickle", "rb" ) )
#y_test_NN = pickle.load( open( writing_dir+"/y_test_NN.pickle", "rb" ) )


#proba_sorted = np.zeros([10,58])
#for i in range(10):
#    print(i)
#    # logit l1
#    proba = list(chain.from_iterable(y_pred_proba_logit_RF_SVM[i][1:]))
#    gt = list(chain.from_iterable(y_test_logit_RF_SVM[i][1:]))
#    fpr, tpr, thresholds = roc_curve(gt, proba)
#    print(auc(fpr, tpr))
#    proba_sorted[i,:] = np.asarray(proba)[list(np.argsort(gt))] 
#    
#gt_sorted = np.sort(gt)
#proba = proba_sorted[9]
#fpr, tpr, thresholds = roc_curve(gt_sorted, proba)
#auc(fpr, tpr)

plt.figure()
lw = 2
# =============================================================================
# Plot models
# =============================================================================
i = 9
#Logistic Regression L1
proba = list(chain.from_iterable(y_pred_proba_logit_RF_SVM[i][1:]))
gt = list(chain.from_iterable(y_test_logit_RF_SVM[i][1:]))
fpr, tpr, thresholds = roc_curve(gt, proba)
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='Logistic Regression (L1) (area = %0.4f)' % auc(fpr, tpr))
#Logistic Regression L2
proba = list(chain.from_iterable(y_pred_proba_logit_RF_SVM[10+i][1:]))
gt = list(chain.from_iterable(y_test_logit_RF_SVM[10+i][1:]))
fpr, tpr, thresholds = roc_curve(gt, proba)
plt.plot(fpr, tpr, color='blue',
         lw=lw, label='Logistic Regression (L2) (area = %0.4f)' % auc(fpr, tpr))
#SVM
proba = list(chain.from_iterable(y_pred_proba_logit_RF_SVM[20+i][1:]))
gt = list(chain.from_iterable(y_test_logit_RF_SVM[20+i][1:]))
fpr, tpr, thresholds = roc_curve(gt, proba)
plt.plot(fpr, tpr, color='green',
         lw=lw, label='SVM (Linear) (area = %0.4f)' % auc(fpr, tpr))
#Decision Tree
proba = list(chain.from_iterable(y_pred_proba_logit_RF_SVM[30+i][1:]))
gt = list(chain.from_iterable(y_test_logit_RF_SVM[30+i][1:]))
fpr, tpr, thresholds = roc_curve(gt, proba)
plt.plot(fpr, tpr, color='red',
         lw=lw, label='Random Forest (area = %0.4f)' % auc(fpr, tpr))

#Adaboost
proba = list(chain.from_iterable(y_pred_proba_logit_RF_SVM[40+i][1:]))
gt = list(chain.from_iterable(y_test_logit_RF_SVM[40+i][1:]))
fpr, tpr, thresholds = roc_curve(gt, proba)
plt.plot(fpr, tpr, color='cyan',
         lw=lw, label='Adaboost (area = %0.4f)' % auc(fpr, tpr))
#GBM
proba = list(chain.from_iterable(y_pred_proba_logit_RF_SVM[50+i][1:]))
gt = list(chain.from_iterable(y_test_logit_RF_SVM[50+i][1:]))
fpr, tpr, thresholds = roc_curve(gt, proba)
plt.plot(fpr, tpr, color='magenta',
         lw=lw, label='GBM (area = %0.4f)' % auc(fpr, tpr))
#NN(L2)
proba = list(chain.from_iterable(y_pred_proba_logit_RF_SVM[60+i][1:]))
gt = list(chain.from_iterable(y_test_logit_RF_SVM[60+i][1:]))
fpr, tpr, thresholds = roc_curve(gt, proba)
plt.plot(fpr, tpr, color='black',
         lw=lw, label='Neural Network (L2) (area = %0.4f)' % auc(fpr, tpr))

# =============================================================================
# Plot finished
# =============================================================================
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()
