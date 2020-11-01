# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 11:19:26 2020

@author: Mylène
"""

import numpy as np

def evaluate_model(Y_test,y_pred):
    epsilon = 0.0001
    TP = np.sum(np.multiply((1*np.array(Y_test==1)),(1*np.array(y_pred==1))))
    FP = np.sum(np.multiply((1*np.array(Y_test==0)),(1*np.array(y_pred==1))))
    TN = np.sum(np.multiply((1*np.array(Y_test==0)),(1*np.array(y_pred==0))))
    FN = np.sum(np.multiply((1*np.array(Y_test==1)),(1*np.array(y_pred==0))))
    accuracy = (TP+TN)/(TP+FP+TN+FN)
    recall = TP/(TP+FN+epsilon)
    precision = TP/(TP+FP+epsilon)
# Calculate F_score using beta=0.5 and correct values for precision and recall
    f_score = (1 + 0.5*0.5)*precision*recall/((0.5*0.5*precision + recall)+epsilon)
    return([accuracy,f_score])
