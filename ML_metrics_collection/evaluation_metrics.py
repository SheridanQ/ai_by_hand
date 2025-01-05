"""
This script collects some ML evaluation metrics.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def MSE(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)


def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))


def MedAE(y_true, y_pred):
    """https://www.kaggle.com/competitions/playground-series-s3e25"""
    return np.median(np.abs(y_true-y_pred))


def RMSE(y_true, y_pred):
    """https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality"""
    return np.sqrt(np.mean((y_true-y_pred)**2))


def RMSLE(y_true, y_pred):
    """https://www.kaggle.com/competitions/store-sales-time-series-forecasting"""
    return np.sqrt(np.mean((np.log1p(y_pred)-np.log1p(y_true))**2))


def log_loss(y_true, y_pred, eps=10**(-15)):
    """https://www.kaggle.com/competitions/playground-series-s3e26"""
    y_true = np.clip(y_true, eps, 1-eps)
    y_pred = np.clip(y_pred, eps, 1-eps)
    return -np.mean(np.sum(y_true*np.log(y_pred), axis=-1))


def accuracy(y_true, y_pred):
    """https://www.kaggle.com/competitions/ml-olympiad-toxic-language-ptbr-detection"""
    return np.mean(y_true == y_pred)


def roc_auc(y_true, y_pro):
    """https://www.kaggle.com/competitions/playground-series-s4e3"""
    steps = 10000
    x = []
    y = []
    for step in range(steps):
        step_idx = step/steps
        y_pred = np.where(y_pro >= step_idx, 1, 0)
        true_positive_rate = np.sum(
            np.where(y_true+y_pred == 2, 1, 0))/np.sum(y_true)
        false_positive_rate = np.sum(
            (y_true == 0) & (y_pred == 1))/np.sum(y_true == 0)
        x.append(false_positive_rate)
        y.append(true_positive_rate)

    # Shell's Sort(nlogn)
    mid = steps//2
    while (mid):
        for i in range(0, mid):
            for j in range(i+mid, steps, mid):
                if x[j] < x[j-mid]:
                    k = j
                    while ((k >= mid) & (x[k] <= x[k-mid])):
                        if x[k] < x[k-mid]:
                            x[k], x[k-mid] = x[k-mid], x[k]
                            y[k], y[k-mid] = y[k-mid], y[k]
                        elif (x[k] == x[k-mid]) & (y[k] < y[k-mid]):
                            y[k], y[k-mid] = y[k-mid], y[k]
                        k -= mid
        mid = mid//2
    AUC = 0
    for i in range(len(y)-1):
        AUC += (x[i+1]-x[i])*(y[i+1]+y[i])/2
    return AUC


def KL_divergence(p, q, epsilon=10**(-15)):
    """https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification"""
    p = torch.clip(p, epsilon, 1-epsilon)
    q = F.log_softmax(q, dim=1)
    return torch.mean(torch.sum(p*(torch.log(p)-q), dim=1))


def cross_entropy_loss(y_pro, y_target, eps=1e-15):
    """https://www.kaggle.com/competitions/digit-recognizer"""
    y_target = torch.eye(y_pro.shape[-1])[y_target]
    y_pro = torch.clip(y_pro, eps, 1-eps)
    y_target = torch.clip(y_target, eps, 1-eps)
    return -torch.mean(torch.sum(y_target*torch.log(y_pro), dim=1), dim=0)


def focal_loss(y_pro, y_target, eps=1e-15, gamma=0.25):
    y_target = torch.eye(y_pro.shape[-1])[y_target]
    y_pro = torch.clip(y_pro, eps, 1-eps)
    y_target = torch.clip(y_target, eps, 1-eps)
    return -torch.mean(torch.sum(y_target*((1-y_pro)**gamma)*torch.log(y_pro), dim=1), dim=0)
