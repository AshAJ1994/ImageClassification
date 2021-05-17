import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st
from sklearn import metrics
from GradCAM_Visualization import *
import pandas as pd
import openpyxl
from pandas import ExcelWriter

def auc(X, Y):
    return 1 / (len(X) * len(Y)) * sum([kernel(x, y) for x in X for y in Y])


def kernel(X, Y):
    return .5 if Y == X else int(Y < X)


def structural_components(X, Y):
    V10 = [1 / len(Y) * sum([kernel(x, y) for y in Y]) for x in X]
    V01 = [1 / len(X) * sum([kernel(x, y) for x in X]) for y in Y]
    return V10, V01


def get_S_entry(V_A, V_B, auc_A, auc_B):
    return 1 / (len(V_A) - 1) * sum([(a - auc_A) * (b - auc_B) for a, b in zip(V_A, V_B)])


def z_score(var_A, var_B, covar_AB, auc_A, auc_B):
    return (auc_A - auc_B) / ((var_A + var_B - 2 * covar_AB) ** (.5))


# Model A (random) vs. "good" model B
# preds_A = np.array([.5, .5, .5, .5, .5, .5, .5, .5, .5, .5])
# preds_B = np.array([.2, .5, .1, .4, .9, .8, .7, .5, .9, .8])
# actual= np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
resultDF = pd.DataFrame()
preds_A = y_pred_prob_A
preds_A = [round(elem, 4) for elem in preds_A]
resultDF['ModelA_Preds'] = preds_A

preds_B = y_pred_prob_B
preds_B = [round(elem, 4) for elem in preds_B]
resultDF['ModelB_Preds'] = preds_B

actual = y_true
resultDF['Actual_Class'] = actual

# resultDF.to_excel('Result_DeLongTest.xlsx', index=False)

def group_preds_by_label(preds, actual):
    X = [p for (p, a) in zip(preds, actual) if a]
    Y = [p for (p, a) in zip(preds, actual) if not a]
    return X, Y

X_A, Y_A = group_preds_by_label(preds_A, actual)
X_B, Y_B = group_preds_by_label(preds_B, actual)

V_A10, V_A01 = structural_components(X_A, Y_A)
V_B10, V_B01 = structural_components(X_B, Y_B)

auc_A = auc(X_A, Y_A)
auc_B = auc(X_B, Y_B)

# Compute entries of covariance matrix S (covar_AB = covar_BA)
var_A = (get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)
         + get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
var_B = (get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)
         + get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
covar_AB = (get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)
            + get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))

# Two tailed test
z = z_score(var_A, var_B, covar_AB, auc_A, auc_B)
p = st.norm.sf(abs(z))*2
print('finished calculation')