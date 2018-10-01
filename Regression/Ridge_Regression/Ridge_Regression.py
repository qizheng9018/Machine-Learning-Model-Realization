
# coding: utf-8

import math
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

def ridge_regression(X, y, lambd = 0.2):
    X = mat(X)
    y = mat(y)
    XTX = X.T*X
#     m, _ = XTX.shape
#     I = np.matrix(np.eye(m))
#     w = (XTX + lambd*I).I*X.T*y
    denom = XTX + eye(shape(X)[1])*lambd
    k = (X.T*(y.T))
    w = denom.I * k
    return w

def ridge_mat(X, y, ntest):
    X = mat(X)
    y = mat(y)
    _, n = X.shape
    ws = np.zeros((ntest, n))
    for i in range(ntest):
        w = ridge_regression(X, y, lambd = i)
        ws[i, :] = w.T
    return ws

def RMSE(X, y, ws, l):
    ws_1 = ws[0:l,:]
    RMSE = range(l)
    for i in range(len(ws_1)):
        y_2_pre = mat(ws_1[i])*mat(X.T)
        y_2_pre = array(y_2_pre)
        suma = 0
        
        for j in range(len(y)):
            a = float(y[j])
            b = float(y_2_pre[0][j])
            
            suma = suma + (a-b)*(a-b)
        b= suma/42
        c = sqrt(b)
        RMSE[i] = c
        suma = 0
    return RMSE

def X_p(X, p):
    if p == 1:
        return X
    else:
        m, n = X.shape
        R = np.zeros((m, n+6*(p-1)))
        
        for i in range(m):
            for j in range(n):
                R[i][j] = X[i][j]
        for i in range(p-1):
            for k in range(m):
                for j in range(6):
                    if i == 0:
                        R[k][n+j] = X[k][j]*X[k][j]
                    elif i == 1:
                        R[k][n+j] = X[k][j]*X[k][j]*X[k][j]
            n = n+6
    return R

if '__main__' == __name__:
    test = 5001
    X_1 = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw1-data/X_train.csv","rb"),delimiter=",",skiprows=0)
    y_1 = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw1-data/y_train.csv","rb"),delimiter=",",skiprows=0)
    X_2 = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw1-data/X_test.csv","rb"),delimiter=",",skiprows=0)
    y_2 = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw1-data/y_test.csv","rb"),delimiter=",",skiprows=0)
    U,S,VT=linalg.svd(X_1)


ws = ridge_mat(X_1, y_1, 5001)
freedom = range(5001)
sums = 0
for i in range(test):
    for j in range(len(S)):
        sums = sums + (S[j]*S[j]/(i+S[j]*S[j]))        
    freedom[i] = sums
    sums = 0
fig = plt.figure()
ax = fig.add_subplot(111)
w_1 = ws[:,0]
w_2 = ws[:,1]
w_3 = ws[:,2]
w_4 = ws[:,3]
w_5 = ws[:,4]
w_6 = ws[:,5]
w_7 = ws[:,6]
plt.plot(freedom, w_1, label = "Dim1")
plt.plot(freedom, w_2, label = "Dim2")
plt.plot(freedom, w_3, label = "Dim3")
plt.plot(freedom, w_4, label = "Dim4")
plt.plot(freedom, w_5, label = "Dim5")
plt.plot(freedom, w_6, label = "Dim6")
plt.plot(freedom, w_7, label = "Dim7")
ax.set_title(u"RIDGE REGRESSION")
plt.xlabel('df(lambda)')
plt.ylabel('w_rr')
plt.legend()
plt.show()
        
RMSE_1 = RMSE(X_2, y_2, ws, 51)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(RMSE_1, label = "RMSE_1")
ax.set_title(u"ROOT MEAN SQUARED ERROR")
plt.xlabel('lambda')
plt.ylabel('RMSE')
plt.legend()
plt.show()

X_1_1 = X_p(X_1, 1)
X_2_1 = X_p(X_2, 1)
ws_1 = ridge_mat(X_1_1, y_1, 501)
RMSE_1 =  RMSE(X_2_1, y_2, ws_1, 501)
X_1_2 = X_p(X_1, 2)
X_2_2 = X_p(X_2, 2)
ws_2 = ridge_mat(X_1_2, y_1, 501)
RMSE_2 =  RMSE(X_2_2, y_2, ws_2, 501)
X_1_3 = X_p(X_1, 3)
X_2_3 = X_p(X_2, 3)
ws_3 = ridge_mat(X_1_3, y_1, 501)
RMSE_3 =  RMSE(X_2_3, y_2, ws_3, 501)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(RMSE_1, label = "RMSE_1")
plt.plot(RMSE_2, label = "RMSE_2")
plt.plot(RMSE_3, label = "RMSE_3")
ax.set_title(u"ROOT MEAN SQUARED ERROR")
plt.xlabel('lambda')
plt.ylabel('RMSE')
plt.legend()
plt.show()





