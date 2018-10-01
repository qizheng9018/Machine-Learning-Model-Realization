# coding: utf-8
import math
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

x_train = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw3/hw3-data/gaussian_process/X_train.csv","rb"),delimiter=",",skiprows=0)
y_train = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw3/hw3-data/gaussian_process/y_train.csv","rb"),delimiter=",",skiprows=0)
x_test = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw3/hw3-data/gaussian_process/X_test.csv","rb"),delimiter=",",skiprows=0)
y_test = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw3/hw3-data/gaussian_process/y_test.csv","rb"),delimiter=",",skiprows=0)

def Kernelij(X1, X2, i, j, b):
    sum1 = 0
    e = math.e
    if len(X1[0]) == 1:
        sum1 = (X1[i] - X2[j])**2
    else:
        for p in range(7):
            sum1 = sum1 + (X1[i][p] - X2[j][p])**2
    Kij = e**(-sum1/b)
    return Kij

def Knre(x_train, b, xigema2):
    n = len(x_train)
    Kn = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            Kn[i][j] = Kernelij(x_train, x_train, i, j, b)
    I = np.matrix(np.eye(n))
    summ = np.add(xigema2*I, Kn)
    return summ.I, Kn

def KDnall(x_test, x_train, b):
    n = len(x_train)
    m = len(x_test)
    KDn = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            KDn[i][j] = Kernelij(x_test, x_train, i, j, b)
    return KDn
    
b1 = [5, 7, 9, 11, 13, 15]
xigema21 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 
set1 = []
for b in b1:
    for xigema2 in xigema21:
        set1.append([b, xigema2])

n = len(y_train)
m = len(y_test)
y = np.zeros((n,1))

for i in range(n):
    y[i] = y_train[i]

    
y_pre = np.zeros((60, m))
for i in  range(len(set1)):
    b = set1[i][0]
    xigema2 = set1[i][1]
    KnI, Kn = Knre(x_train, b, xigema2)
    KDn = KDnall(x_test, x_train, b)
    for j in range(len(x_test)):        
        a = KDn[j]
        b = a.dot(KnI)
        c = b.dot(y)
        y_pre[i][j] = c


RMSE = []
for i in range(60):
    sum1 = 0
    for j in range(42):
        sum1 = sum1 + (y_test[j] - y_pre[i][j])**2
    rmse = sqrt(sum1/42)
    RMSE.append(rmse)
print RMSE
print min(RMSE)
print RMSE.index(min(RMSE))
print set1[30]

k = len(x_train)
x4 = np.zeros((k,1))
for i in range(k):
    x4[i] = x_train[i][3]
y4 = np.zeros((k,1))
for i in range(len(y_train)):
    y4[i] = y_train[i]

fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('Scatter Plot')  
plt.xlabel('X')  
plt.ylabel('Y')   
ax1.scatter(x4,y4,c = 'b',marker = 'o')     
plt.show() 

n = len(y_train)
y_pred = np.zeros((n, 1))
b = 5
xigema2 = 2
y = np.zeros((n,1))
for i in range(n):
    y[i] = y_train[i]
#print x4[0]
KnI, Kn = Knre(x4, b, xigema2)
#print Kn
KDn = KDnall(x4, x4, b)
#print KDn
y_pred = KDn.dot(KnI.dot(y))

xq = []
for i in range(350):
    xq.append(x4[i][0]) 
#print xq
a = np.argsort(xq)[:]
x41 = []
y41 = []
for i in range(len(a)):
    k = a[i]
    x41.append(x4[k][0])
    y41.append(float(y_pred[k][0]))
#print x41
#print y41
#print len(x41)
#print len(y41)
fig = plt.figure()   
ax1.set_title('Scatter Plot')  
plt.xlabel('X')  
plt.ylabel('Y')   
plt.plot(x41,y41)     
plt.show() 
