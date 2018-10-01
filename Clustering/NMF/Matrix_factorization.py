
# coding: utf-8

import math
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

rtest = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw4/hw4-data/ratings_test.csv","rb"),delimiter=",",skiprows=0)
rtrain = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw4/hw4-data/ratings.csv","rb"),delimiter=",",skiprows=0)

print type(rtrain[0][0])

form = np.zeros((943, 1682))
for i in range(len(rtrain)):
    form[int(rtrain[i][0]-1)][int(rtrain[i][1]-1)] = float(rtrain[i][2])
form_1 = np.zeros((943, 1682))
for i in range(len(rtest)):
    form_1[int(rtest[i][0]-1)][int(rtest[i][1]-1)] = float(rtest[i][2])

def MF(u, vt, form, form_1, steps=100, xigma2 = 0.25):
    formt = form.T
    cov = eye(10)
    L = []
    a = np.zeros((1,10))
    for k in range(steps):
        print k
        for i in range(len(u)):
            sumu1 = np.zeros((10, 10))
            sumu2 = np.zeros((10, 1))
            for q in range(len(form[0])):
                if form[i][q] != 0:
                    sumu1 += np.dot(vt[q].reshape(10, 1), vt[q].reshape(1, 10))
                    sumu2 += (vt[q].reshape(10, 1))*form[i][q]
            u[i] = np.dot(np.linalg.inv(np.add(xigma2*cov, sumu1)), sumu2).T
        for j in range(len(vt)):
            sumv1 = np.zeros((10, 10))
            sumv2 = np.zeros((10, 1))
            for p in range(len(formt[0])):
                if formt[j][p] != 0:
                    sumv1 += np.dot(u[p].reshape(10, 1), u[p].reshape(1, 10))
                    sumv2 += (u[p].reshape(10, 1))*formt[j][p]
            vt[j] = np.dot(np.linalg.inv(np.add(xigma2*cov, sumv1)), sumv2).T
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sumr = 0
        countr = 0
        v = vt.T
        R = np.dot(u, v)
        K = np.add(form, -R)
        for m in range(len(form)):
            for n in range(len(form[0])):
                if form[m][n] != 0:
                    sum1 += (K[m][n])**2
                if form_1[m][n] != 0 and k == 99:
                    countr += 1
                    sumr += (form_1[m][n] - R[m][n])**2   
        for i in range(len(u)):
            for j in range(10):
                sum2 += (u[i][j])**2
        for i in range(len(vt)):
            for j in range(10):
                sum3 += (vt[i][j])**2
        if k == 99:
            print countr 
            RMSE = sqrt(sumr/countr)          
        L.append(-2*sum1 - 0.5*sum2 - 0.5*sum3)
        #print L[k]
    return L, RMSE, u, vt

RMSE = []
L = np.zeros((1,100))
ut = np.zeros((943, 1))
vtt = np.zeros((1682, 1))
for i in range(10):
    u = np.zeros((943, 10))
    vt = np.zeros((1682, 10))
    cov = eye(10)
    mean = [0 for i in range(10)]
    u = np.random.multivariate_normal(mean, cov, 943)
    vt = np.random.multivariate_normal(mean, cov, 1682)
    L_1, RMSE_1, utmp, vtmp = MF(u, vt, form, form_1, steps=100, xigma2 = 0.25)
#     print L_1
#     print utmp
#     print vtmp
    L = np.row_stack((L, L_1))
    ut = np.column_stack((ut, utmp))
    vtt = np.column_stack((vtt, vtmp))
    RMSE.append(RMSE_1)
# OF = []
# for i in range(10):
#     OF.append(int(L[i][99])

print len(L[1])

f, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
plt.plot(range(1,99), L[1][2:], label="time_1")
plt.plot(range(1,99),L[2][2:], label="time_2")
plt.plot(range(1,99),L[3][2:], label="time_3")
plt.plot(range(1,99),L[4][2:], label="time_4")
plt.plot(range(1,99),L[5][2:], label="time_5")
plt.plot(range(1,99),L[6][2:], label="time_6")
plt.plot(range(1,99),L[7][2:], label="time_7")
plt.plot(range(1,99),L[8][2:], label="time_8")
plt.plot(range(1,99),L[9][2:], label="time_9")
plt.plot(range(1,99),L[10][2:], label="time_10")
plt.xlabel('iteration')
plt.ylabel('Objective-function')
plt.legend()
plt.show()

of = []
for i in range(10):
    of.append(L[i+1][99])
print of
print RMSE
k = np.argsort(of)[:]
print k

c = of.index(max(of))
print c
print vtt
print type(vtt[0][91])

vmin = np.zeros((1682, 10))
for i in range(1682):
    for j in range(10):
        k = 10*c+1
        vmin[i][j] = float(vtt[i][k+j])
v1 = vmin[49]
v2 = vmin[484]
v3 = vmin[181]
v1d=[]
v2d=[]
v3d=[]
for i in range(1682):
    dis1=0
    dis2=0
    dis3=0
    for j in range(10):
        dis1 += (v1[j] - vmin[i][j])**2
        dis2 += (v2[j] - vmin[i][j])**2
        dis3 += (v3[j] - vmin[i][j])**2
    v1d.append(sqrt(dis1))
    v2d.append(sqrt(dis2))
    v3d.append(sqrt(dis3))
near1 = np.argsort(v1d)[:11]
near2 = np.argsort(v2d)[:11]
near3 = np.argsort(v3d)[:11]
print near1
print near2
print near3
d1 = []
d2 = []
d3 = []
for i in range(10):
    a = near1[i+1]
    b = near2[i+1]
    c = near3[i+1]
    d1.append(v1d[a])
    d2.append(v2d[b])
    d3.append(v3d[c])
print d1
print d2
print d3
   



