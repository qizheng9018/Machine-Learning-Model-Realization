# coding: utf-8
import math
from numpy import *
import numpy as np
import random
import matplotlib.pyplot as plt

x_train = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw3/hw3-data/boosting/X_train.csv","rb"),delimiter=",",skiprows=0)
y_train = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw3/hw3-data/boosting/y_train.csv","rb"),delimiter=",",skiprows=0)
x_test = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw3/hw3-data/boosting/X_test.csv","rb"),delimiter=",",skiprows=0)
y_test = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw3/hw3-data/boosting/y_test.csv","rb"),delimiter=",",skiprows=0)
x_train1 = np.zeros((len(x_train), 6))
x_test1 = np.zeros((len(x_test), 6))

for i in range(len(x_train)):
    for j in range(5):
        x_train1[i][j] = x_train[i][j]
    x_train1[i][5] = 1
print x_train1[0]
print len(x_train1)

for i in range(len(x_test)):
    for j in range(5):
        x_test1[i][j] = x_test[i][j]
    x_test1[i][5] = 1
print x_train1
print len(x_test1)

def iter_method(wt):
    #print wt
    total = sum(wt) 
    #print total
    rad = random.uniform(0,total)    
    #print rad
    cur_total = 0  
    res = 0 
    for i in range(len(wt)):  
        cur_total += wt[i]  
        if rad <= cur_total:  
            res = i   
            break  
    return i 

def Bt(y_train, x_train, wt, dictb):
    n, m = x_train.shape
    x_bt = np.zeros((n, m))
    y_bt = np.zeros((n, 1)) 
    for i in range(n):
        k = iter_method(wt)
        if k not in dictb:
            dictb[k] = 1
        else:
            dictb[k] += 1
        #print k
        y_bt[i] = y_train[k]  
        x_bt[i] = x_train[k]
    return x_bt, y_bt, dictb

def LS(y_train, x_train, wt, dictb):
    n, m = x_train.shape
    x_bt = np.zeros((n, m))
    y_bt = np.zeros((n, 1))
    x_bt, y_bt, dictb = Bt(y_train, x_train, wt, dictb) 
    #print x_bt
    #print y_bt
    a = x_bt.T
    XTX = a.dot(x_bt)
#     XTX = mat(XTX)
    res = np.linalg.inv(XTX)
    k = a.dot(y_bt)
    w = (res).dot(k)
    #print w
    return w, dictb

def e_a(wt, y_train, x_train, w):
    e = math.e
    #print x_train
    y_pr = x_train.dot(w)
    y_k = []      
    et = 0
    for i in range(len(y_train)):
        if y_train[i]*y_pr[i] < 0:
            et = et + wt[i]
            y_k.append(-1)
        else:
            y_k.append(1)
    if et > 0.5:
        for i in range(len(w)):
            w[i] = -w[i]
        y_pr = x_train.dot(w)
        y_k = []     
        et = 0
        for i in range(len(y_train)):
            if y_train[i]*y_pr[i] < 0:
                et = et + wt[i]
                y_k.append(-1)
            else:
                y_k.append(1)
    #print et            
    k = np.log((1-et)/et)
    at = 0.5*k
    hatw = []
    sumw = 0
    for i in range(len(x_train)):
        p = wt[i]*(e**(-at*y_k[i]))
        hatw.append(p)
        sumw = sumw + p
    wt1 = []
    for i in range(len(wt)):
        wt1.append(hatw[i]/sumw) 
    return wt1, at, et  

def fboost(x_test, T, y_train, x_train):
    dictb = {}
    m = len(x_test)
    n = len(y_train)
    y_b = np.zeros((T, m))
    sumx = np.zeros((m, 1))
    wt = []
    liste = []
    lista = []
    #print n
    for i in range(n):
        n = float(n)
        a = float(float(1)/n)
        #print a
        wt.append(a)
    #print wt
    count1 = 0
    for j in range(T):
        count1 += 1
        if count1 % 10 == 0:
            print count1
        w, dictb = LS(y_train, x_train, wt, dictb)         
        wt1, at, et= e_a(wt, y_train, x_train, w)
        liste.append(et)
        lista.append(at)
        y_pre = x_test.dot(w)
        for i in range(len(y_pre)):
            if y_pre[i] < 0:
                y_pre[i] = -1
            else:
                y_pre[i] = 1
        wt = wt1
        #print at
        sumx = np.add(sumx, at*y_pre)
        #print sumx
        for i in range(len(sumx)):
            if sumx[i]< 0:
                y_b[j][i] = -1
            else:
                y_b[j][i] = 1
    return y_b, liste, lista, dictb

T = 1500
y_b, liste1, lista1, dictb1 = fboost(x_test1, T, y_train, x_train1)

T = 1500
y_br, liste2, lista2, dictb2 = fboost(x_train1, T, y_train, x_train1)

accte = []
for i in range(len(y_b)):
    count = float(0)
    for j in range(len(y_b[0])):
        if y_b[i][j] != y_test[j]:
            count += 1
    k = float(0)
    m = float(len(y_b[0]))
    k = count/m
    accte.append(k)
print len(accte)
print min(accte)
#print accte

acctr = []
for i in range(len(y_br)):
    count = float(0)
    for j in range(len(y_br[0])):
        if y_br[i][j] != y_train[j]:
            count += 1
    k = float(0)
    m = float(len(y_br[0]))
    k = count/m
    acctr.append(k)
print len(acctr)
print min(acctr)
#print acctr

f, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
ax1.set_xlabel("T")
ax1.set_ylabel("Boost")
plt.plot(accte, label = "testing error")
plt.plot(acctr, label = "trainning error")
plt.legend()
plt.title("error value")
plt.show()

up = []
m = len(liste2)
e = math.e
for i in range(m):
    sumt = 0
    for j in range(i+1):
        sumt = sumt + (0.5 -liste2[j])**2
    p = -2*sumt
    r = e**p
    up.append(r)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(up)
ax.set_title(u"upper bound")
plt.xlabel('t')
plt.ylabel('upper bound value')
plt.legend()
plt.show()

x = dictb1.keys()
y = dictb1.values()

plt.xlabel('X')
plt.ylabel('times')
plt.title('Appear Frequency')
a = plt.subplot(1, 1, 1)
plt.ylim=(10, 40000)
plt.bar(x, y, facecolor='blue')
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(211)
plt.plot(liste1)

plt.xlabel('t')
plt.ylabel('epsilon')
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(212)
plt.plot(lista1)
plt.xlabel('t')
plt.ylabel('a_t')
plt.legend()
plt.show()


