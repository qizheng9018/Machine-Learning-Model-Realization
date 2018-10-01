
# coding: utf-8

import math
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

x_train = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw2/hw2-data/X_train.csv","rb"),delimiter=",",skiprows=0)
y_train = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw2/hw2-data/y_train.csv","rb"),delimiter=",",skiprows=0)
x_test = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw2/hw2-data/X_test.csv","rb"),delimiter=",",skiprows=0)
y_test = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw2/hw2-data/y_test.csv","rb"),delimiter=",",skiprows=0)
y_pre = range(len(y_test))

def theta_hat(x, y, d, sign):
    sum_x = 0
    count = 0
    if d<= 53:
        for i in range(len(x)):
            if y[i] == sign:
                count = count + 1
                sum_x = sum_x + x[i][d]
        theta = sum_x/count
        return theta
    else:        
        for i in range(len(x)):
            if y[i] == sign:
                count = count + 1
                m = math.log(x[i][d], math.e)
                sum_x = sum_x + m
        theta = count/sum_x
        return theta

theta =  random.random(size=(57,2))

for i in range(57):
    for j in range(2):
        theta[i][j] = theta_hat(x_train, y_train, i, j)    

sum_y=0        
for i in range(len(y_train)):
    sum_y = sum_y + y_train[i]
pi = sum_y/len(y_train) 

theta_t = theta.T
# plt.figure
# plt.subplot(211)
# plt.stem(range(54), theta_t[0][:54], '-.')
# plt.xlabel('Dimension')
# plt.ylabel('Bernoulli parameter')
# plt.legend()
# plt.subplot(212)
# plt.stem(range(54), theta_t[1][:54], '-.')
# plt.xlabel('Dimension')
# plt.ylabel('Bernoulli parameter')
# plt.legend()
# plt.show()
f, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
ax1.set_xlabel("Dimensions")
ax1.set_ylabel("Bernoulli Parameters")
markerline, stemlines, baseline = ax1.stem(range(1, 55), theta_t[0][:54], '-.')
plt.setp(markerline, 'markerfacecolor', 'green')
plt.setp(stemlines, 'color', 'green')
markerline, stemlines, baseline = ax1.stem(range(1, 55), theta_t[1][:54], '-.')
plt.setp(markerline, 'markerfacecolor', 'orange')
plt.setp(stemlines, 'color', 'orange')
plt.setp(baseline, 'color', 'black', 'linewidth', 2)
plt.legend(["Class y=0", "Class y=1"], loc='best', numpoints=2)
plt.title("Bernoulli Parameters [y = 0(Blue), y = 1(Red)]")
plt.show()

def bayes_classifier(x, y, theta_y, pi):
    y_pre =  range(len(y))
   
    for i in  range(len(y)):
        multi_1 = 1-pi
        multi_2 = pi
        j = 0
        while j < len(x[0]):
            if j <= 53:
                multi_1 = multi_1*(theta_y[j][0]**x[i][j])*((1-theta_y[j][0])**(1-x[i][j]))
                multi_2 = multi_2*(theta_y[j][1]**x[i][j])*((1-theta_y[j][1])**(1-x[i][j]))
                j = j + 1
            else:
                multi_1 = multi_1*(theta_y[j][0])*(x[i][j]**(-1-theta_y[j][0]))
                multi_2 = multi_2*(theta_y[j][1])*(x[i][j]**(-1-theta_y[j][1]))
                j = j + 1
        if multi_1 >= multi_2:
            y_pre[i] = 0
        else:
            y_pre[i] = 1
    return y_pre

def accuracy(y_pre, y_test):
    acc = float(0)
    count = float(0)
    for i in range(len(y_pre)):
        if y_pre[i] == y_test[i]:
            count = count + 1
    a = float(len(y_pre))
    acc = count/a
    return acc

y_pre = bayes_classifier(x_test, y_test, theta, pi) 
acc = accuracy(y_pre,y_test)
#print acc
count1 = count2 = count3 = count4 = 0
for i in range(len(y_pre)):
    if y_pre[i] ==0 and y_test[i] == 0:
        count1 += 1
    if y_pre[i] ==0 and y_test[i] == 1:
        count2 += 1
    if y_pre[i] ==1 and y_test[i] == 0:
        count3 += 1
    if y_pre[i] ==1 and y_test[i] == 1:
        count4 += 1
        
print count1
print count2
print count3
print count4

def normalization(A):
    for k in range(54, 57):
        a = 0
        c = A[0][k]
        for i in range(len(A)):
            if A[i][k]<= c:
                c = A[i][k]
                a = i
        b = 0
        d = A[0][k]
        for i in range(len(A)):
            if A[i][k]>= d:
                d = A[i][k]
                b = i 
        e = d-c
        for i in range(len(A)):
            A[i][k] = (A[i][k] - c)/e
    return A
            
x_train_n = normalization(x_train)
x_test_n = normalization(x_test)
print x_test_n[1][54]

def KNN(x_test, x_train):    
    distance =  random.random(size=(len(x_test),len(x_train)))   
    for i in range(len(x_test)):       
        for j in range(len(x_train)):
            sumd = 0
            for k in range(57):
                sumd = sumd + abs(x_test[i][k]-x_train[j][k])
            distance[i][j] = sumd
    return distance
dis = random.random(size=(len(x_test),len(x_train)))
dis = KNN(x_test_n, x_train_n)
b = np.argsort(dis[5])[:20]
print b

def knnpre(k, dis, y_train):
    data = []
    for i in range(len(dis)):
        a = dis[i]
        b = np.argsort(a)[:k]
        data.append(b)
    #print data[0]      
    y_pre2 = range(len(data))
    count3 = 0
    for i in range(len(data)):
        count0 = count1 = 0
        for j in range(len(data[0])):
            if y_train[data[i][j]] == 0:
                count0 += 1
            elif y_train[data[i][j]] == 1:
                count1 += 1
        if count0 > count1:
            y_pre2[i]=0
        elif count1 > count0:
            y_pre2[i]=1
        else:
            if count3%2==0:
                y_pre2[i]=0
                count3 +=1
            if count3%2==1:
                y_pre2[i]=1
                count3 +=1
        #print count0, count1, count3
    return y_pre2

def acck(k, dis, y_train, y_test):
    acc = []
    for i in range(1, k+1):
        y_pre_k = knnpre(i, dis, y_train)
        #print y_pre_k
        acc1 = float(0)
        countk = float(0)
        for i in range(len(y_pre_k)):
            if y_pre_k[i] == y_test[i]:
                countk = countk + 1
        a1 = float(len(y_pre_k))
        acc1 = countk/a1
        #print acc1
        acc.append(acc1)
    return acc

acc = acck(20, dis, y_train, y_test)
fig = plt.figure()
plt.plot(range(1, 21),acc)
ax.set_title(u"KNN")
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

