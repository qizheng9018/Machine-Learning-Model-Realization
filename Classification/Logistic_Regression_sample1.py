import math
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

x_train = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw2/hw2-data/X_train.csv","rb"),delimiter=",",skiprows=0)
y_train = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw2/hw2-data/y_train.csv","rb"),delimiter=",",skiprows=0)

for i in range(len(y_train)):
    if y_train[i] == 0:
        y_train[i] = -1
x_train1 =  random.random(size=(4508,58))
for i in range(4508):
    for j in range(58):
        if j == 57:
            x_train1[i][j] =1
        else:
            x_train1[i][j] = x_train[i][j]
w = np.zeros((58,1))
print x_train[0]

def xigema_sum(y_train, w, x_train):
    sumx = np.zeros((58,1))
    suml = 0
    e = math.e
    for i in range(4508):
        x_train = mat(x_train)
        b = x_train[i]
        #print b
        #print w
        f = b.dot(w)
        #print f
        f = int(f)
        if y_train[i] == 1:
            if f > 600:
                m =  1/(1+e**(-f))
                k = np.log(m)
                #print k
            elif f < -600:
                m = (e**f)/(1+e**f)
                k = f
                #print k
            else:   
                m = (e**f)/(1+e**f)
                k = np.log(m)
                #print k
            #print k
            suml = suml + k
            a = 1-m
            c = a*y_train[i]*(x_train[i].T)
            sumx = np.add(sumx, c)
        if y_train[i] == -1:
            if f > 600:
                m = e**(-f)/(1+e**(-f))
                k = -f
                #print k
            elif f < -600:
                m = 1-((e**f)/(1+e**f))
                k = np.log(m)
                #print k
            else:    
                m = e**(-f)/(1+e**(-f))
                k = np.log(m)
                #print k
            suml = suml + k
            a = 1-m
            c = a*y_train[i]*(x_train[i].T)
            sumx = np.add(sumx, c)
    print suml
    return sumx, suml

def yita(t):
    a = 1/(100000*sqrt(t+1))
    return a

def update(w, t, y_train, x_train):
    L = range(t)
    count = 0
    counta = 0
    for i in range(t):
        
        if count % 100 == 0:
            counta += 1
            #print counta
        count += 1
        q1, L[i] = xigema_sum(y_train, w, x_train)
        q2 = q1*yita(i)
        w = np.add(w, q2) 
        #print w
    return L
    
L = update(w, 10, y_train, x_train1)    


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(L)
ax.set_title(u"Likelihood")
plt.xlabel('k')
plt.ylabel('likelihood')
plt.legend()
plt.show()




