
# coding: utf-8

import math
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

x1_mean = [0, 0]
x1_cov = [[1, 0], [0, 1]]
x2_mean = [3, 0]
x2_cov = [[1, 0], [0, 1]]
x3_mean = [0, 3]
x3_cov = [[1, 0], [0, 1]]
x1 = np.random.multivariate_normal(x1_mean, x1_cov, 500)
#print np.mean(x1, axis=0)
x2 = np.random.multivariate_normal(x2_mean, x2_cov, 500)
#print np.mean(x2, axis=0)
x3 = np.random.multivariate_normal(x3_mean, x3_cov, 500)
#print np.mean(x3, axis=0)
# x = np.zeros((500, 2))
# x[:100]=x1[:100]
# x[101:350]=x2[101:350]
# x[351:]=x3[351:]

x = np.zeros((500, 2))
for i in range(len(x)):
    x[i] = 0.2*x1[i]+0.5*x2[i]+0.3*x3[i]
# print x

def K_means(K, x, t):
    u = []
    for i in range(K):
        a = [np.random.random(), np.random.random()]
        u.append(a)
   # print u
    c = range(len(x))
    dis = range(K)
    of = [0 for i in range(t)]
    for i in range(t):
        uk = {}
        nk = {}
        count = 0
        for p in range(K):
            uk[p+1] = 0
            nk[p+1] = 0
        for j in range(len(x)):
            for l in range(K):
                dis[l] = (x[j][0]-u[l][0])**2+(x[j][1]-u[l][1])**2
            c[j] = dis.index(min(dis))+1
            #print min(dis)
            
            of[i] = of[i] + min(dis)
            nk[c[j]] += 1
            uk[c[j]] += x[j] 
        #print u
        for k in range(K):
            if nk[k+1] == 0:
                u[k] = [0, 0]
            else:
                u[k] = uk[k+1]/nk[k+1]
    return of, c


l = np.zeros((4, 20))
c = np.zeros((4, 500))
for i in range(2, 6):
    l1, c[i-2] = K_means(i, x, 20)
    print(len(l1))
    for j in range(len(l1)):
        l[i-2][j]= l1[j]
#print l
c3 = c[1]-1
c5 = c[3]-1
# print c3
# print c5


fig = plt.figure()
plt.plot(range(2, 21), l[0][1:], label = "K=2")
plt.plot(range(2, 21), l[1][1:], label = "K=3")
plt.plot(range(2, 21), l[2][1:], label = "K=4")
plt.plot(range(2, 21), l[3][1:], label = "K=5")
plt.xlabel('t')
plt.ylabel('Objective-function')
plt.legend()
plt.show()

fig = plt.figure()
plt.plot(range(2, 21), l[0][1:], label = "K=2")
plt.xlabel('t')
plt.ylabel('Objective-function')
plt.legend()
plt.show()

fig = plt.figure()
plt.plot(range(2, 21), l[1][1:], label = "K=3")
plt.xlabel('t')
plt.ylabel('Objective-function')
plt.legend()
plt.show()

fig = plt.figure()
plt.plot(range(2, 21), l[2][1:], label = "K=4")
plt.xlabel('t')
plt.ylabel('Objective-function')
plt.legend()
plt.show()

fig = plt.figure()
plt.plot(range(2, 21), l[3][1:], label = "K=5")
plt.xlabel('t')
plt.ylabel('Objective-function')
plt.legend()
plt.show()

print type(c3[1])

px=[]
py=[]
for i in range(len(x)):
    px.append(x[i][0])
    py.append(x[i][1])
color1 = ['red', 'blue', 'yellow' ]
color2 = ['red', 'blue', 'yellow', 'green', 'brown']
fig = plt.figure()  
ax1 = fig.add_subplot(111)  
ax1.set_title('Scatter Plot')  
plt.xlabel('X')  
plt.ylabel('Y')   
for i in range(len(c3)):
    plt.scatter(px[i], py[i], c = color1[int(c3[i])])     
plt.show() 

fig = plt.figure()  
ax1 = fig.add_subplot(111) 
ax1.set_title('Scatter Plot')  
plt.xlabel('X')  
plt.ylabel('Y')   
for i in range(len(c3)):
    ax1.scatter(px[i], py[i], c = color2[int(c5[i])],marker = 'o')     
plt.show() 
