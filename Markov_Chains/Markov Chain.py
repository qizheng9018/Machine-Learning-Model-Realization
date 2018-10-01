
# coding: utf-8

import math
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
score = np.loadtxt(open("/Users/zhengqi/Downloads/machine-learning/hw5/hw5-data/CFB2017_scores.csv","rb"),delimiter=",",skiprows=0)
print (score[1][0])

Mhat = np.zeros((763, 763))
M = np.zeros((763, 763))
for i in range(len(score)):
    #print score[i]
    if score[i][1] > score[i][3]:
        a = int(score[i][0])-1
        b = int(score[i][2])-1
        Mhat[a][a] = Mhat[a][a] + 1 + score[i][1]/(score[i][1]+score[i][3])
        Mhat[b][a] = Mhat[b][a] + 1 + score[i][1]/(score[i][1]+score[i][3])
        Mhat[b][b] = Mhat[b][b] + score[i][3]/(score[i][1]+score[i][3])
        Mhat[a][b] = Mhat[a][b] + score[i][3]/(score[i][1]+score[i][3])
    else:
        a = int(score[i][0])-1
        b = int(score[i][2])-1
        Mhat[b][b] = Mhat[b][b] + 1 + score[i][3]/(score[i][1]+score[i][3])
        Mhat[a][b] = Mhat[a][b] + 1 + score[i][3]/(score[i][1]+score[i][3])
        Mhat[a][a] = Mhat[a][a] + score[i][1]/(score[i][1]+score[i][3])
        Mhat[b][a] = Mhat[b][a] + score[i][1]/(score[i][1]+score[i][3])

for i in range(len(Mhat)):
    c = np.sum(Mhat[i])
    for j in range(len(Mhat[i])):
        M[i][j] = Mhat[i][j]/c

print M


def markov(M, t):
    tmp = np.ones((1, 763))/763
    for i in range(t):
        w_t = tmp.dot(M)
        tmp = w_t
    return w_t

w_t_10 = markov(M, 10)
w_t_100 = markov(M, 100)
w_t_1000 = markov(M, 1000)
w_t_10000 = markov(M, 10000)
print markov(M, 10000)
print w_t_10000
w_t_11 = np.argsort(w_t_10)[:]
w_t_101 = np.argsort(w_t_100)[:]
w_t_1001 = np.argsort(w_t_1000)[:]
w_t_10001 = np.argsort(w_t_10000)[:]
list1 = []
list2 = []
list3 = []
list4 = []
list11 = []
list21 = []
list31 = []
list41 = []
for i in range(763):
    list1.append(w_t_11[0][i])
    list2.append(w_t_101[0][i])
    list3.append(w_t_1001[0][i])
    list4.append(w_t_10001[0][i])
w_t_12 = list1[::-1][:25]
w_t_102 = list2[::-1][:25]
w_t_1002 = list3[::-1][:25]
w_t_10002 = list4[::-1][:25]
for i in range(25):
    list11.append(w_t_10[0][w_t_12[i]])
    list21.append(w_t_100[0][w_t_102[i]])
    list31.append(w_t_1000[0][w_t_1002[i]])
    list41.append(w_t_10000[0][w_t_10002[i]])

print w_t_12
print list11
print w_t_102
print list21
print w_t_1002
print list31
print w_t_10002
print list41

def markov2(M, w_t):
    w_t = w_t.dot(M)
    return w_t

w_0 = np.ones((1, 763))/763
MT = np.transpose(M)
a, b = np.linalg.eig(MT)
print np.argmax(a)
winfty = b[:, 16]
a = np.sum(winfty)
winfty = winfty.reshape(1, 763)
for i in range(763):
    winfty[0][i] = winfty[0][i]/a
print np.sum(winfty)
dis = []
for i in range(10000):
    count  = 0
    if i == 0:
        w_t = markov2(M, w_0)
        print w_t
        for i in range(763):
            count = count + abs(winfty[0][i]-w_t[0][i])
        dis.append(count)
    else:
        w_t = markov2(M, w_t)
        for i in range(763):
            count = count + abs(winfty[0][i]-w_t[0][i])
        dis.append(count)

fig = plt.figure()
plt.plot(dis)
plt.xlabel('Iteration time')
plt.ylabel('L1 Distance')
plt.legend()
plt.show()
