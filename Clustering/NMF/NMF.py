
# coding: utf-8
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import re
import csv

x = np.zeros((3012, 8447))
count = 0
f = open(r"/Users/zhengqi/Downloads/machine-learning/hw5/hw5-data/nyt_data.txt", "r")
line = f.readline()
while line:
    a = line.split(",")
    for k in a:
        num, time = k.split(":")
        x[int(num)-1][count] += int(time)
    count += 1
    line = f.readline()

f.close()
# for line in lines:
#     list1.append(line)
print count

w = np.zeros((3012, 25))
h = np.zeros((25, 8447))
for i in range(3012):
    for j in range(25):
        w[i][j] = 1.0 + random.random()
for i in range(25):
    for j in range(8447):
        h[i][j] = 1.0 + random.random()

def D(w, h, x, t):
    count = 0
    listo = []
    for i in range(t):
        m = np.dot(w, h)
        m = m + 10**(-25)
        
        count += 1
        print count
        x_wh = x/m
        
        wt = w.T
        one = np.ones((3012, 3012))
        d_1 = np.dot(wt, one)
        wover = wt/d_1
        h = h * np.dot(wover, x_wh)
        
        m = np.dot(w, h) + 10**(-25)
        x_wh = x/m
        ht = h.T
        one = np.ones((8447, 8447))
        d_2 = np.dot(one, ht)
        hover = ht/d_2
        w = w * np.dot(x_wh, hover)
        
        m = np.dot(w, h) + 10**(-25)
        
        obj = -np.sum(np.add(x*np.log(m), -m))
        listo.append(obj)
    #         for i in range(3012):
    #             for k in range(25):
    #                 sum1 = 0
    #                 for j in range(8447):
    #                     if m[i][j] == 0:
    #                         m[i][j] = 10**(-16)
    #                     sum1 += h[k][j]*x[i][j]/m[i][j]
    #                 sum2 = np.sum(h[k])
    #                 if sum2 == 0:
    #                     sum2 = 10**(-16)
    #                 w[i][k] = w[i][k]*sum1/sum2
    #         count += 1
    #         print count
    #         for k in range(25):
    #             for j in range(8447):
    #                 sum1 = 0
    #                 for i in range(3012):
    #                     if m[i][j] == 0:
    #                         m[i][j] = 10**(-16)
    #                     sum1 += w[i][k]*x[i][j]/m[i][j]
    #                 sum2 = np.sum(w[:,k])
    #                 if sum2 == 0:
    #                     sum2 = 10**(-16)
    #                 h[k][j] = h[k][j]*sum1/sum2
    return listo, w, h

listob, wf, hf = D(w, h, x, 101)

fig = plt.figure()
plt.plot(listob[1:])
plt.xlabel('Iteration time')
plt.ylabel('Objective result')
plt.legend()
plt.show()
#print w
for j in range(25):
    sumj = np.sum(w[:, j])
    for i in range(3012):
        w[i][j] = w[i][j]/sumj

index = np.zeros((25, 10))
value = np.zeros((25, 10))
words = []
for i in range(25):
    index[i] = np.argsort(w[:,i])[::-1][:10]
for i in range(25):
    for j in range(10):
        index[i][j] = int(index[i][j])
print index
for i in range(25):
    for j in range(10):
        #print index[i][j]
        value[i][j] = w[int(index[i][j])][i]
print value

word = []
f = open(r"/Users/zhengqi/Downloads/machine-learning/hw5/hw5-data/nyt_vocab.dat", "r")
line = f.readline()
while line:
    a = line[:-1]
    word.append(a)
    line = f.readline()
f.close()

for i in range(25):
    for j in range(10):
        words.append(word[int(index[i][j])])

count = 0
for i in range(25):
    list1 = []
    for j in range(10):
        list1.append(words[10*count+j])
    print list1
    count += 1

csvFile2 = open('/Users/zhengqi/Downloads/machine-learning/hw5/hw5-data/2.csv','w') # 设置newline，否则两行之间会空一行
writer = csv.writer(csvFile2)
for i in range(25):
    for j in range(10):
        list2 = []
        list2.append(index[i][j])
        list2.append(words[10*i+j])
        list2.append(value[i][j])
        writer.writerow(list2)
csvFile2.close()



