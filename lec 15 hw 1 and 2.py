# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 02:52:16 2018

@author: towar
"""


import numpy as np

N0 = 1000000
N1 = 0
N2 = 0
N3 = 0

np.random.seed()


x1 = np.zeros(N0)
x2 = np.zeros(N0)
x3 = np.zeros(N0)

for i in range(N0):
    
    x1[i] = np.random.randint(1,7, dtype = int)
    x2[i] = np.random.randint(1,7, dtype = int)
    x3[i] = x1[i] + x2[i]
    if x1[i] == 6 and x2[i] == 6:
        N1 = N1 + 1

    
#print('x1',x1)
#print('x2', x2)
print('double sixes: ', N1)
print('probability of double six: ', 100*N1/N0, '%')
