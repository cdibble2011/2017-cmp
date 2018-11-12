# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 03:41:19 2018

@author: towar
"""

# A code to generate pseudo random numbers

import matplotlib.pyplot as plt
import numpy as np

# a few constants
N = 1000
N2 = 2*N +1
notzero = 1 

# define some storage space
resultsx = np.zeros(N2)
resultsacm = []

# storaage for graphing a,c,m values as tuples of x y z
# m
mline = []
# a
aline = []
#c
cline = []

# define function to test N number of different a c and m values
# then store them to graph later

def randoint(x,N2):
    # random variable to decide what is positive or negative
    randomvarx = 1
    randomvara = -1
    randomvarc = -1
    # counter variables
    acmcount = 0
    l = 0
    # init random tries
    x = np.random.randint(11) + notzero*randomvarx
    a = np.random.randint(21) + notzero*randomvara
    c = np.random.randint(3) + notzero*randomvarc
    m = np.random.randint(4) + notzero
    #print(acmcount, 'acmcount')
    while l < N2:    
        x = (a*x+c)%m
        resultsx[l] = x 
        l += 1
    # test random tries to see if they fit the criteria
    for i in range(N2):
        if -6 < resultsx[i] and resultsx[i] < 6:
            acmcount += 1
        # store a,c,m values if they fit
        if acmcount == N2:
            resultsacm.append((a,c,m))
            aline.append(a)
            cline.append(c)
            mline.append(m)
            #print(resultsx)
            
    #return resultsx

# run the function
for i in range(N):
    randoint(i,N2)

# print the resulting successes
print(resultsacm)

# plot successes to try and visualize a pattern
plt.figure(figsize = (15,15))
ax = plt.axes(projection='3d')

# Data for a three-dimensional line

ax.plot3D(aline, cline, mline, 'none')

# Data for three-dimensional scattered points
ax.scatter3D(aline, cline, mline, c='r', cmap='Greens');