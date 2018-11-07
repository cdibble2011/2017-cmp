# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:54:26 2018

@author: towar
"""

# MC integral
import matplotlib.pyplot as plt
from math import sin
from random import random
import numpy as np

N= 10000

def f(x):
    y = np.sqrt(1-(x-1)**2)
    return y

def g(x):
     y = 2-np.sqrt(4-x**2)
     return y 
    
def MC(N):
    count = 0
    for i in range(N):
        x = 2*random()
        y = random()
        if g(x)<y<f(x):
            count += 1
    I = 2*count/N
    return I

x1 = np.linspace(0,2,N)

fig1 = plt.figure(figsize = (15,15))
ax1 = fig1.add_subplot(111, aspect='equal')
ax1.plot(x1,f(x1),'k')
ax1.plot(x1,g(x1),'k')
ax1.set_xlim(0,2)
ax1.set_ylim(0,2)
ax1.fill_between(x1, f(x1), g(x1), where=f(x1)>g(x1), facecolor='green')

print(MC(N))