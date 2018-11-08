# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 23:50:01 2018

@author: towar
"""
# MC integral

# libraries
import matplotlib.pyplot as plt
from random import random
import numpy as np
from scipy.integrate import quad

# input and constants
N= input(prompt = "enter a number of iterations: ")
N = int(N)

integrals = input(prompt = "enter which integral you would like to evaluate (1 or 2): ")
integrals = int(integrals)

a, b = 0, 1
x = np.linspace(a,b,N)

# functions
def f(x):
    if integrals == 1:
        y = 1/np.sqrt(x)/(1 + np.exp(x))
    if integrals == 2:
        y = 1/x/(1 + np.exp(x))
    return y

def w(x):
    if integrals == 1:
        y = 1/np.sqrt(x) 
    if integrals == 2:
        y = 1/x
    return y

def p(x):
    if integrals == 1:
        y = x*x
    if integrals == 2:
        y = x
    return y

# improved monte carlo algo
def IMC(N):
    constant = quad(w,a,b)[0]
    I = 0
    for i in range(N):
        x = random()
        y = p(x)
        I += f(y)/w(y)
    return I/N*constant

# OG monte carlo
def MC(N):
    I = 0
    for i in range(N):
        x = random()
        I += f(x)
    return I/N

# General guesses of MC algos and comparison
res_IMC = []
res_MC = []
for i in range(N):
    res_IMC.append(IMC(N))
    res_MC.append(MC(N))

# plot of guesses
plt.figure(figsize = (15,15))
plt.plot(res_IMC, label='IMC')
plt.plot(res_MC, label='MC')
plt.legend()
plt.show()

# plot of of curves
plt.figure(figsize = (15,15))
plt.plot(x, f(x), label='f(x)')
plt.plot(x, w(x), label='w(x)')
plt.show()

# plot of weighting
res = []
for i in range(N):
    x = random()
    res.append(x*x)
    
plt.figure(figsize = (15,15))
plt.hist(res, bins=100)
plt.show()

# output of answers
print('improved: ', IMC(N))
print('original: ', MC(N))
print('from scipy: ', quad(f, a, b)[0])