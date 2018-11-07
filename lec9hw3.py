# -*- coding: utf-8 -*-
"""
Spyder Editor

Curtis Dibble
Phys300
6 Nov 2018
Lecture 9 HW
"""
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from numpy.fft import *

# import and shape the data

website = 'http://www-personal.umich.edu/~mejn/cp/data/circular.txt'
#open url
response = urllib.request.urlopen(website)
data = response.read()      
data1 = data.decode('utf-8') 
data2 = np.fromstring(data1,sep=' ',dtype=float)
data3 = np.reshape(data2, (501,501))
dataA = data3

# some constants
n = len(data2)  # number of data points
n1 = 1000
# Various raw data transformations to graph
x = np.linspace(0,n1, n)
y1 = data2

c1 = np.fft.fft(y1)
c2 = c1[np.abs(c1)>n1]
y2 = np.fft.ifft(c1).real/len(c2)
y3 = np.fft.ifft(c2).real/len(c2)

print(len(c1), 'length of c1')
print('sqrt c1', np.sqrt(len(c1)))
print(len(c2), 'length of c2')
print('sqrt c2', np.sqrt(len(c2)))
print(len(y2), 'length of y2')
print('sqrt y2', np.sqrt(len(y2)))
print(len(y3), 'length of y3')
print('sqrt y3', np.sqrt(len(y3)))

# Graphs
    # Original Image plot
img1 = plt.imshow(dataA)
plt.savefig('circular.png')
plt.figure(figsize = (15,15))
plt.show()

fig, dataplot = plt.subplots()
plt.figure(figsize = (15,15))
dataplot.semilogx(x,y1, 'r-', label = 'Pre-Fourier plot')
dataplot.legend(loc='upper left')
plt.show()


fig, ax1 = plt.subplots()
plt.figure(figsize = (15,15))
ax1.plot(np.abs(c2), 'y-', label='Magnitude')
ax2 = ax1.twinx()
ax2.semilogy(np.angle(y3), 'g.-', label='Phase')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.savefig('fft2.png')
plt.show()


