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
picsize = (10,10)
# Various raw data transformations to graph
x = np.linspace(0,n1, n)
y1 = data2

# perform fft's
c1 = np.fft.fft(y1)
c2 = c1[np.abs(c1)>n1]
y2 = np.fft.ifft(c2).real/len(c2)

# reshape the altered data to produce a compressed image
n2 = np.sqrt(len(y2)).astype(int)

dataB = y2
dataB = dataB[:n2**2]
dataB = np.reshape(dataB, (n2,n2))


    


# Debug statements
#print(dataB)
#print(len(c1), 'length of c1')
#print('sqrt c1', np.sqrt(len(c1)))
#print(len(c2), 'length of c2')
#print('sqrt c2', np.sqrt(len(c2)))
#print(len(y2), 'length of y2')
#print('sqrt y2', np.sqrt(len(y2)))


# Graphs
    # Original Image plot
plt.figure(figsize = picsize)
img1 = plt.imshow(dataA)
plt.savefig('circular.png')
plt.show()

    # compressed image plot
plt.figure(figsize = picsize)
img2 = plt.imshow(dataB)
plt.savefig('circular2.png')
plt.show()

    # graph of fourier transform
plt.figure(figsize = picsize)
plt.plot(x,np.abs(c1), '-b',label = 'Fourier transform')
plt.savefig('Fourier.png')
plt.show()


    # logarithmic graph of data prior to transform
plt.figure(figsize = picsize)
plt.semilogx(x,y1, 'r-', label = 'Pre-Fourier plot')
plt.legend(loc='upper left')
plt.savefig('prefourier.png')
plt.show()

    # graph detailing magnitude of coefficients
plt.figure(figsize = picsize)
x2 = np.linspace(0,n1, len(y2))
plt.semilogx(x,y1, 'o-', label='original')
plt.semilogx(x2,y2, 'g-', label='Fourier')
plt.legend(loc='upper right')
plt.savefig('loggraph.png')
plt.show()


