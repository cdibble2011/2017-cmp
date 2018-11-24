import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random
import pandas as pd

# given fuction
x_min, x_max = -2, 2                          # range of x
y_min, y_max = -2, 2                          # range of x

# number of iterations
N = 100000

# initialize the arrays
results = []
xa = []
ya = []

# define the function to check
def f2(x,y):
    return 1/2*x**2 + 1/4*y**2

# define the monte carlo function
def MC1(N):
    count = 0
    for i in range(N):
        x = 2*random()
        y = 2*random()
        # fill the data arrays
        results.append(f2(x,y))
        xa.append(x)
        ya.append(y)
    # return the minimum of the function
    return min(results)

# print the minimum of f2(x,y)
print("the minimum of the function is: ", MC1(N))

# init the data frame
raw_data = {'x'    :  xa,
            'y'    :  ya,
            'z'    :  results}

df = pd.DataFrame(raw_data, columns = ['x','y','z'])

# print the minimum x,y,z values
print("minimums for x,y, an z are: ")
print(df[['x', 'y','z']][df['z'] == df['z'].min()])

# generate the 3d minimum point to graph from the data frame
x0 = df['x'][df['z'] == df['z'].min()]
y0 = df['y'][df['z'] == df['z'].min()]
z0 = df['z'][df['z'] == df['z'].min()]
# segregate the tested values to graph from the data frame
x1 = df['x']
y1 = df['y']
z1 = df['z']

# generate the 3d graph
nx = np.linspace(x_min,x_max,100)
ny = np.linspace(y_min,y_max,100)

x,y = np.meshgrid(nx,ny)

z = f2(x, y)
fig = plt.figure()
ax = fig.gca(projection='3d')
# plot the contour
ax.plot_surface(x,y,z, cmap='summer', alpha=0.8)
# plot the minimum
ax.scatter(x0,y0,z0, zdir = 'z', s = 100, c = 'red')
# plot the tested values
ax.scatter(x1,y1,z1, zdir = 'z', s = 10, c = 'blue')
cset = ax.contour(x,y,z, zdir='z', offset=0)
# label the graph
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f')

plt.show()