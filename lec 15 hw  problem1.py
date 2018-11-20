%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# number of iterations
N = 10001

# range of x,y
a,b = -2,2                                    
x_min, x_max = a, b                           
y_min, y_max = a, b

xtested = np.zeros(N+1)
ytested = np.zeros(N+1)

# f(x,y)
h = lambda x: x*x/2
g = lambda x: h(x)/2
f = lambda x,y: h(x) + g(y)
# weighted function of x
wx = lambda x: (np.sqrt(2)*x - 1)
# weighted function of y
wy = lambda x: (y - 1)
# weighted function of x and y
w = lambda x,y: wx(x)*wy(y)
# integrated weighted function of x and y
p = lambda x,y: w(x,y)/((np.abs(x_min)+np.abs(x_max))*(np.abs(y_min) + np.abs(ymax)))
# probability density function

x0 = np.zeros(N)
y0 = np.zeros(N)

# randomly pick a point in the given range
def xoyotest():
    i = 0
    j = 0
    while i != N:
        i += 1
        xtest = x_min + np.random.random()*(x_max-x_min) 
        ytest = y_min + np.random.random()*(y_max-y_min) 
        xtested[i] = xtest
        ytested[i] = ytest
        zmin = f(xtest,ytest)
        if zmin <.005:
            j += 1
            x0[j] = xtest
            y0[j] = ytest
            
xoyotest()
x0 = np.trim_zeros(x0)
y0 = np.trim_zeros(y0)
#xtested = np.trim_zeros(xtested)
#ytested = np.trim_zeros(ytested)

x0range = np.linspace(a,b,len(x0))
y0range = np.linspace(a,b,len(y0))

reasonabletests = np.round(100*len(x0)/N,2)

print('x tests: ',xtested)
print('y tests: ',ytested)
print('x0: ', x0)
print('y0: ', y0)
print('percentace of successful minimum tests: ', reasonabletests,'%')

# plots of:

    # x vs y
plt.plot(x0, y0, 'yo')
plt.plot(x0range, f(x0range,y0range), 'go')
plt.text(0, -1, 'x vs y graph', fontsize=15)
plt.show()

    # x vs z
plt.plot(x0range, f(x0range,y0range))
plt.plot(x0range, f(x0,y0), 'ro')
plt.text(0, -1, '$x_0$ vs z graph', fontsize=15)
plt.show()

    # y vs z
plt.plot(y0range, f(x0range,y0range))
plt.plot(y0range, f(x0,y0), 'go')
plt.text(0, -1, '$y_0$ vs z graph', fontsize=15)
plt.show()

    # x,y vs z
nx = np.linspace(x_min,x_max,N)
ny = np.linspace(y_min,y_max,N)

x3D,y3D = np.meshgrid(nx,ny)

z = f(x3D, y3D)
z0 = f(x0,y0)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x3D,y3D,z, cmap='summer', alpha=0.8)
ax.scatter(x0,y0,z0, zdir = 'z', s = 30, c = 'red')
cset = ax.contour(x3D,y3D,z, zdir='z', offset=0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('where z = 0')
plt.show()

