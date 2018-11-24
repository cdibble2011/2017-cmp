%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

N = 1000
X = []

# function
def f1(x): 
    return x[0]**2/2 + x[1]**2/3 - x[0]*x[1]/4

# variable stepwidth
def derivative2(f, xy, d=0.001):
    x, y = xy[0], xy[1]
    fx = (f([x+d/2,y])-f([x-d/2,y]))/d
    fy = (f([x,y+d/2])-f([x,y-d/2]))/d
    return np.array([fx,fy])

# given minimization function from lecture 16
def minimize_var(f,x0, N):    
    # initial values
    x_now = x0
    x_prev = None
    converged = False
    # points
    x_hist = []
    x_hist.append(x_now)
    # algo
    for i in range(N):
        df_now = derivative2(f2, x_now) 
        if x_prev is None:
            dx = 0.01
        else:
            df_prev = derivative2(f2, x_prev)
            dd = df_now - df_prev
            dx = np.dot(x_now - x_prev, dd) / (np.linalg.norm(dd))**2
        x_next = x_now - df_now*dx
        # output
        print("step:    ", f(x_now), f(x_next))
        if f(x_next)>f(x_now):
            converged = True
            break
        else:
            x_prev = x_now
            x_now = x_next
            x_hist.append(x_now)

    return converged, np.array(x_hist), f(x_now)

# optimize via scipy
def scipymin(f,N):
    xa = np.exp(np.random.random())
    ya = np.exp(1/np.random.random())
    initial_guess = [xa,ya]
    result = optimize.minimize(f, initial_guess)
    if result.success:
        fitted_params = result
        print("scipy's optimize results: ",fitted_params)
           # J += 1
    else:
        raise ValueError("what scipy has to say: ",result.message)
    x = result.x[0]
    y = result.x[1]
    return x,y


# plot results of the given algo
[x0, y0] = init(x_min, x_max, y_min, y_max)
converged, x_hist, f_min = minimize_fix(f2, [x0,y0])
x,y = np.meshgrid(nx,ny)
z = f1([x, y])
fig = plt.figure()
levels = np.arange(np.min(z), np.max(z), 0.3)
plt.contour(x,y,z, levels=levels)

plt.plot(x_hist[:,0], x_hist[:,1], 'ro-')
plt.show()
print('results of f_min: ', f_min, '  number of iterations:  ', len(x_hist))
print(x_hist[0], f1(x_hist[0]))
print(x_hist[-1], f1(x_hist[-1]))

# find x0,y0 with scipy
spx , spy = scipymin(f1,N)

# plot it
[x0, y0] = init(x_min, x_max, y_min, y_max)
x,y = np.meshgrid(nx,ny)
z = f1([x, y])
fig = plt.figure()
levels = np.arange(np.min(z), np.max(z), 0.3)
plt.contour(x,y,z, levels=levels)

plt.plot(spx, spy, 'ro-')
plt.show()