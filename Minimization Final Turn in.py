%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as optimize
import numdifftools as nd
import math

N = 1000
prec = .001

# Limits
# -(3 char identifier) <= x,y <= +(3 char identifier)
    # Title: given function 
    # Func ID: f1
    # Limits: -2 <= x,y <= 2
    # Expected returns:
    # Min = f(0,0) = 0
# Limit
f10 = 2
# Expected values
f1E = [0.0,0.0,0.0]
# Title
f1N = 'Given'

# Benchmark/test functions
    # given function for project
    #       -2 <= x,y <= 2
    #       f(x,y) = x**2/s + y**2/3 -xy/4
    # Min = f(0,0) = 0
def f1(x): 
    x, y = x[0], x[1]
    return x**2/2 + y**2/3 - x*y/4

# magnitude of a vector
def mag(x):
    return np.sqrt(x[0]**2 + x[1]**2)

# variable stepwidth
def derivative2(f, x, d=0.001):
    x, y = x[0], x[1]
    fx = (f([x+d/2,y])-f([x-d/2,y]))/d
    fy = (f([x,y+d/2])-f([x,y-d/2]))/d
    return np.array([fx,fy])
    
# Defines a random starting point for the optimization function as given in lecture
def init(x_min, x_max, y_min, y_max):
    x0 = x_min+np.random.random()*(x_max-x_min)
    y0 = y_min+np.random.random()*(y_max-y_min)
    return [x0, y0]      
    
# reshapes results appropriately
def array_fix(x):
    x = [x[0,0],x[1,1]]
    x = np.asarray(x,order = 1)
    return x

# if gk = 0, stop, else dk = -Hk*g(k)
def term_test(xk,f):
    G = nd.Gradient(f)
    if np.linalg.norm(G(xk)) <= prec:
        print('term_test: True', np.linalg.norm(G(xk)))
        return True
    else:
        print('term_test: False')
        return False

    
# optimize via scipy for the sake of comparison
def scipymin(f,xlims, ylims):
    ax,bx = xlims[0],xlims[1]
    ay,by = ylims[0],ylims[1]
    initial_guess = init(ax,bx,ay,by)
    result = optimize.minimize(f, initial_guess)
    if result.success:
        fitted_params = result
        print("scipy's optimize results: ",fitted_params)
    else:
        raise ValueError("what scipy has to say: ",result.message)
    filter(lambda v: v==v, result)
        
    return result.x, result.nit
  
# given minimization function from
# lecture 16 for comparison's sake

def minimize_var(f,xlims, ylims):    
    # initial values
    ax,bx = xlims[0],xlims[1]
    ay,by = ylims[0],ylims[1]
    x_now = init(ax,bx,ay,by)
    x_prev = None
    converged = False
    # points
    x_hist = []
    x_hist.append(x_now)
    i = 0
    # algo
    while converged == False:
        df_now = derivative2(f, x_now) 
        if x_prev is None:
            dx = 0.01
        else:
            df_prev = derivative2(f, x_prev)
            dd = df_now - df_prev
            dx = np.dot(x_now - x_prev, dd) / (np.linalg.norm(dd))**2
        x_next = x_now - df_now*dx
        i += 1
        # output
        #print("step:    ", f(x_now), f(x_next))
        if f(x_next)>f(x_now):
            converged = True
            break
        else:
            x_prev = x_now
            x_now = x_next
            x_hist.append(x_now)
    x_hist = np.array(x_hist)
    return x_hist, i


# Conjugate Gradient Descent
# Accept the function, its limits, 
# and the expected value provided by benchmark
# x**(n+1) = x[n] - gamma[n]*GradF(x[n])
# variable stepsize
def GD_min1(f,xlims, ylims):
    xa = []
    i = 0
    ax,bx = xlims[0],xlims[1]
    ay,by = ylims[0],ylims[1]
    x_now = init(ax,bx,ay,by) 
    gamma = prec
    converged = False
    xa.append(x_now)
    while converged == False or i < N:
        converged = term_test(x_now,f)
        if converged == True:
            break
        else:
            df = nd.Gradient(f)
            x_next = x_now - gamma*df(x_now)
            a = (x_next - x_now)
            b = a.T
            c = (df(x_next)- df(x_now))
            gamma = b*c/mag(c)**2
            x_now = x_next
            xa.append(x_now)
            i += 1
            
    xa = np.array(xa)
    f_min = f(x_now)
    #print('best guess min: ',f_min)
    #print('min xy vals: ', xa)
    return xa, i

# Quasi Newton
# Rank 1 Algo
def Rank1_min(f,xlims, ylims):
    xa = []
       
    ax,bx = xlims[0],xlims[1]
    ay,by = ylims[0],ylims[1]
    x_now = init(ax,bx,ay,by)
    # debug values
    #x_now = [2,2]
    x_now = np.asarray(x_now)
    xa.append(x_now)
    converged = False
    G = nd.Gradient(f)
    H = nd.Hessian(f)
    
    # initial function values
    x0 = x_now    
    f0 = f(x0)        
    g0 = G(x0)
    H0 = np.identity(2)
    k = 0 
    L = 1
    # termination condition
    epsilon = prec

    # Direction d(k)
    def dk(H,xk):        
        return -H*G(xk)
    
    # Step size a(k)
    def a_now(H,xk,f):
        a = np.linalg.norm(f(xk))
        tau = np.linspace(1,0,N)
        c = np.linspace(0,1,N)
        c = c[np.random.randint(0,N)]
        # print() is a debug statement here
        #print('c',c)
        m = dk(H,xk)
        #print('m initial', m)
        m = array_fix(m)
        p = m
        #print('m adjusted', m)
        #print('G(f(xk))', G(xk))
        #print('m array', m)
        m = np.dot(np.transpose(m),G(xk))
        #print('m dot product', m)
        t = -c*m
        j = 0
        a_converge = False
        #print('f(xk)', f(xk))
        #print('p = dk(xk,H)', p)
        
        # Backtracking Line Search to find appropriate value of 'a'
        while a_converge == False or j <= N:
            ajt_test = f(xk) - f(xk +np.multiply(a,p))
            j += 1
            #print('a-test result, j', ajt_test, j)
            if a*t <= ajt_test:
                a_converge = True
                return a
            if j == N:
                return a
            else:
                a = a*tau[j]
                #print('new a', a)
    
    # Compute x(k+1)
    def x_next(xk,f,H):
        direction = dk(H,xk)
        direction = array_fix(direction)
        return xk + a_now(H,xk,f)*direction
    
    # Delta x(k)
    def delta_xk(xk1, xk):
        return xk1 - xk
    
    # Delta g(k), or change in the gradient
    def delta_gk(xk,xk1):
        return H(xk1 - xk)
    
    # H(k+1) = H + some stuff
    def H_formula(H,xk,xk1,f):
        a = delta_gk(xk,xk1)
        b = delta_xk(xk,xk1)
        c = H*a
        d = b - c
        e = (H + d*d.T/a.T/c)
        #print('New H: ', e)
        #print('k', k)
        return e
    
    def algo(xk,H,f):
        xk1 = x_next(xk,f,H)
        Hk1 = H_formula(H,xk,xk1,f)
        return xk1, Hk1
    
    x_now, H_now = algo(x_now,H0,f)
    xa.append(x_now)
    converged = term_test(x_now,f)
    k += 1
    while converged == False:
        x_now, H_now = algo(x_now, H_now,f)
        xa.append(x_now)
        k += 1
        converged = term_test(x_now,f)
   
    xa = np.array(xa)
    xa.shape = (-1,2)
    f_min = f(x_now)
    #print('best guess min: ',f_min)
    #print('min xy vals: ', xa)
    return xa, k
           
# Generate 2 graphs,
# surface plot, and line plot of 
# benchmark function and
# minimization function results
# f         = benchmark function
# ID        = max/min
# X         = results to graph
# niter     = number of iterations

def Graphit2D(f, ID, X, niter, name):
    # x,y limits based on ID of the function
    a, b = ID[0], ID[1]
    # x,y values returned from optimization function
    x, y = X[:,0], X[:,1]
    # meshgrid definitions
    nx = np.linspace(a,b,N)
    ny = np.linspace(a,b,N)
    x_mesh, y_mesh = np.meshgrid(nx,ny)
    f_mesh = f([x_mesh,y_mesh])
    fig = plt.figure()
    plt.title(name)
    plt.xlabel('x')
    plt.ylabel('y')
    levels = np.arange(np.min(f_mesh), np.max(f_mesh), .3)
    Graph = plt.contour(x_mesh, y_mesh, f_mesh, levels)
    plt.clabel(Graph, inline=1, fontsize=10)    
    
    plt.plot(x,y, 'ro-')
    plt.show()
    print('Number of iterations: ', niter)

def Graphit2Dsp(f, ID, X, niter, name):
    # x,y limits based on ID of the function
    a, b = ID[0], ID[1]
    # x,y values returned from optimization function
    x, y = X[0], X[1]
    # meshgrid definitions
    nx = np.linspace(a,b,N)
    ny = np.linspace(a,b,N)
    x_mesh, y_mesh = np.meshgrid(nx,ny)
    f_mesh = f([x_mesh,y_mesh])
    fig = plt.figure()
    plt.title(name)
    plt.xlabel('x')
    plt.ylabel('y')
    levels = np.arange(np.min(f_mesh), np.max(f_mesh), .3)
    Graph = plt.contour(x_mesh, y_mesh, f_mesh, levels)
    plt.clabel(Graph, inline=1, fontsize=10)    
    
    plt.plot(x,y, 'ro-')
    plt.show()
    print('Number of iterations: ', niter)
    
def Graphit3D(f, ID, X, niter,name):
    a, b = ID[0], ID[1]
    x, y = X[0], X[1]
    # Results coordinates in 3d
    xc, yc = np.meshgrid(x,y)
    zc = f([xc,yc])
    
    # 3d function plot
    nx = np.linspace(a,b,N)
    ny = np.linspace(a,b,N)
    xg, yg = np.meshgrid(nx,ny)
    zg = f([xg,yg]) 
   
    # plot it
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xg,yg,zg, cmap='summer', alpha=0.8)
    ax.scatter(xc,yc,zc, 'ro-', s=200)
    cset = ax.contour(xg,yg,zg, zdir='zg', offset=0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f')
    plt.show()
   # print('Number of iterations: ', niter)



XY = [-2,2]  

    
X1, I1 = Rank1_min(f1, XY, XY)
X2, I2 = scipymin(f1,XY,XY)
X3, I3 = minimize_var(f1, XY,XY)
X4, I4 = GD_min1(f1, XY,XY)

Graphit2D(f1, XY, X1, I1,'Rank1 algorithm')
Graphit3D(f1, XY, X1, I1,'Rank1 algorithm')

Graphit2Dsp(f1, XY, X2, I2,'scipys minimizer')
Graphit3D(f1, XY, X2, I2,'scipys minimizer')

Graphit2D(f1, XY, X3, I3,'Lecture provided minimizer')
Graphit3D(f1, XY, X3, I3,'Lecture provided minimizer')

Graphit2D(f1, XY, X4, I4,'Gradient Descent')
Graphit3D(f1, XY, X4, I4,'Gradient Descent')

