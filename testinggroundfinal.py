%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as optimize
import numdifftools as nd
import math
import timeit

N = 100
prec = .0001

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
    #print('[x0,y0]:', [x0,y0])
    return [x0, y0]

# if gk = 0, stop, else dk = -Hk*g(k)
def term_test(xk,f):
    G = nd.Gradient(f)
    if np.linalg.norm(G(xk)) <= prec:
        #print('term_test: True', np.linalg.norm(G(xk)))
        return True
    else:
        #print('term_test: False')
        return False
    
# reshapes results appropriately
def array_fix(x):
    x = [x[0,0],x[1,1]]
    x = np.asarray(x,order = 1)
    return x

# wrap the function to sent to timeit
# from https://www.pythoncentral.io/time-a-python-function/
def wrapper(func,f,x):
    def wrapped():
        return func(f,x)
    return wrapped


# Conjugate Gradient Descent
# Accept the function, its limits, 
# and the expected value provided by benchmark
# x(n+1) = x[n] - gamma[n]*GradF(x[n])
# variable stepsize
def GD_min1(f,x_init): 
    # N = Number of intervals
    # xa = storage array for values to return
    # i = counter
    # init(ax,bx,ay,by) = function call to gen random
    # values for initial guess
    # prec = .0001
    # gamma = prec is initial step size so algo
    # doesn’t overshoot on the first run
    xa = []
    i = 0
    x_init = x_init[0:2]
    x_now = x_init
    gamma = prec
    converged = False
    xa.append(x_now)
    
    # loop the algo until the term_test conditions
    # are satisfied and it returns a ‘True’ value
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
            gamma = b*c/np.linalg.norm(c)**2
            x_now = x_next
            xa.append(x_now)
            i += 1

    # reshape ‘xa’ as a (-1,2) array to return
    # f_min: return the minimum value, or how close
    # the algo got to zero    
    # return the number of iterations 
    # for comparison later
    xa = np.array(xa)
    f_min = f(x_now)
    return i, f_min, x_now

    
def CD_min1(f, x_init):
    #print('x_init:', x_init)
    x_init = x_init[0:2]
    #print('x_init:', x_init)
    xa = []
    np.asarray(xa)
    # counters
    k = 0
    # inital values
    x_now = x_init
    x0 = x_init
    A = nd.Hessian(f)
    A = A(x_now)
    
    # internal functions
    def CD_algo_init(x):
        r = -A*x
        p = r
        return r,p
        
    def a_now(rk,pk):
        ak = rk.T*rk/(pk.T*A*pk)
        return ak
        
    def x_k1(xk,ak,pk):
        xk1 = xk+ak*pk
        return xk1
    
    def r_k1(rk,A,ak,pk):
        rk1 = rk - ak*A*pk
        return rk1
    
    def Beta_k(rk1,rk):
        Bk = rk1.T*rk1/(rk.T*rk)
        return Bk
    
    def p_k1(rk1,Bk,pk):
        pk1 = rk1 +Bk*pk
        return pk1
    
    # iterate through algo until a zero appears
    rk, pk = CD_algo_init(x0)
    while np.abs(f(x_now)) > prec or k < N:
        ak = a_now(rk,pk)
        #print('ak:',ak)
        x_next = x_k1(x_now,ak,pk)
        #print('x_next: ',x_next)
        rk1 = r_k1(rk,A,ak,pk)
        #print('rk1:', rk1)
        if np.linalg.norm(rk1) <= prec:
            xa.append(x_next)
            k += 1
            break
        else:
            Bk = Beta_k(rk1,rk)
            pk = p_k1(rk1,Bk,pk)
            x_now = x_next
            xa.append(x_now)
            #xa.append(x_now[0,1])
            k += 1
    
    # reshape the data from a mixed list
    # of floats and arrays to a list of 
    # useable values
    for i in range(k):
        x = xa[i]
        x1 = x[0]
        x1a = x1[0]
        x1b = x1[1]
        x2 = x[1]
        x2a = x2[0]
        x2b = x2[1]
        x0.append(x1a)
        x0.append(x1b)
        x0.append(x2a)
        x0.append(x2b)
    
    # reshape list into an array of coordinates
    x0 = np.asarray(x0, dtype = np.float32)
    x0.shape = (-1,2)
    f_min = f(x_now)
    #print('f_min:',f_min)
    return k, f_min, x_now


# Quasi-Newton
# Symmetric Rank-1 
# Accept the function, its limits, 
# and the expected value 
# provided by benchmark
def Rank1_min(f, x_init):
    xa = []
    x_now = x_init[0:2]
    x_now = np.asarray(x_now)
    xa.append(x_now)
    converged = False
    G = nd.Gradient(f)
    H = nd.Hessian(f)
    
    # initial function values
    x0 = x_now[0:2]    
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
    
    # Step Size a(k)
    def a_now(H,xk,f):
        a = np.linalg.norm(f(xk))
        tau = np.linspace(1,0,N)
        c = np.linspace(0,1,N)
        c = c[np.random.randint(0,N)]
        m = dk(H,xk)
        m = array_fix(m)
        p = m
        m = np.dot(np.transpose(m),G(xk))
        t = -c*m
        j = 0
        a_converge = False
        # Backtracking Line Search to find
        # appropriate value of ‘a’
        while a_converge == False or j <= N:
            ajt_test = f(xk) - f(xk + np.multiply(a,p))
            j += 1
            if a*t <= ajt_test:
                a_converge = True
                return a
            if j == N:
                return a
            else:
                a = a*tau[j]
    
    # Compute x(k+1)
    def x_next(xk,f,H):
        direction = dk(H,xk)
        direction = array_fix(direction)
        return xk + a_now(H,xk,f)*direction
    
    # Delta x(k)
    def delta_xk(xk1, xk):
        return xk1 - xk
    
    # Delta x(k), or change in the gradient
    def delta_gk(xk,xk1):
        return H(xk1 - xk)
    
    # H(k+1) = H + some stuff
    def H_formula(H,xk,xk1,f):
        a = delta_gk(xk,xk1)
        b = delta_xk(xk,xk1)
        c = H*a
        d = b - c
        e = (H + d*d.T/a.T/c)
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
    return k, f_min, x_now


scipy.optimize.fmin_cg(f1,x0y0[0],full_output =True)
    
    
    
XY = [-2,2]    
# initial values
x0y0 = []
# indexed iteration counters
iG = []
iC = []
iS = []

# indexed f_minimums
fG = []
fC = []
fS = []

# indexed XY where minimum is found
xG = []
xC = []
xS = []

# indexed timer
TG = []
TC = []
TS = []
# generate standardized data
for i in range(N):
    XYZ = init(-2,2,-2,2)
    x0y0.append(XYZ)

# time the functions
for i in range(N):   
    wrappedC = wrapper(CD_min1,f1,x0y0[i])
    tC = timeit.timeit(wrappedC, number = 10)
    print('CD timeit', tC)
    wrappedG = wrapper(GD_min1,f1,x0y0[i])
    tG = timeit.timeit(wrappedG, number = 10)
    print('GD timeit', tG)
    wrappedS = wrapper(Rank1_min,f1,x0y0[i])
    tS = timeit.timeit(wrappedS, number = 10)
    print('SR1 timeit', tS)

    TG.append(tG)
    TC.append(tC)
    TS.append(tS)

# run some data to make comparisons
for i in range(N):
    jg, fg, xg = GD_min1(f1, x0y0[i])
    jc, fc, xc = CD_min1(f1,x0y0[i])
    js, fs, xs = Rank1_min(f1,x0y0[i])
  
    iG.append(jg)
    fG.append(fg)
    xG.append(xg)
        
    iC.append(jc)
    fC.append(fc)
    xC.append(xc)
    
    iS.append(js)
    fS.append(fs)
    xS.append(xs)    
    
    
xmagG = []
xmagC = []
xmagS = []

for i in range(N):
    x = 0 - np.abs(mag(xG[i]))
    xmagG.append(x)
    x = 0 - np.abs(mag(xC[i]))
    xmagC.append(x)
    x = 0 - np.abs(mag(xS[i]))
    xmagS.append(x)
    
    
# iteration comparison
# exclude CD as it universally
# takes 1 step
a, b = 0, N
x = np.linspace(a,b,N)
yg1 = iG
#yc1 = iC
ys1 = iS
fig1 = plt.figure()
plt.title('iteration comparison')
plt.xlabel('index number')
plt.ylabel('# of iterations to find a zero')
plt.plot(x,yg1, label = 'Grad Desc', color = 'red')
plt.plot(x,ys1, label = 'Quas Newt', color = 'blue')
plt.legend(loc='lower right')
plt.show()

# minimization comparison
# exclude CD as it is in a league
# of its own
x = np.linspace(a,b,N)
yg2 = fG
ys2 = fS
fig2 = plt.figure()
plt.title('minimization comparison GD vs SR1')
plt.xlabel('index number')
plt.ylabel('distance from zero')
plt.plot(x, yg2, color = 'red',label = 'Grad Desc')
plt.plot(x,ys2, color = 'blue', label = 'Quas Newt')
plt.legend(loc='lower right')
plt.show()

# CD specific minimaztion view
x = np.linspace(a,b,N)
yc2 = fC
fig3 = plt.figure()
plt.title('minimization view of Conj Desc algo')
plt.xlabel('index number')
plt.ylabel('distance from zero')
plt.plot(x,yc2, color = 'black', label = 'Conj Desc')
plt.legend(loc='lower right')
plt.show()

#Magnitude of distance from analytic zero
x = np.linspace(a,b,N)
yc3 = xmagC
fig5 = plt.figure()
plt.title('Magnitude of distance from analytic zero, Conj Desc')
plt.xlabel('index number')
plt.ylabel('distance from zero')
plt.plot(x,yc3, color = 'black', label = 'Conj Desc')
plt.legend(loc='lower right')
plt.show()

x = np.linspace(a,b,N)
yg3 = xmagG
ys3 = xmagS
fig6 = plt.figure()
plt.title('Magnitude of distance from analytic zero, GD vs SR1')
plt.xlabel('index number')
plt.ylabel('distance from zero')
plt.plot(x, yg3, color = 'red',label = 'Grad Desc')
plt.plot(x,ys3, color = 'blue', label = 'Quas Newt')
plt.legend(loc='lower right')
plt.show()


x = np.linspace(a,b,N)
yg4 = TG
ys4 = TS
fig8 = plt.figure()
plt.title('Runtime Comparison')
plt.xlabel('index number')
plt.ylabel('Average runtime for 100 iterations of given [x0,y0], GD and SR1')
plt.plot(x, yg4, color = 'red',label = 'Grad Desc')
plt.plot(x,ys4, color = 'blue', label = 'Quas Newt')
plt.legend(loc='lower right')
plt.show()

x = np.linspace(a,b,N)
yc4 = TC
fig9 = plt.figure()
plt.title('Runtime Comparison')
plt.xlabel('index number')
plt.ylabel('Average runtime for 100 iterations of given [x0,y0], Conj Desc')
plt.plot(x,yc4, color = 'black', label = 'Conj Desc')
plt.legend(loc='lower right')
plt.show()