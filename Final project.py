%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as optimize
import test
import pandas as pd
import numdifftools as nd

N = 1000
prec = .1

# Limits
# -(3 char identifier) <= x,y <= +(3 char identifier)
# Limits defined in function, by passing 3 char ID
# and setting the interval from there   
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


    # Title: Ackley function 
    # Func ID: f2
    # Limit ID: f2E
    # Limits: -5 <= x,y <= 5
    # Expected returns:
    # Min = f(0,0) = 0
# Limit
f20 = 5.0
# Expected values
f2E = [0.0,0.0,0.0]
# Title
f2N = 'Ackley'


    # Title: Beale function 
    # Func ID: f3
    # Limit ID: f3E
    # Limits: -4.5 <= x,y <= 4.5
    # Expected returns:
    # Min = f(3,0.5) = 0
# Limit
f30 = 4.5
# Expected values
f3E = [3.0,0.5,0.0]
# Title
f3N = 'Beale'


    # Title: Cross-in-tray function 
    # Func ID: f4
    # Limit ID: f4E
    # Limits: -10 <= x,y <= 10
    # Expected returns:
    #       { f(1.34941,  -1.34941)    = -2.0621
    #       { f(1.34941,   1.34941)    = -2.0621
    # Min = { f(-1.34941,  1.34941)    = -2.0621
    #       { f(-1.34941, -1.34941)    = -2.0621
# Limit
f40 = 10.0
# Expected values
f4E = [1.34941,1.34941,-2.0621]
# Title
f4N = 'Cross-in-Tray'


    # Title: Eggholder function 
    # Func ID: f5
    # Limit ID: f5E
    # Limits: -512 <= x,y <= 512
    # Expected returns:
    # Min = f(512, 404.2319) = -959.6407
# Limit
f50  = 512.0
# Expected values
f5E = [512, 404.2318,-959.6407]
# Title
f5N = 'Eggholder'


    # Title: Holder Table function 
    # Func ID: f6
    # Limit ID: f6E
    # Limits: -10 <= x,y <= 10
    # Expected returns:
    #      { f(8.05502,9.66459)      = -19.2085
    #      { f(-8.05502, 9.66459)    = -19.2085
    # Min= { f(8.05502, -9.66459)    = -19.2085      
    #      { f(-8.05502, -9.66459)   = -19.2085
# Limit
f60  = 10.0
# Expected values
f6E = [8.05502,9.66459,-19.2085]
# Title
f6N = 'Holder Table'

# Benchmark/test functions
    # given function for project
    #       -2 <= x,y <= 2
    #       f(x,y) = x**2/s + y**2/3 -xy/4
    # Min = f(0,0) = 0
def f1(x): 
    x, y = x[0], x[1]
    return x**2/2 + y**2/3 - x*y/4


    # Ackley function: 
    #       -5 <= x,y <= 5
    #       f(x,y) = -20exp(-.2sqrt(.5(x**2 + 
    #       y**2))) - exp(cos 2pi x + cos 2pi y) + e + 20, 
    #       f(x,y) = -20exp(a -b) + np.exp(1) + 20
    # Min = f(0,0) = 0
def f2(x):
    x, y = x[0], x[1]
    x2 = x**2
    y2 = y**2
    a = -.2*np.sqrt(.5*(x2 +y2))
    b = np.exp(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))
    c = -20*np.exp(a - b) + np.exp(1) + 20
    return c

    # Beale function
    #       -4.5 <= x,y <= 4.5
    #       f(x,y) = (1.5 - x + xy)**2 + 
    #       (2.25 - x +xy**2)**2 +
    #       (2.625 - x + xy**3)**2
    # Min = f(3,0.5) = 0
def f3(x):
    x, y = x[0], x[1]
    y2 = y**2
    y3 = y**3
    a = (1.5 - x + x*y)**2
    b = (2.25 - x +x*y2)**2
    c = (2.625 - x + x*y3)**2
    return a + b + c

    # Cross-in-tray function
    #       -10 <= x,y <= 10
    #       f(x,y) = -.0001(|abg|+1)**(1/10)
    #       a = sinx
    #       b = siny
    #       c = 100 - np.sqrt(x**2 + y**2)/pi
    #       d = np.exp(np.abs(c))
    #       g = np.abs(a*b*d) + 1
    #       h = -.0001*(g**.1)
    #       { f(1.34941,  -1.34941)    = -2.0621
    #       { f(1.34941,   1.34941)    = -2.0621
    # Min = { f(-1.34941,  1.34941)    = -2.0621
    #       { f(-1.34941, -1.34941)    = -2.0621
def f4(x):
    x, y = x[0], x[1]
    x2 = x**2
    y2 = y**2
    a = np.sin(x)
    b = np.sin(y)
    c = 100 - np.sqrt(x2 + y2)/np.pi
    d = np.exp(np.abs(c))
    g = np.abs(a*b*d) + 1
    return -.0001*(g**.1)

    # Eggholder function
    #       -512 <= x,y <= 512
    #       f(x,y) = -(y+47)sin(sqrt(abs(x/2 + (y + 47)) -
    #            xsin(sqrt(abs(x -(y + 47))))
    #       f(x,y) = -a*sin(sqrt(b)) - x*sin(sqrt(c))
    # Min = f(512, 404.2319) = -959.6407
def f5(x):
    x, y = x[0], x[1]
    a = y + 47
    b = np.abs(x/2 + a)
    c = np.abs(x - a)
    return -a*np.sin(np.sqrt(b)) - x*np.sin(np.sqrt(c))

    # Holder table function
    #      -10 <= x,y <= 10
    #      f(x,y) = -abs(sinxcosyexp(abs(1-sqrt(x2 + y2)/pi)))
    #      { f(8.05502,9.66459)      = -19.2085
    #      { f(-8.05502, 9.66459)    = -19.2085
    # Min= { f(8.05502, -9.66459)    = -19.2085      
    #      { f(-8.05502, -9.66459)   = -19.2085
def f6(x):
    x, y = x[0], x[1]
    x2, y2 = x**2, y**2
    a = np.sin(x)
    b = np.cos(y)
    c = np.sqrt(x2+y2)/np.pi
    d = np.exp(np.abs(1-c))
    return -np.abs(a*b*d)

# magnitude of a vector
def mag(x):
    return np.sqrt(x[0]**2 + x[1]**2)

# variable stepwidth
def derivative2(f, x, d=0.001):
    x, y = x[0], x[1]
    fx = (f([x+d/2,y])-f([x-d/2,y]))/d
    fy = (f([x,y+d/2])-f([x,y-d/2]))/d
    return np.array([fx,fy])

# Hessian Generator
#def Hessian_gen(f,X):
    

# Defines a random starting point for the optimization function as given in lecture
def init(x_min, x_max, y_min, y_max):
    x0 = x_min+np.random.random()*(x_max-x_min)
    y0 = y_min+np.random.random()*(y_max-y_min)
    return [x0, y0]       
            
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



# Accept the function, its limits, 
# and the expected value provided by benchmark
# x**(n+1) = x[n] - gamma[n]*GradF(x[n])
# fixed step
def GD_min1(f,xlims, ylims):
    i = 0
    ax,bx = xlims[0],xlims[1]
    ay,by = ylims[0],ylims[1]
    x_now = init(ax,bx,ay,by) 
    gamma = .001
    converged = False
    xa.append(x_now)
    while converged == False:
        if mag(x_now) <= prec:
            converged = True
        else:
            x_next = x_now - gamma*nda.Gradient(f(x_now))
            a = (x_next - x_now).T
            b = (nda.Gradient(f(x_next))- nda.Gradient(f(x_now)))
            gamma = a*b/mag(b)**2
            x_now = x_next
            xa.append(x_now)
         
    
    #gamma = .001
    #xa = []
    #ux = np.array([ax,bx])
    #vy = np.array([ay,by])
    #last_i = np.sqrt(ux.dot(vy))
    #xa.append(x_now)
    
    #while last_i > prec and i < N:
    #    x_last = x_now
    #    x_now = x_now - gamma*derivative2(f,x_last)
    #    a = x_now - x_last
        #last_i = np.sqrt(a.dot(a))
    #    last_i = f(x_now)
    #    b = derivative2(f,x_now)- derivative2(f,x_last)
    #    gamma = (x_now - x_last).T*b/np.sqrt(b.dot(b))
    #   xa.append(x_now)
    #    i += 1
        
    xa = np.array(xa)
    f_min = f(x_now)
    #print('best guess min: ',f_min)
    #print('min xy vals: ', xa)
    return xa, i

# Conjugate Gradient Descent

def CD_min1(f,xlims, ylims):
    # init x value statements, and the return value array
    ax,bx = xlims[0],xlims[1]
    ay,by = ylims[0],ylims[1]
    x_now = init(ax,bx,ay,by)
    xa = []
    xa.append(x_now[0])
    xa.append(x_now[1])
    # counters
    k = 0
    # inital values
    x0 = x_now
    #A1 = derivative2(f,x0)
    #A = derivative2(f,A1)
    A = np.gradient(f)
    #GRAD = -A*x0
    mag_A = np.sqrt(A[0]**2 + A[1]**2)
    # debug statements
    print('A: ', A)
    print('mag_A: ', mag_A)
    #print('grad:', GRAD)
    
    # internal functions
    def CD_algo_init(r,f,x):
        r =  - A*x
        #print('r_now', r)
        p = r
        #print('p_now', p)
        return r,p
        
    def CD_algo_r_loop(r0,p0,x0):
        aalpha = r0.T*r0
        balpha = p0.T*A*p0
        alpha = aalpha/balpha
        x1 = x0+alpha*p0
        r1 = r0 - alpha*A*p0
        return r1, r0, x1
        
    def CD_algo_x_loop(r1,r0,p0):
        Beta = (r1.T*r1)/(r0.T*r0)           
        p1 = r1 + Beta*p0
        return p1
    
    def inner_loop(rk,pk,xk):
        j = 0
        converged = False
        while converged == False or j < N: 
            rk1, rk, xk1 = CD_algo_r_loop(rk,pk,xk)

            if mag(rk) > prec: 
                converged = True
            if converged == True:                
                return rk1, rk, xk1
            # Debug statement
            #if j%100 == 0:
            #    print('looped r: ',rk1)
            #    print('mag_r: ', mag(rk1))
            #    print('r(k+1)', rk1)
            #    print('rk', rk)
            if j >= N:
                return rk1, rk, xk1
                break               
            else:
                rk = rk1
                # loop counter
                j += 1
        
    # initialize r and p
    r_now, p_now = CD_algo_init(GRAD,f,x_now)
    #print('init vals r_now, p_now: ', r_now, p_now)
    #print('f(x_now): ', f(x_now))
   
    # iterate through algo until a zero appears
    while np.abs(f(x_now)) > prec:
        x_prev = x_now
        r_now, r_prev, x_now = inner_loop(r_now,p_now,x_now)
        # find the next p
        p_next = CD_algo_x_loop(r_now,r_prev,p_now)
        p_now = p_next
        xa.append(x_now[0])
        xa.append(x_now[1])
        k += 1
        if k >= N:
            break
   

    #print(xa)
    xa = np.asarray(xa)
    xa.shape = (-1,2)
    #print('xa: ',xa)
    #print('k: ',k)
    return xa, k
    
    
# Quasi-Newton, 
# Boyden-Fletcher-Goldfarb-Shanno formulation
# 'BFGS'

def QN_min1(f, xlims, ylims):
    i = 0
    ax,bx = xlims[0],xlims[1]
    ay,by = ylims[0],ylims[1]
    x_now = init(ax,bx,ay,by) 
    
    
    Bk*pk = -derivative2(f,x_now)
    ak = np.argmin(f(x_now + a*pk))
    sk = ak*pk
    yk = derivative2(f, x_next) - derivative2(f, x_now)
    Bk1 = Bk + yk*(np.array([yk[0],yk[1]].T)/((np.array([yk[0],yk[1]].T)*sk) - 
          Bk*sk*(np.array([sk[0],sk[1]].T)*Bk/((np.array([sk[0],sk[1]].T)*Bk*sk)



    
# Generate 2 graphs,
# surface plot, and line plot of 
# benchmark function and
# minimization function results
# f         = benchmark function
# ID        = function's idnetifier for max/min
# EV        = known expected value of function  
# X         = results to graph
# niter     = number of iterations

def Graphit2D(f, ID, X, niter):
    # x,y limits based on ID of the function
    a, b = ID[0], ID[1]
    # x,y values returned from optimization function
    x, y = X[:,0], X[:,1]
    # meshgrid definitions
    nx = np.linspace(a,b,N)
    ny = np.linspace(a,b,N)
    x_mesh, y_mesh = np.meshgrid(nx,ny)
    f_mesh = f([x_mesh,y_mesh])
    # label, plot, and define contour lines of the figure
    #title = EV[3]
    fig = plt.figure()
    #plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    levels = np.arange(np.min(f_mesh), np.max(f_mesh), .3)
    Graph = plt.contour(x_mesh, y_mesh, f_mesh, levels)
    plt.clabel(Graph, inline=1, fontsize=10)    
    
    plt.plot(x,y, 'ro-')
    plt.show()
   # print('Number of iterations: ', niter)

def Graphit3D(f, ID, X, niter):
    a, b = -ID, ID
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
    ax.scatter(xc,yc,zc, 'bo-', s=200)
    cset = ax.contour(xg,yg,zg, zdir='zg', offset=0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f')
    plt.show()
   # print('Number of iterations: ', niter)

emptyset_meta = np.zeros(1)
emptyset_data = np.zeros(1)

zerosx = np.array([f1E[0],f2E[0],f3E[0],f4E[0],f5E[0],f6E[0]], dtype = float)
zerosy = np.array([f1E[1],f2E[1],f3E[1],f4E[1],f5E[1],f6E[1]], dtype = float)
zerosz = np.array([f1E[2],f2E[2],f3E[2],f4E[2],f5E[2],f6E[2]], dtype = float)
names = np.array([f1N,f2N,f3N,f4N,f5N,f6N], dtype = object)
ranges = np.array([f10,f20,f30,f40,f50,f60], dtype = float)

algo_use = np.array([scipymin,minimize_var,GD_min1, CD_min1], dtype = object)
algo_name = np.array(['scipymin','minimize_var','GD_min1', 'CD_min1'], dtype = object)


x_mins = -ranges
x_maxs = ranges
y_mins = -ranges
y_maxs = ranges

colname_meta = {'function':names, 'Expected_values_x':zerosx, 'Expected_values_y':zerosy,'Expected_values_z':zerosz, 
                'x_min': x_mins, 'x_max': x_maxs, 'y_min': y_mins, 'y_max': y_maxs}
               

colname_data = {'x0':emptyset_data, 'y0':emptyset_data}

colname_stats = {'optimization_algo':emptyset_meta, 'iterations':emptyset_meta}

df_meta = pd.DataFrame(colname_meta)
df_data = pd.DataFrame(colname_data)
df_stats = pd.DataFrame(colname_stats)

#print(df_meta)
#print(df_data)
#print(df_stats)
def build_DFs(indicator_algo, indicator_f,f):
    f_name = df_meta.loc[indicator_f].values[0:1]
    expected_values = df_meta.loc[indicator_f].values[1:4]
    x_lims = df_meta.loc[indicator_f].values[4:6]
    y_lims = df_meta.loc[indicator_f].values[6:8]
    X, I = algo_use[indicator_algo](f,x_lims, y_lims)

    dfd_data = {'x0':X[:,0], 'y0':X[:,1]}
    dfd = pd.DataFrame(dfd_data)

    F_index = np.array(['optimization_algo','function', 'Expected_values','x_lim', 'y_lim', 'iterations'], dtype = object)
    F_data = np.array([algo_name[indicator_algo],f_name,expected_values,x_lims,y_lims,I], dtype = object)
    dfs = pd.DataFrame(F_data, index = F_index)
    #print(X[:,0],X[:,1])
    return dfd,dfs


f1_mv_data, f1_minvar = build_DFs(1,0,f1)
f1_GD_data, f1_GD = build_DFs(2,0,f1)
f1_CD_data, f1_CD = build_DFs(3,0,f1)

f2_mv_data, f2_minvar = build_DFs(1,1,f2)
f2_GD_data, f2_GD = build_DFs(2,1,f2)
f2_CD_data, f2_CD = build_DFs(3,1,f2)

f3_mv_data, f3_minvar = build_DFs(1,2,f3)
f3_GD_data, f3_GD = build_DFs(2,2,f1)
f3_CD_data, f3_CD = build_DFs(3,2,f1)

print(f1_mv_data.head())
print(f1_minvar)
#Graphit2D(f1, f1_mv_data.loc[4], [f1_minvar.loc[1],f1_minvar.loc[2]], f1_mv_data.loc[6])
print(f1_GD_data.head())
print(f1_GD)
print(f1_CD_data.head())
print(f1_CD)

print(f2_mv_data.head())
print(f2_minvar)
print(f2_GD_data.head())
print(f2_GD)
print(f2_CD_data.head())
print(f2_CD)

print(f3_mv_data.head())
print(f3_minvar)
print(f3_GD_data.head())
print(f3_GD)
print(f3_CD_data.head())
print(f3_CD)








#print('name: ', f_name)
#print('EV: ',expected_values)
#print('x: ',x_lims)
#print('y: ',y_lims)
#print('return values: ',X)
#print('intervals: ',I)
    
    
    
    
    
    
    
# df_meta structure    
#         function  Expected_values_x  Expected_values_y  Expected_values_z  \
# 0          Given            0.00000            0.00000             0.0000   
# 1         Ackley            0.00000            0.00000             0.0000   
# 2          Beale            3.00000            0.50000             0.0000   
# 3  Cross-in-Tray            1.34941            1.34941            -2.0621   
# 4      Eggholder          512.00000          404.23180          -959.6407   
# 5   Holder Table            8.05502            9.66459           -19.2085   

#    x_min  x_max  y_min  y_max  
# 0   -2.0    2.0   -2.0    2.0  
# 1   -5.0    5.0   -5.0    5.0  
# 2   -4.5    4.5   -4.5    4.5  
# 3  -10.0   10.0  -10.0   10.0  
# 4 -512.0  512.0 -512.0  512.0  
# 5  -10.0   10.0  -10.0   10.0  

# df_data structure
#     x0   y0  x_min  y_min  f_min
# 0  0.0  0.0    0.0    0.0    0.0

# df_stats structure
#    optimization_algo     iterations  pct_dev_from_EV
# 0      1.273197e-313  1.273197e-313    1.273197e-313