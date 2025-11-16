# Assignment11
# Aditya Raj, Roll no- 2311013
import math
from My2lib import *

#given function
def f1(x):
    return 1/x
def f2(x):
    return x*math.cos(x)
def f3(x):
    return math.sin(x)**2
 

int1 = int_simp(f1, 1, 2, 24)
print("The value of function 1 using simpson's rule with N= 28 is", int1)
mid = midpoint_integration(f1, 1, 2, 10)
print("The value of function 1 using midpoint rule with N= 10 is", mid)

int = int_simp(f2, 0, math.pi/2, 28)
print("The value of function 2 using simpson's rule with N= 28 is", int)
mid2 = midpoint_integration(f2, 0, math.pi/2, 10)
print("The value of function 2 using midpoint rule with N= 10 is", mid2)

o,p,q,r,t = monte_carlo_int(f3,-1,1,100,LCG,123456,1e-3)
print("The value of the integral using monte carlo accurate upto 4 decimal place is", o, "with standard deviation", p)

# Plot convergence
Plot(q,r,
     title='Convergence of Monte Carlo Integration for sin^2(x)',
     xlabel='Number of random points (N)',
     ylabel='Estimated Integral (F_N)',
     file_name='DATA\monte_carlo_convergence.png')
# plot of convergence for standard deviation
Plot(q,t,
     title='Convergance of Standard Deviation of Monte Carlo Integration for sin^2(x)',
     xlabel='Number of random points (N)',
     ylabel='Standard Deviation (σ_N)',
     file_name='DATA\monte_carlo_stddev_convergence.png')

# ======================MY SOLUTION=========================
'''
The value of function 1 using simpson's rule with N= 28 is 0.6931472743443177
The value of function 1 using midpoint rule with N= 10 is 0.6928353604099603
The value of function 2 using simpson's rule with N= 28 is 0.5707965784470774
The value of function 2 using midpoint rule with N= 10 is 0.573442705787416
Increasing number of points to 200
Increasing number of points to 400
Increasing number of points to 800
Increasing number of points to 1600
Increasing number of points to 3200
Increasing number of points to 6400
Increasing number of points to 12800
Increasing number of points to 25600
Increasing number of points to 51200
Desired tolerance achieved.
The value of the integral using monte carlo accurate upto 4 decimal place is 0.5453881646654606 with standard deviation 0.22307093890810584

I HAVE SAVED THE PLOT IN THE FOLDER DATA AS monte_carlo_convergence.png
I HAVE SAVED THE PLOT IN THE FOLDER DATA AS monte_carlo_stddev_convergence.png
'''