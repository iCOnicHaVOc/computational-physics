# Assignment16
# Aditya Raj, Roll no- 2311013
import math
from My2lib import *

# for Q1 
x = [2,3,5,8,12]
y = [10,15,25,40,60]

x_interp = 6.7
y_interp = lagrange_interpolation(x, y, x_interp)
print(f"Lagrange Interpolation at x={x_interp}: y={y_interp}")

# for Q2
x = [2.5,3.5,5,6,7.5,10,12.5,15,17.5,20.5]
y = [13,11,8.5,8.2,7,6.2,5.2,4.8,4.6,4.3]

sigma_i = [1.0 for _ in x]  

def logf1x(a,b,x): # power law
    return math.log(a) + b * math.log(x)
def logf2x(a,b,x): # exponential
    return math.log(a)+ b*x

a_pow, b_pow, sigma_a_pow, sigma_b_pow, r2_pow = linear_regression_log(x, y, sigma_i, 'pow')
a_expo, b_expo, sigma_a_expo, sigma_b_expo, r2_expo = linear_regression_log(x, y, sigma_i, 'expo')
print(f"Power Law Model: a={math.exp(a_pow):.4f}±{math.exp(a_pow)*sigma_a_pow:.4f}, b={b_pow:.4f}±{sigma_b_pow:.4f}, r^2={r2_pow:.4f}")
print(f"Exponential Model: a={math.exp(a_expo):.4f}±{math.exp(a_expo)*sigma_a_expo:.4f}, b={b_expo:.4f}±{sigma_b_expo:.4f}, r^2={r2_expo:.4f}")


# ======== MY Solutionn =========
# Lagrange Interpolation at x=6.7: y=33.49999999999999
# Power Law Model: a=21.0464±21.9157, b=-0.5374±0.4723, r^2=0.7750
# Exponential Model: a=12.2130±7.6380, b=-0.0585±0.0540, r^2=0.5762
