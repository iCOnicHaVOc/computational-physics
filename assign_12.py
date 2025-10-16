# Assignment12
# Aditya Raj, Roll no- 2311013
import math
import numpy as np
from My2lib import *

def fun1(x):
    return (x**2 / (1+ x**4)) # interval is [-1,1]

def fun2(x):
    return math.sqrt(1+x**4) # interval is [0,1]


int , st = gaussian_quad(fun1, -1, 1, 1e-9, 0.487495494)
print("The value of the integral of question 1 using gaussian quadrature is", int)

int2 , st2 = gaussian_quad(fun2, 0, 1, 1e-9, 1.089429413)
print("The value of the integral of question 2 using gaussian quadrature is", int2)
tar_value = 1.089429413
n_sim = 10

while True:
    ints = int_simp(fun2, 0, 1, n_sim)
    if abs(ints - tar_value) < 1e-9:
        break
    n_sim += 2 # must be even
print("The value of the integral of question 2 using Simpson's rule is", ints, "and is achieved in", n_sim, "steps")

# ======== MY Solutionn =========
# Desired tolerance achieved with n = 14
# The value of the integral of question 1 using gaussian quadrature is 0.48749549425855665
# Desired tolerance achieved with n = 8
# The value of the integral of question 2 using gaussian quadrature is 1.0894294131091897
# The value of the integral of question 2 using Simpson's rule is 1.0894294123885135 and is achieved in 18 steps