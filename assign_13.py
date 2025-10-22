# Assignment12
# Aditya Raj, Roll no- 2311013

import math
import numpy as np
from My2lib import *

# LHS of the equations of the form f(y(x),x), keeping RHS as dy/dx
def fun1(x,y):
    return y - x**2
def ana_fun1(x):
    return x**2 + 2*x + 2 - 2*math.exp(x)

def fun2(x,y):
    return (x + y)**2
def ana_fun2(x):
    return math.tan(x + (math.pi/4)) - x

# calculation for anaytical value
# for q1
x1_ana = []
y1_ana = [] 
for i in np.arange(0, 2.1, 0.1):
    x1_ana.append(i)
    y1_ana.append(ana_fun1(i))
# for q2
x2_ana = []
y2_ana = [] 
for i in np.arange(0, (math.pi/5), 0.1):
    x2_ana.append(i)
    y2_ana.append(ana_fun2(i))

# Question 1
f1, g1 = predictor_corrector(fun1, 0, 0, 0.1, 0, 2)
plot_comparison(x1_ana, y1_ana, f1, g1, 'Predictor-Corrector Method vs Analytical Solution for Question 1', 'x', 'y', 'DATA\pridic_for_EQ1.png' )  

# Question 2
f , g = predictor_corrector(fun2, 0, 1, 0.1, 0, math.pi/5)
plot_comparison(x2_ana, y2_ana, f, g, 'Predictor-Corrector Method vs Analytical Solution for Question 2', 'x', 'y', 'DATA\pridic_for_EQ2.png' )