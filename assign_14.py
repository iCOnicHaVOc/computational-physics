# Assignment14
# Aditya Raj, Roll no- 2311013

import math
import numpy as np
from My2lib import *

# LHS of the equations of the form f(y(x),x), keeping RHS as dy/dx

def fun1(x,y):
    return (x + y)**2
def ana_fun1(x):
    return math.tan(x + (math.pi/4)) - x

# for damped harmonic oscillator x'' + mu x' + w**2 x = 0
k = 1.0
m = 1.0
mu = 0.15
w = math.sqrt(k/m)

def fun2_dxdt(v):
    return v
def fun2_dvdt(x, v):
    return -(mu/m) * v - ((k)/m) * x

# Analytical sol for Question 1
x1_ana = []
y1_ana = [] 
for i in np.arange(0, (math.pi/5 + 0.1), 0.1):
    x1_ana.append(i)
    y1_ana.append(ana_fun1(i))
    
# RK4 Method for Question 1 with h = 0.1 , 0.25, 0.45
h = [0.1, 0.25, 0.45]
for step in h:
    x1_rk4, y1_rk4 = RK4_non_coupled(fun1, 0, 1, step, 0, math.pi/5)
    plot_comparison(x1_ana, y1_ana, x1_rk4, y1_rk4,  f'Analytical vs RK4 Solution for h={step}','x', 'y', f'DATA\Question1_RK4_h{step}.png')

# RK4 Method for Question 2
yu,gu,nu = RK4_coupled(fun2_dxdt, fun2_dvdt, 1.0, 0.0, 0.0, 0.1, 0.0, 40.0)
Plot(yu, gu, 'Damped Harmonic Oscillator using RK4 Method','Time (s)', 'Displacement (m)',  'DATA\Damped_HO_RK4_Dis.png')
Plot(yu, nu, 'Damped Harmonic Oscillator Velocity using RK4 Method','Time (s)', 'Velocity (m/s)',  'DATA\Damped_HO_RK4_Velocity.png')

# for Total Energy plot
KE = 0.5 * m * np.multiply(nu, nu) 
PE = 0.5 * k * np.multiply(gu, gu)  
TE = KE + PE                       
Plot(yu, TE, 'Phase Space Plot of Damped Harmonic Oscillator', 'Time (s)', 'Total Energy (J)',  'DATA\Damped_HO_RK4_Total_Energy.png')

# for phase space plot
p = np.multiply(m, nu)  # momentum p = m*v
Plot(gu, p, 'Phase Space Plot of Damped Harmonic Oscillator', 'Displacement (m)', 'Momentum (kg m/s)',  'DATA\Damped_HO_RK4_Phase_Space.png')

# ======== MY Solutionn ========= IS SAVED IN DATA FOLDER AS PNG FILES ========