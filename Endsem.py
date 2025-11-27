# Aditya raj 
# roll No: 2311013
# Date: 17/11/2025
# Endsem Exam

from My2lib import *
import numpy as np
import matplotlib.pyplot as plt
import math

#Q1
print()
print('Solution for Q1\n')


#Q2
print()
print('Solution for Q2\n')

A = [[4,-1,0,-1,0,0],[-1,4,-1,0,-1,0],[0,-1,4,0,0,-1],[-1,0,0,4,-1,0],[0,-1,0,-1,4,-1],[0,0,-1,0,-1,4]]
B = [2,1,2,2,1,2]

A1,B1 = GSeidel(A,B, 1e-6,50,None)
print("By Gauss-Seidel- Solution is",A1, 'and is achived in', B1, 'steps')

#Q3
print()
print('Solution for Q3\n')

def F(x):
    return [2.5 - x * math.exp(x)] 

def J(x):
    return [[-(1 + x) * math.exp(x)]]  

x0 = [1.1]  # initial guess
delta = 1e-6
eps = 1e-6
itter = 20

root, steps = newton_raphson_multi(F, J, x0, delta, eps, itter, inverse_matrix, multiply_matrices)

print("By Newton-Raphson: The spring stretches to", root[0], "meters in", steps, "steps.")

#Q4
print()
print('Solution for Q4\n')

L = 2  # in meters

# Formula: x_cm = intgr.(x * λ(x))dx / intgr.(λ(x))dx from 0 to L
def lamda(x):
    return x**2

# Numerator: integration(x * x^2)dx = integr.(x^3 dx) from 0 to L
def numerator(x):
    return x**3
o,p,q,r,t = monte_carlo_int(numerator,0,L,100,LCG,123456,1e-3)
print("The value of the numerator using monte carlo is", o)

# Denominator: intgr.(x^2 dx) from 0 to L
def denominator(x):
    return x**2
o1,p1,q1,r1,t1 = monte_carlo_int(denominator,0,L,100,LCG,123456,1e-3)
print("The value of the denominator using monte carlo is", o1)

# Centre of mass
x_cm = o / o1
print("Centre of mass:",x_cm, "meters")

#Q5
print()
print('Solution for Q5\n')

# Parameters
g = 10
gamma = 0.02
v0 = 10
y0 = 0
t0 = 0
h = 0.01

# Calculate theoretical upper bound (no air resistance)
def calc_upper_bound(v0, g):
    """
    H = v0^2 / (2*g)
    """
    return v0**2 / (2*g)

t_max = calc_upper_bound(v0, g)
print(f"Theoretical upper bound (no air resistance): {t_max:.2f} meters")

# dy/dt = v
def f1(v):
    return v

# dv/dt = -gamma*v - g
def f2(y, v):
    return -gamma * v - g

# Run RK4
t_vals, y_vals, v_vals = RK4_coupled(f1, f2, y0, t0, v0, h, t0, t_max)

# Estimate max height (when velocity crosses zero)
max_height = None
for i in range(1, len(v_vals)):
    if v_vals[i] < 0:
        max_height = y_vals[i-1]
        break

print(f"Maximum height reached: {max_height:.2f} meters")
print(y_vals,v_vals)
Plot(v_vals,y_vals,"Velocity vs Height with Air Resistance","Velocity (m/s)","Height (m)",'DATA\endsem_q5_plot.png')
print('The plot is saved in DATA folder with name - endsem_q5_plot')

#Q6
print()
print('Solution for Q6\n')

# Parameters
L = 2.0
nx = 20
nt = 5000
dx = L / (nx - 1)
dt = 0.0004  
r = dt / dx**2

# Grid setup
x = np.linspace(0, L, nx)
u = 20 * np.abs(np.sin(np.pi * x))  # initial condition
u_new = np.zeros_like(u)

# Time steps to capture
snapshot_steps = [0, 10, 20, 50, 100, 200, 500, 1000]
snapshots = {step: u.copy() if step == 0 else None for step in snapshot_steps}

# Time evolution
for n in range(1, nt + 1):
    for i in range(1, nx - 1):
        u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])
    u_new[0] = 0.0
    u_new[-1] = 0.0
    u = u_new.copy()
    if n in snapshot_steps:
        snapshots[n] = u.copy()

# Plotting (Copied from My2lib with change to plot multiple lists)
plt.figure(figsize=(10, 6))
for step in snapshot_steps:
    plt.plot(x, snapshots[step], label=f"t={step * dt:.3f}")
plt.xlabel("Position x")
plt.ylabel("Temperature u(x,t)")
plt.title("Temperature Evolution in 1D Heat Equation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('DATA\endsem_q6_plot.png')
plt.show()

print('The plot is saved in DATA folder with name - endsem_q6_plot')

#Q7
print()
print('Solution for Q7\n')

data = np.loadtxt('DATA\esem4fit.txt')
r = data[:,0]
h = data[:,1]
degree = 4 # Highest power of x

coeffs, sigma_a, chi2 = poly_fit_least_squares(r,h, degree)

print("Coefficients of fited polynomial is:", coeffs)

# Custom polynomial evaluation function
def eval_poly(coeffs, x):
    y = np.zeros_like(x, dtype=float)
    n = len(coeffs)
    for i in range(n):
        y += coeffs[i] * x**(n - i - 1)
    return y

# Generate smooth curve for plotting
r_fit = np.linspace(min(r), max(r), 500)
h_fit = eval_poly(coeffs, r_fit)

plot_comparison(r,h,r_fit,h_fit,"Polynomial Fit to Data",'r','h','DATA\endsem_q7_plot.png')
print('The plot is saved in DATA folder with name - endsem_q7_plot')
