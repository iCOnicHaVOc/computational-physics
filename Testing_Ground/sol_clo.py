# P346 Mid-Semester Examination 2021 - Solutions using My2lib
# Import your library
from My2lib import *
import math

print("="*70)
print("P346 Mid-Semester Examination 2021 - Solutions")
print("="*70)

# ============================================================================
# Question 1: Wien's Displacement Law using Newton-Raphson
# ============================================================================
print("\n" + "="*70)
print("Q1: Wien's Displacement Law - Newton-Raphson Method")
print("="*70)

# Constants
h = 6.626e-34  # Planck's constant (m^2 kg/s)
k = 1.381e-23  # Boltzmann constant (m^2 kg/Ks^2)
c = 3e8        # Speed of light (m/s)

# Define the function: f(x) = (x-5)e^x + 5 = 0
def f_wien(x):
    return (x - 5) * math.exp(x) + 5

# Derivative: f'(x) = e^x + (x-5)e^x = x*e^x - 4*e^x
def df_wien(x):
    return x * math.exp(x) - 4 * math.exp(x)

# Initial guess
x0 = 5.0
delta = 1e-10
eps = 1e-4
max_iter = 100

# Solve using Newton-Raphson from My2lib
x_solution, steps = newton_raphson(f_wien, df_wien, x0, delta, eps, max_iter)

print(f"\nSolution for x: {x_solution:.6f}")
print(f"Converged in {steps} iterations")

# Calculate Wien's constant b
b = (h * c) / (x_solution * k)
print(f"\nWien's constant b = {b:.6e} m·K")
print(f"Wien's constant b = {b:.4f} m·K (to precision 10^-4)")

# ============================================================================
# Question 2: Matrix Inverse using Gauss-Jordan Elimination
# ============================================================================
print("\n" + "="*70)
print("Q2: Matrix Inverse - Gauss-Jordan Elimination")
print("="*70)

# Define the matrix
A = [
    [0, 0, 0, 2],
    [0, 0, 3, 0],
    [0, 4, 0, 0],
    [5, 0, 0, 0]
]

print("\nOriginal Matrix A:")
print_matrix(A, "A")

# To find inverse, solve A*X = I for each column of I
n = len(A)
A_inv = []

for i in range(n):
    # Create the i-th column of identity matrix
    b = [0] * n
    b[i] = 1
    
    # Solve using Gauss-Jordan
    aug, x = GaussJordan(A, b)
    A_inv.append(x)

# Transpose to get the inverse matrix (columns become rows)
A_inverse = [[A_inv[j][i] for j in range(n)] for i in range(n)]

print("\nInverse Matrix A^-1:")
print_matrix(A_inverse, "A^-1")

# Verify: A * A^-1 should give identity
verification = multiply_matrices(A, A_inverse)
print("\nVerification (A * A^-1):")
print_matrix(verification, "A * A^-1")

# ============================================================================
# Question 3: Linear Equations using LU Decomposition
# ============================================================================
print("\n" + "="*70)
print("Q3: Solving Linear Equations - LU Decomposition")
print("="*70)

# Coefficient matrix
M = [
    [3, -7, -2, 2],
    [-3, 5, 1, 0],
    [6, -4, 0, -5],
    [-9, 5, -5, 12]
]

# Constants vector
b = [-9, 5, 7, 11]

print("\nCoefficient Matrix M:")
print_matrix(M, "M")

print("\nConstants vector b:", b)

# Perform LU decomposition
L, U = LU_decom(M)

print("\nLower triangular matrix L:")
print_matrix(L, "L")

print("\nUpper triangular matrix U:")
print_matrix(U, "U")

# Solve Ly = b using forward substitution
y = forward_substitution(L, b)
print("\nIntermediate solution y (from Ly = b):", y)

# Solve Ux = y using backward substitution
x = backward_substitution(U, y)

print("\nFinal Solution x:")
for i, val in enumerate(x, 1):
    print(f"  x{i} = {val:.6f}")

# Verify the solution
print("\nVerification:")
for i in range(len(M)):
    result = sum(M[i][j] * x[j] for j in range(len(x)))
    print(f"  Equation {i+1}: {result:.6f} (expected {b[i]})")

# ============================================================================
# Question 4: Root Finding - Bisection vs Regula-Falsi
# ============================================================================
print("\n" + "="*70)
print("Q4: Root Finding Comparison - Bisection vs Regula-Falsi")
print("="*70)

# Define the function: f(x) = 4e^(-x)sin(x) - 1 = 0
def f_root(x):
    return 4 * math.exp(-x) * math.sin(x) - 1

# Parameters
a, b = 0, 1
delta = 1e-10
eps = 1e-4
beta = 1.5

print(f"\nFunction: f(x) = 4e^(-x)sin(x) - 1")
print(f"Interval: [{a}, {b}]")
print(f"Precision: {eps}")

# Method 1: Bisection (Midpoint method)
print("\n" + "-"*70)
print("Method 1: Bisection (Midpoint Method)")
print("-"*70)
root_bisect, steps_bisect = root_bisection(f_root, a, b, delta, eps, beta)
print(f"Root: {root_bisect:.6f}")
print(f"Iterations: {steps_bisect}")
print(f"f(root) = {f_root(root_bisect):.10f}")

# Method 2: Regula-Falsi
print("\n" + "-"*70)
print("Method 2: Regula-Falsi (False Position Method)")
print("-"*70)
root_falsi, steps_falsi = root_falsi(f_root, a, b, delta, eps, beta)
print(f"Root: {root_falsi:.6f}")
print(f"Iterations: {steps_falsi}")
print(f"f(root) = {f_root(root_falsi):.10f}")

# Comparison
print("\n" + "-"*70)
print("Convergence Comparison")
print("-"*70)
print(f"Bisection:    {steps_bisect} iterations")
print(f"Regula-Falsi: {steps_falsi} iterations")
print(f"\nRegula-Falsi is {'faster' if steps_falsi < steps_bisect else 'slower'} "
      f"by {abs(steps_bisect - steps_falsi)} iterations")

print("\n" + "="*70)
print("All solutions completed!")
print("="*70)