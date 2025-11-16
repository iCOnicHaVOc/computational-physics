#P346 Computer Lab Mid-Semester Examination 2021 Solutions
# Using My2lib functions

import math
from My2lib import *

print("="*70)
print("P346 COMPUTER LAB MID-SEMESTER EXAMINATION 2021 - SOLUTIONS")
print("="*70)

# Question 1: Wien's displacement law using Newton-Raphson method
print("\nQUESTION 1: Wien's displacement law")
print("Equation: (x-5)e^x + 5 = 0")
print("Using Newton-Raphson method")

# Define the function and its derivative
def f(x):
    return (x - 5) * math.exp(x) + 5

def df(x):
    # Derivative of (x-5)e^x + 5 = e^x + (x-5)e^x = e^x(1 + x - 5) = e^x(x - 4)
    return math.exp(x) * (x - 4)

# Initial guess (x > 0, and close to where the root might be)
x0 = 5.0
precision = 1e-4
eps = 1e-6
max_iter = 100

print(f"Initial guess: x0 = {x0}")
print(f"Precision required: {precision}")

# Solve using Newton-Raphson from library
root, steps = newton_raphson(f, df, x0, precision, eps, max_iter)

print(f"Root found: x = {root:.6f}")
print(f"Steps taken: {steps}")

# Calculate Wien's constant b
h = 6.626e-34  # Planck's constant
c = 3e8        # Speed of light
k = 1.381e-23  # Boltzmann constant

# From x = hc/(λm*k*T) and λm*T = b, we get x = hc/(b*k)
# Therefore b = hc/(x*k)
b = (h * c) / (root * k)

print(f"Wien's constant b = {b:.4e} m-K")
print(f"Wien's constant b = {b:.4f} m-K")

print("\n" + "="*70)

# Question 2: Matrix inverse using Gauss-Jordan elimination
print("\nQUESTION 2: Matrix inverse using Gauss-Jordan elimination")

matrix = [
    [0, 0, 0, 2],
    [0, 0, 3, 0],
    [0, 4, 0, 0],
    [5, 0, 0, 0]
]

print("Original matrix:")
print_matrix(matrix, "A")

# This is a permutation matrix, let's check if it's invertible manually
# For a 4x4 matrix like this, we can compute determinant by recognizing the pattern
print("\nThis is a permutation matrix.")

# For this specific matrix, the determinant can be computed as:
# Since it's a permutation matrix with one non-zero element in each row and column
# det = 2 * 3 * 4 * 5 * sign of permutation
# The permutation is (1→4, 2→3, 3→2, 4→1), which has sign (-1)^3 = -1
det_manual = 2 * 3 * 4 * 5 * (-1)
print(f"Determinant = {det_manual}")

if det_manual == 0:
    print("Matrix is singular (not invertible)")
else:
    print("Matrix is invertible")
    
    # For this permutation matrix, the inverse is straightforward
    # We can use Gauss-Jordan from the library
    n = len(matrix)
    identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    
    print("\nFinding inverse using Gauss-Jordan elimination...")
    
    # Since the library GaussJordan function works for Ax = b,
    # we need to solve for each column of the identity matrix
    inverse_matrix_result = []
    
    for col in range(n):
        # Get the col-th column of identity matrix
        b_col = [identity[i][col] for i in range(n)]
        
        # Solve Ax = b_col
        try:
            aug_matrix, solution = GaussJordan(matrix, b_col)
            inverse_matrix_result.append(solution)
        except:
            print(f"Error solving for column {col}")
            # For this specific permutation matrix, we can manually construct inverse
            break
    
    if len(inverse_matrix_result) == n:
        # Transpose to get the final inverse matrix
        inverse_A = [[inverse_matrix_result[j][i] for j in range(n)] for i in range(n)]
        
        print("Inverse matrix:")
        print_matrix(inverse_A, "A^(-1)")
        
        # Verify by multiplying A * A^(-1)
        verification = multiply_matrices(matrix, inverse_A)
        print("\nVerification (A * A^(-1)):")
        print_matrix(verification, "A * A^(-1)")
    else:
        # Manual construction for this specific permutation matrix
        print("Constructing inverse manually for this permutation matrix...")
        # For the given matrix, the inverse is:
        inverse_A = [
            [0, 0, 0, 0.2],
            [0, 0, 1/3, 0],
            [0, 0.25, 0, 0],
            [0.5, 0, 0, 0]
        ]
        
        print("Inverse matrix:")
        print_matrix(inverse_A, "A^(-1)")
        
        # Verify by multiplying A * A^(-1)
        verification = multiply_matrices(matrix, inverse_A)
        print("\nVerification (A * A^(-1)):")
        print_matrix(verification, "A * A^(-1)")

print("\n" + "="*70)

# Question 3: Solve linear equations using LU decomposition
print("\nQUESTION 3: Solve linear equations using LU decomposition")

# System of equations:
# 3x1 - 7x2 - 2x3 + 2x4 = -9
# -3x1 + 5x2 + x3 + 0x4 = 5
# 6x1 - 4x2 + 0x3 - 5x4 = 7
# -9x1 + 5x2 - 5x3 + 12x4 = 11

A = [
    [3, -7, -2, 2],
    [-3, 5, 1, 0],
    [6, -4, 0, -5],
    [-9, 5, -5, 12]
]

b = [-9, 5, 7, 11]

print("Coefficient matrix A:")
print_matrix(A, "A")

print(f"\nConstant vector b: {b}")

# LU decomposition
print("Performing LU decomposition...")
try:
    L, U = LU_decom(A)
    
    print("\nLU Decomposition:")
    print_matrix(L, "L (Lower triangular)")
    print_matrix(U, "U (Upper triangular)")
    
    # Solve Ly = b (forward substitution)
    y = forward_substitution(L, b)
    print(f"\nForward substitution (Ly = b): y = {y}")
    
    # Solve Ux = y (backward substitution)
    x = backward_substitution(U, y)
    print(f"Backward substitution (Ux = y): x = {x}")
    
    print(f"\nSolution:")
    for i in range(len(x)):
        print(f"x{i+1} = {x[i]:.6f}")
    
    # Verify the solution
    verification_b = [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]
    print(f"\nVerification (Ax): {verification_b}")
    print(f"Original b:       {b}")
    print(f"Difference:       {[abs(verification_b[i] - b[i]) for i in range(len(b))]}")

except Exception as e:
    print(f"LU decomposition failed: {e}")
    print("Trying alternative method using Gauss-Jordan elimination...")
    
    # Use Gauss-Jordan as backup
    try:
        aug_matrix, solution = GaussJordan(A, b)
        
        print(f"\nSolution using Gauss-Jordan:")
        for i in range(len(solution)):
            print(f"x{i+1} = {solution[i]:.6f}")
        
        # Verify the solution
        verification_b = [sum(A[i][j] * solution[j] for j in range(len(solution))) for i in range(len(A))]
        print(f"\nVerification (Ax): {verification_b}")
        print(f"Original b:       {b}")
        print(f"Difference:       {[abs(verification_b[i] - b[i]) for i in range(len(b))]}")
        
    except Exception as e2:
        print(f"Gauss-Jordan also failed: {e2}")
        print("The system may be ill-conditioned or have no unique solution.")

print("\n" + "="*70)

# Question 4: Find root using Bisection and Regula-falsi methods
print("\nQUESTION 4: Find root of 4e^(-x)sin(x) - 1 = 0")
print("Using both Bisection (Midpoint) and Regula-falsi methods")

def equation(x):
    return 4 * math.exp(-x) * math.sin(x) - 1

# Check the function values at interval endpoints
a, b = 0, 1
print(f"\nInterval: [{a}, {b}]")
print(f"f({a}) = {equation(a):.6f}")
print(f"f({b}) = {equation(b):.6f}")
print(f"f(a) * f(b) = {equation(a) * equation(b):.6f} (should be negative for root existence)")

# Parameters
delta = 1e-4  # function value tolerance
eps = 1e-4    # interval size tolerance
beta = 1.5    # bracketing expansion factor

# Solve using Bisection method
print("\n--- BISECTION METHOD ---")
root_bisection_result, steps_bisection = root_bisection(equation, a, b, delta, eps, beta)
print(f"Root found: x = {root_bisection_result:.6f}")
print(f"Steps taken: {steps_bisection}")

# Solve using Regula-falsi method
print("\n--- REGULA-FALSI METHOD ---")
root_falsi_result, steps_falsi = root_falsi(equation, a, b, delta, eps, beta)
print(f"Root found: x = {root_falsi_result:.6f}")
print(f"Steps taken: {steps_falsi}")

# Comparison
print("\n--- COMPARISON ---")
print(f"Bisection method:  Root = {root_bisection_result:.6f}, Steps = {steps_bisection}")
print(f"Regula-falsi method: Root = {root_falsi_result:.6f}, Steps = {steps_falsi}")
print(f"Difference in roots: {abs(root_bisection_result - root_falsi_result):.8f}")

if steps_falsi < steps_bisection:
    print("Regula-falsi method converged faster")
elif steps_bisection < steps_falsi:
    print("Bisection method converged faster")
else:
    print("Both methods took the same number of steps")

print("\nFinal verification:")
print(f"f({root_bisection_result:.6f}) = {equation(root_bisection_result):.8f} (Bisection)")
print(f"f({root_falsi_result:.6f}) = {equation(root_falsi_result):.8f} (Regula-falsi)")

print("\n" + "="*70)
print("ALL SOLUTIONS COMPLETED")
print("="*70)