# Aditya raj 
# roll No: 2311013
# Date: 15/10/2025
# Midsem Exam

from My2lib import *
import math


# Question 2
print("\nQUESTION 2\n")
def f(x):
    return (x - 5) * math.exp(x) + 5
def df(x):
    return math.exp(x) * (x - 4)

# Solve using Newton-raphson from library
root, step = newton_raphson(f, df, 5, 1e-4, 1e-6, 100)
print('By Newton Raphson- The root of the function is ',root,'reached in',step,'steps')

# Calculate Wien's constant b
h = 6.626e-34  # Planck's constant
c = 3e8        # Speed of light
k = 1.381e-23  # Boltzmann constant

b = (h * c) / (root * k)

print(f"Wien's constant b = {b:.4e} m-K")

# ======== MY Solutionn =========
# The value function at root is 2.978897128969038e-10
# By Newton Raphson- The root of the function is  4.96511423174643 reached in 3 steps
# Wien's constant b = 2.8990e-03 m-K


# QUESTION 4
print("\nQUESTION 4\n")
# function to read matrix and vector from file
def read_matrix(file_path):
    with open(file_path, 'r') as file:
        matrix = [list(map(float, line.strip().split())) for line in file]
    return matrix
def read_list(file_path):
    with open(file_path, 'r') as file:
        lst = [float(line.strip()) for line in file]
    return lst

A = read_matrix(r'DATA\matrix_A')
b = read_list(r'DATA\Vector_b')

x, ii = GSeidel(A, b, 1e-6, 50, None)
print("By Gauss-Seidel- Solution is",x, 'and is achived in', ii, 'step')

# ======== MY Solutionn =========
# By Gauss-Seidel- Solution is 
# [1.4999998297596437, -0.4999999999999992, 1.9999999999999998, -2.4999999148640373, 1.0000000000000004, -0.9999999999957907] and is achived in 13 step


# QUESTION 3
print("\nQUESTION 3\n")
# inv using LU decomposition

M = read_matrix(r'DATA\Matrix_q3')
n = len(M)
identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    
d = determinant(M)
print(f"Determinant of M is: {d}")
if d == 0:
    print("Matrix is singular, inverse does not exist.")    
else:
    print("Matrix is invertible.")
    print("\nFinding inverse using LU decomposition...")
    # we need to solve for each column of the identity matrix
    inverse_matrix_result = []
        
    for col in range(n):
        # Get the col-th column of identity matrix
        b_col = [identity[i][col] for i in range(n)]
        # Solve Ax = b_col
        L,U = LU_decom(M)
        y = forward_substitution(L, b_col)
        x = backward_substitution(U, y)
        inverse_matrix_result.append(x)

    # Transpose to get the final inverse matrix
    inverse_M = [[inverse_matrix_result[j][i] for j in range(n)] for i in range(n)]
    print_matrix(inverse_M, "M^(-1)")
        
    # Checking using my inv function
    m_inv = inverse_matrix(M)
    print("\nInverse matrix using my inv function for verification:")
    print_matrix(m_inv, "M^(-1)")

'''
# =========== MY Solutionn =========
Determinant of M is: 68.71680000000005
Matrix is invertible.

Finding inverse using LU decomposition...

M^(-1):
   -0.7079      2.5314      2.4312      0.9666     -3.9023
   -0.1934      0.3101      0.2795      0.0577     -0.2941
    0.0217      0.3655      0.2861      0.0506     -0.2899
    0.2734     -0.1299      0.1316     -0.1410      0.4489
    0.7815     -2.8751     -2.6789     -0.7011      4.2338

Inverse matrix using my inv function for verification:

M^(-1):
   -0.7079      2.5314      2.4312      0.9666     -3.9023
   -0.1934      0.3101      0.2795      0.0577     -0.2941
    0.0217      0.3655      0.2861      0.0506     -0.2899
    0.2734     -0.1299      0.1316     -0.1410      0.4489
    0.7815     -2.8751     -2.6789     -0.7011      4.2338
'''

# Question 1
print("\nQUESTION 1\n")
# Generate random points using LCG
P_x, lda = LCG(45, 1000)
P_y, lda = LCG(28, 1000)

# As my LCG generates numbers in [0,1], we need to scale them to the required ranges
P_x = [4 * x - 2 for x in P_x]  # scale to [-2,2]
P_y = [2 * y - 1 for y in P_y]  # scale to [-1,1]

# Estimate area of ellipse with semi-minor axis = 1, semi-major axis = 2
a = 2
b = 1
def check_ellipse(x_list, y_list):
    Area_n = []
    total_points = len(x_list)
    inside = 0
    for point_num, (x, y) in enumerate(zip(x_list, y_list), start=1):
        # Ellipse equation: (x/2)^2 + (y/1)^2 <= 1
        if (x / 2) ** 2 + (y / 1) ** 2 <= 1:
            inside += 1
        area_est = 4 * 2 * 1 * (inside / point_num)  # Rectangle area = 4 (x in [-2,2], y in [-1,1])
        Area_n.append(area_est)
    return Area_n

area_estimates = check_ellipse(P_x, P_y)
print(len(area_estimates), 'estimates generated.')
p_avg = area_estimates[len(area_estimates)//2:]
print(len(p_avg), 'estimates considered for averaging.')
avg_area = sum(p_avg) / len(p_avg)
print('Estimated area of the ellipse is:', avg_area)
ar = math.pi * a*b
print('Actual area of the ellipse is:', ar)
print('Percentage error is:', abs((avg_area - ar) / ar) * 100, '%')

# ======== MY Solutionn =========
# 1000 estimates generated.
# 500 estimates considered for averaging.
# Estimated area of the ellipse is: 6.192699606130741
# Actual area of the ellipse is: 6.283185307179586
# Percentage error is: 1.4401246601059166 %

