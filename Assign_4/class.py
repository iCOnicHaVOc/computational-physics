import math
from My2lib import *
#Assign 4 
# name = Aditya Raj 
# Roll no - 2311013

A = [[4,1,1,1],[1,3,-1,1],[1,-1,2,0],[1,1,0,2]]
b = [3,3,1,3]

# Question 1 - Cholesky factorisation


k,kd = CholeskyD(A)


y = forward_substitution(k, b)

x = backward_substitution(kd, y)

print("FOR Q1- Solution x:", x)
'''
c1 = multiply_matrices(k,kd)
print(c1)
'''
# Question 2 - Jacobi

O,s = GJacobi(A,b, 1e-6, 200)

print("FOR Q2- Solution is",O, 'and is achived in', s , 'steps')


'''My solution:-
FOR Q1- Solution x: [-1.1102230246251565e-16, 0.9999999999999999, 1.0, 1.0000000000000004]
FOR Q2- Solution is [0.0, 0.9999994039535522, 0.9999997019767761, 0.9999997019767761] and is achived in 41 steps'''
