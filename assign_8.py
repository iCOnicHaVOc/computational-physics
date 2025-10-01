# Assignment8
# Aditya Raj, Roll no- 2311013
from My2lib import *
import math

def fun1(x1,x2,x3):
    f1 = x1**2 + x2 -37
    f2 = x1 - x2**2 - 5
    f3 = x1+x2+x3 - 3
    return [ f1,f2,f3]
def G(x1, x2, x3):
    new_x1 = math.sqrt(37 - x2)
    new_x2 = math.sqrt(x1 - 5)
    new_x3 = 3 - x1 - x2
    return [new_x1, new_x2, new_x3]

# Q1 , PART 1-
x0 = [17, 10, 10]
print('Question-1, part(a)\n')
d1,g1 = fix_point_multi(fun1,G,x0,1e-6,1e-6,20)
print('By Fix point- The root of the function is ',d1,'reached in',g1,'steps\n')

# Q2, PART 2-
print('Question-1, part(b)\n')

def J(x1, x2, x3):
    return [
        [2*x1,   1,   0],  
        [1,   -2*x2,  0],   
        [1,      1,   1]    
    ]

x0b = [17, 10, 10]
sol, steps = newton_raphson_multi(fun1, J, x0b,1e-6,1e-6,50,inverse_matrix,multiply_matrices)

print("By Newton-Raphson - The root of the system is", sol, "reached in", steps, "steps")
'''
# ----------------- MY SOLUTION --------------
Question-1, part(a)

The value of function at root is [4.0135324042012144e-07, -1.1020231038827433e-07, 2.9115092647913343e-07]
By Fix point- The root of the function is  [6.000000027700167, 1.000000068951236, -3.999999805500476] reached in 12 steps

Question-1, part(b)

The value of function at root is [4.476419235288631e-13, -6.59703403016465e-11, 8.881784197001252e-16]
By Newton-Raphson - The root of the system is [6.0, 1.0, -4.0] reached in 8 steps
'''