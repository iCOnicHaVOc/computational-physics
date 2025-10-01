import math
from My2lib import *
# Assignment - 6
#Name - Aditya raj , Roll no. - 2311013


def fun1(x):
    return math.log(x/2) - math.sin(5*x/2)
def fun2(x):
    return -x - math.cos(x)


#Question_!

print("For question 1\n")
x,y = root_bisection(fun1,3,4,1e-6,1e-6,1e-6)
print("By bisection- Root (c) = ", x, 'Reached in',y, 'Steps')
print()
x1,y1 = root_falsi(fun1,1,2,1e-6,1e-6,1e-6)
print("By Root Falsi - Root (c) = ", x1, 'Reached in',y1, 'Steps')

# Question_2 
print()
print("For Question 2\n")
q,w,e = bracket(fun2,2,4,0.05)
print('Roots are in interval [',q,',',w,'] and reached in',e, "steps")
print('To verify the interval, we check the roots\n')
# to check
tt, rr = root_falsi(fun2,-1,7,1e-6,1e-6,1e-6)
print("By Root Falsi - Root (c) = ", tt, 'Reached in',rr, 'Steps')

'''
My solutions
For question 1

Checking Value of f(c) is 3.381375837663292e-07
By bisection- Root (c) =  2.555964946746826 Reached in 19 Steps

Checking Value of f(c) is -1.1173088024230005e-08
By Root Falsi - Root (c) =  2.5559650975808643 Reached in 4 Steps

For Question 2

Roots are in interval [ -1.1874849202000002 , 4 ] and reached in 10 steps
To verify the interval, we check the roots

Checking Value of f(c) is -1.0964843455418816e-07
By Root Falsi - Root (c) =  -0.7390850676991185 Reached in 5 Steps
'''
