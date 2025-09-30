import math
from My2lib import *
# Assignment 7
# Name - Aditya Raj, Roll no- 2311013

def fun1(x):
    return 3*x + math.sin(x) - math.exp(x)
def D_fun1(x):
    return 3 + math.cos(x) - math.exp(x)

def fun2(x):
    return x**2 -2*x-3
def D_fun2(x):
    return 2*x - 2
def G_fun2(x):
    return 3/x + 2
def G1_fun2(x):
    return 3/(x-2)

#Question1 
print('for question 1\n')

d,g = newton_raphson(fun1,D_fun1,1.1,1e-6,1e-6,20)
print('By Newton Raphson- The root of the function is ',d,'reached in',g,'steps')

w,ee = root_bisection(fun1,0,1,1e-6,1e-6,0.1)
print("By bisection- Root (c) = ", w, 'Reached in',ee, 'Steps')

x1,y1 = root_falsi(fun1,0,1,1e-6,1e-6,0.1)
print("By Root Falsi - Root (c) = ", x1, 'Reached in',y1, 'Steps')
# Question 2

print('for question2\n')
# I found out that for a particular g(x) the def gets stuck on a particular root 
# no matter what your initial x_0 is. So i have given two g(x) that are giving two roots of the function.
d1,g1 = fix_point(fun2,G_fun2,-10,1e-6,1e-6,200)
print('By Fix point- The root of the function with g(x) = 3/x + 2 is ',d1,'reached in',g1,'steps')

d2,g2 = fix_point(fun2,G1_fun2,-10,1e-6,1e-6,200)
print('By Fix point- The root of the function with g(x) = 3/(x-2)  is ',d2,'reached in',g2,'steps')

'''  My solution - 
for question 1

The value function at root is -4.30371871473767e-09
By Newton Raphson- The root of the function is  0.36042170124008527 reached in 5 steps
Checking Value of f(c) is -1.1357753471052945e-07
By bisection- Root (c) =  0.36042165756225586 Reached in 20 Steps
Checking Value of f(c) is 2.749072325336499e-07
By Root Falsi - Root (c) =  0.3604218128434815 Reached in 6 Steps
for question2

The value function at root is 5.368843929431932e-07
By Fix point- The root of the function with g(x) = 3/x + 2 is  3.000000134221094 reached in 16 steps
The value function at root is -7.719697401320502e-07
By Fix point- The root of the function with g(x) = 3/(x-2)  is  -0.9999998070075556 reached in 15 steps'''
