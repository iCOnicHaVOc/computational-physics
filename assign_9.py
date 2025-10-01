# Assignment8
# Aditya Raj, Roll no- 2311013
from My2lib import *
# list will contain poly coffn. with dec power
Eq1 = [1,-1,-7,1,6]
Eq2 = [1,0,-5,0,4]
Eq3 = [2.0,0.0,-19.5,0.5,13.5,-4.5]
def fun1(x):
    return x**4 -x**3- 7*x**2 +x +6
def fun2(x):
    return x**4 -5*x**2 + 4
def fun3(x):
    return 2.0*x**5 - 19.5*x**3 + 0.5*x**2 + 13.5*x - 4.5

 
initial_guesses = [3,1,-2,-1]  
all_roots = find_all_roots(fun1, Eq1, initial_guesses)

initial_guesses = [-2,2,-1,1]  
all_roots = find_all_roots(fun2, Eq2, initial_guesses)

initial_guesses = [0,0,1.85,0,0]  
all_roots = find_all_roots(fun3, Eq3, initial_guesses)

njifebv
kjebvjef
j
#example change in assign9
print("Roots of Eq1 are - ", all_roots) 