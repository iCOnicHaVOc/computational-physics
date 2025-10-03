# Assignment8
# Aditya Raj, Roll no- 2311013
from My2lib import *
# list will contain poly coffn. with dec power
Eq1 = [1,-1,-7,1,6]
Eq2 = [1,0,-5,0,4]
Eq3 = [2.0,0.0,-19.5,0.5,13.5,-4.5]


guess = [-1.9,-0.2,1.1,3.1]  
all_roots, steps = lagurre(Eq1, guess, 20, 1e-6, 1e-6, synthetic_division, differentiation, fun_builder)
print("By Lagure's method - The roots of the function are", all_roots, "reached in", steps, "respective steps")

guess2 = [-1.9,-0.8,1.1,6]
all_roots2, steps2 = lagurre(Eq2, guess2, 20, 1e-6, 1e-6, synthetic_division, differentiation, fun_builder)
print("By Lagure's method - The roots of the function are", all_roots2, "reached in", steps2, "respective steps")

guess3 = [-2,0,2,5,6]
all_roots3, steps3 = lagurre(Eq3, guess3, 30, 1e-6, 1e-6, synthetic_division, differentiation, fun_builder)
print("By Lagure's method - The roots of the function are", all_roots3, "reached in", steps3, "respective steps")   
