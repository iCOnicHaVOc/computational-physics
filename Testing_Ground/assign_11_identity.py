from My2lib import *
from assign_11 import f2, f2, f3

int1 = int_simp(f2, 1, 2, 20)
print("The value of function 1 using simpson's rule with N= 28 is", int1)
mid = midpoint_integration(f2, 1, 2, 10)
print("The value of function 1 using midpoint rule with N= 10 is", mid)
trap = trapZ_integration(f2, 1, 2, 10)
print("The value of function 1 using trapezoidal rule with N= 10 is", trap)

a = (2*mid + trap)/3
print("The value of function 1 using the identity is", a)