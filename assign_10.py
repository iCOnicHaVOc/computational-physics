# Assignment10
# Aditya Raj, Roll no- 2311013
import math
from My2lib import *

#given function
def f1(x):
    return 1/x
def f2(x):
    return x*math.cos(x)
def f3(x):
    return x*math.atan(x)

N = [4,8,15,20]
I = [f1,f2,f3]
int = [[1,2], [0,math.pi/2], [0,1]]

for i in I:
    print("For function", i)
    a = int[I.index(i)][0]
    b = int[I.index(i)][1]
    print("In the interval [", a, ",", b, "]")
    for n in N:
        p = trapZ_integration(i, a, b, n)
        print("The value of the integral using trapezoid with N=", n, "is", p)
        q = midpoint_integration(i, a, b, n)
        print("The value of the integral using midpoint with N=", n, " is", q)


# ======================MY SOLUTION=========================
"""For function <function f1 at 0x000001FF1A58FE20>
In the interval [ 1 , 2 ]
The value of the integral using trapezoid with N= 4 is 0.5630952380952381
The value of the integral using midpoint with N= 4  is 0.6912198912198912
The value of the integral using trapezoid with N= 8 is 0.629538517038517
The value of the integral using midpoint with N= 8  is 0.6926605540432034
The value of the integral using trapezoid with N= 15 is 0.659516758381053
The value of the integral using midpoint with N= 15  is 0.6930084263712957
The value of the integral using trapezoid with N= 20 is 0.6679828689721814
The value of the integral using midpoint with N= 20  is 0.6930690982255869
For function <function f2 at 0x000001FF1A6089A0>
In the interval [ 0 , 1.5707963267948966 ]
The value of the integral using trapezoid with N= 4 is 0.44908523487295693
The value of the integral using midpoint with N= 4  is 0.5874479167573121
The value of the integral using trapezoid with N= 8 is 0.5362028149251044
The value of the integral using midpoint with N= 8  is 0.5749342733821311
The value of the integral using trapezoid with N= 15 is 0.5604222549632509
The value of the integral using midpoint with N= 15  is 0.5719716590967575
The value of the integral using trapezoid with N= 20 is 0.5648768242652932
The value of the integral using midpoint with N= 20  is 0.5714572867152204
For function <function f3 at 0x000001FF1A609800>
In the interval [ 0 , 1 ]
The value of the integral using trapezoid with N= 4 is 0.13359534651990018
The value of the integral using midpoint with N= 4  is 0.2820460493571144
The value of the integral using trapezoid with N= 8 is 0.19867379680888492
The value of the integral using midpoint with N= 8  is 0.2845610193056679
The value of the integral using trapezoid with N= 15 is 0.23633208794067295
The value of the integral using midpoint with N= 15  is 0.28516010270349235
The value of the integral using trapezoid with N= 20 is 0.24798664384725724
The value of the integral using midpoint with N= 20  is 0.28526426016144524
"""
    