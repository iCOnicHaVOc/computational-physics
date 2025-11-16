import math


# takes list of cofficients, can also take a value to be inserted and returns a function
# or a result with the inserted value
def fun_builder(coff):
    # coff is list of cofficients of polynomial in decreasing order of power
    def f(x):
        res = 0
        n = len(coff)
        for i in range(n):
            res += coff[i] * (x ** (n - 1 - i))
        return res
    return f

Eq1 = [1,-1,-7,1,6]


# differentiation fun, 
# when polynomial is given in list of cofficients of form [a_n, a_(n-1),..., a_1, a_0]
def differentiation(coeff):
    n = len(coeff)
    new_coeff = [coeff[i] * (n-1-i) for i in range(n-1)]
    return new_coeff


def synthetic_division(coeffs, r):
    n = len(coeffs)
    qu = [0] * (n - 1) #list of n-1 zeros for quotient
    qu[0] = coeffs[0]

    for i in range(1, n-1): #loop leaving 1st coff (preset above) till 2nd last (len -2, which is n-1 in 0 index)
        qu[i] = coeffs[i] + qu[i-1] * r
    re = coeffs[-1] + qu[-1] * r

    return qu, re

def lagure_try(fun,x0,itter,eps,delta,syn_div,diff,fun_build):
    # fun = function in the form of list of cofficients
    # x0 = list of initial guess
    # itter = max itterations (use to stop infinite loop)
    # eps = interval size tolerance

    root = []
    steps = []

    for guess in x0:
        r1 = guess
        for i in range(itter):
            n = len(fun)-1 # degree of polynomial
            dfun = diff(fun)
            ddfun = diff(dfun)
            fval = fun_build(fun)(r1)

            if fval == 0:
                root.append(r1)
                steps.append(i+1)
                print("The value of function at root is", fun_build(fun)(r1))
                break
            else:
                G = fun_build(dfun)(r1)/fval
                H = G**2 - fun_build(ddfun)(r1)/fval

            discriminant = (n-1)*(n*H-G**2)
            tol = 1e-18
            if discriminant < -tol:
                print("Complex root encountered (discriminant < 0). Skipping this guess.")
                break
            elif discriminant < 0:
                sqrt_term = 0.0
            else:
                sqrt_term = math.sqrt(discriminant)

            rem1 = G + sqrt_term
            rem2 = G - sqrt_term
            rem = max(abs(rem1), abs(rem2))

            if abs(rem) < tol:
                print("Denominator too small, possible division by zero. Skipping this guess.")
                break
            a = n/rem
            r2 = r1 - a

            if abs(r2 - r1) < eps or abs(fun_build(fun)(r2)) < delta:
                root.append(r2)
                steps.append(i+1)
                print("The value of function at root is", fun_build(fun)(r2))
                break
            r1 = r2

        else:
            print(" root Did not converge within the given iterations")
            root.append(r1)
            steps.append(itter)

        qx,re = syn_div(fun,r1)
        if abs(re) > 1e-4:
            print('Non-zero reminder',re)
    return root, steps
'''
guess = [-1.9,-0.2,1.1,3.1]  
all_roots, steps = lagure_try(Eq1, guess, 20, 1e-6, 1e-6, synthetic_division, differentiation, fun_builder)
print("By Lagure's method - The roots of the function are", all_roots, "reached in", steps, "respective steps")
'''
Eq2 = [1,0,-5,0,4]
Eq3 = [2.0,0.0,-19.5,0.5,13.5,-4.5]
'''
guess2 = [-1.9,-0.8,1.1,6]
all_roots2, steps2 = lagure_try(Eq2, guess2, 20, 1e-6, 1e-6, synthetic_division, differentiation, fun_builder)
print("By Lagure's method - The roots of the function are", all_roots2, "reached in", steps2, "respective steps")
'''
guess3 = [-2,0,2,5,6]
all_roots3, steps3 = lagure_try(Eq3, guess3, 30, 1e-6, 1e-6, synthetic_division, differentiation, fun_builder)
print("By Lagure's method - The roots of the function are", all_roots3, "reached in", steps3, "respective steps")   

# PROBLEM
# WHEN TWO ROOTS ARE COINCIDENT, OUTPUT IS REPEATING, BUT IN NORMAL CASE WITH CLOSE GUESS OUTPUT IS ALSO SAME. SO ITS 
# DIFFICULT TO DISTINGUISH BETWEEN THE TWO CASES
# LIKE IN EQ-3 WITH GUESS 0.5,0.6,2,5,6 OUTPUT IS [0.5,0.5,3.0,-1.0,-3.0],
# BUT WITH GUESS -2,0,2,5,6 OUTPUT IS -2.999,-1.000,0.500,3.0,3.0 WHICH IS A PROBLEM
'''
#Roots form desmos
root of Eq1 = [3.0, 1.0, -1.0, -2.0]
root of Eq2 = [2.0, 1.0, -1.0, -2.0]
root of Eq3 = [3.0, 0.5, 0.5 , -1.0 ,-3.0] 
'''
