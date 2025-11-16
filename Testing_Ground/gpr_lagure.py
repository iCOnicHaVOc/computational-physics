import cmath


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


def lagure_try(fun, x0, itter, eps, delta, syn_div, diff, fun_build):
    root = []
    steps = []

    for guess in x0:
        r = guess
        for i in range(itter):
            n = len(fun) - 1  # ✅ degree of polynomial
            dfun = diff(fun)
            ddfun = diff(dfun)

            G = fun_build(dfun)(r) / fun_build(fun)(r)
            H = G ** 2 - fun_build(ddfun)(r) / fun_build(fun)(r)

            denom1 = G + cmath.sqrt((n - 1) * (n * H - G ** 2))
            denom2 = G - cmath.sqrt((n - 1) * (n * H - G ** 2))
            rem = denom1 if abs(denom1) > abs(denom2) else denom2

            a = n / rem
            r_next = r - a

            if abs(r_next - r) < eps or abs(fun_build(fun)(r_next)) < delta:
                root.append(r_next)
                steps.append(i + 1)
                print("Root found:", r_next, "f(root) =", fun_build(fun)(r_next))
                break

            r = r_next
        else:
            print("Did not converge for guess", guess)
            root.append(r)
            steps.append(itter)

        fun, remainder = syn_div(fun, r)
        if abs(remainder) > 1e-4:
            print("Non-zero remainder:", remainder)

    return root, steps

Eq1 = [1, -1, -7, 1, 6]
guess = [4, 2, -3, -2]

roots, steps = lagure_try(Eq1, guess, 30, 1e-6, 1e-6, synthetic_division, differentiation, fun_builder)
print("Roots:", roots)
