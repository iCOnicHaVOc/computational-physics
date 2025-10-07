import math

class MyComplex:
    def __init__(self, real, imag=0.0):
        # Store the real and imaginary parts
        self.r = real
        self.i = imag

    def display_cmplx(self):
        # Print in the form a,bj without spaces
        print(self.r, ",", self.i, "j", sep="")

    def add_cmplx(self, c1, c2):
        # Add corresponding real and imaginary parts
        self.r = c1.r + c2.r
        self.i = c1.i + c2.i
        return MyComplex(self)

    def sub_cmplx(self, c1, c2):
        # Subtract corresponding real and imaginary parts
        self.r = c1.r - c2.r
        self.i = c1.i - c2.i
        return MyComplex(self)

    def mul_cmplx(self, c1, c2):
        # Multiply using (a+bi)(c+di) = (ac - bd) + (ad + bc)i
        self.r = c1.r * c2.r - c1.i * c2.i
        self.i = c1.i * c2.r + c1.r * c2.i
        return MyComplex(self)

    def mod_cmplx(self):
        # Modulus: sqrt(real^2 + imag^2) without NumPy
        return (self.r ** 2 + self.i ** 2) ** 0.5


# Function for matrix multiplication
def multiply_matrices(A, B):
    rowsA, colsA = len(A), len(A[0])
    rowsB, colsB = len(B), len(B[0])

    if colsA != rowsB:
        raise ValueError("Matrix dimensions do not match for multiplication")

    # Create result matrix filled with 0
    result = [[0 for _ in range(colsB)] for _ in range(rowsA)]

    # Multiply manually
    for i in range(rowsA):
        for j in range(colsB):
            for k in range(colsA):
                result[i][j] += A[i][k] * B[k][j]
    return result #A matrix (list of lists)

# class randGEN():
#     def __init__(self):
#         self.rand= []
#         self.index=[]
#     def LCG(self,p,q,r,k):
#         for i in range (0,500): 
#             self.index.append(i)
#             k = (p*k+q)%r 
#             f = k/r
#             self.rand.append(f)
#         print(self.rand)
#         print(self.index)
#         return randGEN(self)
    
# THIS GIVE A SINGLE RANDOM NO. AS OUTPUT
# def LCG(p,q,r,k,n):
#     for i in range (0,n): 
#         k = (p*k+q)%r 
#         f = k/r
#     return f

# ploting the lists 
import matplotlib.pyplot as plt 
#define x and y , as a python list 
def Plot(x, y , title='Pi plot 2000 throws', xlabel='index', ylabel='Values of Pi',file_name='sample_plot.png'):
    plt.figure(figsize=(11, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)  
    plt.savefig(file_name)
    plt.show()

#GIVES A LIST WITH RANDOM NUMBERS BETWEEN 0 AND 1 and THEIR INDEX
def LCG(k,it):
    # k = seed
    # it = no. of random numbers to be generated
    p=1103515245 # a (in sir slide )
    q=12345 # c (in sir slide )
    r=32768 # m (in sir slide )
    Rand = []
    index = []
    for i in range (0,it): 
        index.append(i)
        k = (p*k+q)%r 
        f = k/r
        Rand.append(f)
    return Rand , index

# to store a list in a txt file
def store_list(file_path, lst):
    with open(file_path, 'w') as file:
        for item in lst:
            file.write(f"{str(item)}\n")

# to read a list from a txt file
def read_list(file_path):
    with open(file_path, 'r') as file:
        lst = [float(line.strip()) for line in file]
    return lst


# to store a 2D list (matrix) in a txt file
def store_matrix(file_path, matrix):
    with open(file_path, 'w') as file:
        for row in matrix:
            file.write(' '.join(map(str, row)) + '\n')

# to read a 2D list (matrix) from a txt file
def read_matrix(file_path):
    with open(file_path, 'r') as file:
        matrix = [list(map(float, line.strip().split())) for line in file]
    return matrix

def print_matrix(matrix, name="Matrix"):
    print(f"\n{name}:")
    for row in matrix:
        print("  ".join(f"{val:10.4f}" for val in row))   # 4 decimal places

# Gauss jordan elemination 
def GaussJordan(X,Y):
    # X is the coeff matrix
    # Y is the const matrix
    # Xx = Y
    m= [row[:] for row in X]
    var = Y[:]
    row = len(m)
    colu = len(m[0])
    aug=[]

    for i in range (row):
        l = X[i][:] # l = [row i of X]
        l.append(Y[i]) # l = [row i of X with Y[i] appended]
        aug.append(l) 

    # now for a given row i will loop over all columns 
    ro = 0
    for col in range(colu):
        max_val = abs(X[ro][col])
        sel = ro # selected row index with max pivot element in a column
        # this loop finds the max pivot element in the current column
        for r in range (ro,row):
            if abs(X[r][col]) > max_val:
                max_val = abs(X[r][col])
                sel = r

        # this swaps the choosen pivot row with current row
        aug[ro], aug[sel] = aug[sel], aug[ro] # swap

        pivot_val = aug[ro][col]
        aug[ro] = [v / pivot_val for v in aug[ro]] # making pivot 1
        for r in range(row): # making all element below pivot 0
            if r != ro:
                factor = aug[r][col]
                aug[r] = [aug[r][c] - factor * aug[ro][c] for c in range(colu+1)]
        ro +=1 #looping over all rows
        if ro == row :
            break
    # making a different solution matrix
    x = [aug[i][colu] for i in range(row)]
    return aug, x 

        
# Doolittle approch
def LU_decom(M):
    n = len(M)
    c = len(M[0])

    L = [[0.0]*n for _ in range(n)]
    U = [[0.0]*n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0
        
    for j in range(c):
        for i in range(n):
            if i <=j:
                sigma_u = 0
                for k in range(0,i):
                    sigma_u += L[i][k]*U[k][j]
                U[i][j] = M[i][j] - sigma_u
            else:
                sigma_l = 0
                for ko in range(0,j):
                    sigma_l += L[i][ko]*U[ko][j]
                L[i][j] = (M[i][j] - sigma_l)/U[j][j]
    
    return L,U

# from eq Ax = b, we make L(LT *X) = b
# L * y = b
# LT * x = y
def forward_substitution(L, b):
    n = len(b)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    return y

def backward_substitution(LT, y):
    n = len(y)
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(LT[i][j] * x[j] for j in range(i + 1, n))) / LT[i][i]
    return x

def determinant(A):
    n = len(A)
    L, U = LU_decom(A)
    det = 1.0
    for i in range(n):
        det *= U[i][i]
    return det

def inverse_matrix(A):
    n = len(A)
    L, U = LU_decom(A)
    A_inv = []

    # Solve AX = I column by column
    for i in range(n):
        e = [0]*n
        e[i] = 1
        y = forward_substitution(L, e)
        x = backward_substitution(U, y)
        A_inv.append(x)
    return [list(col) for col in zip(*A_inv)]

# Gjacobi method, Martix should be diagonal dominiant, 
def GJacobi (M, x, precision, itter, guess):
    # for equation M*a=x
    row_M = len(M)
    row_b = len(x)
    col_M = len(M[0])
    steps = 0

    if guess is None:
        input_val = [0.0 for _ in range(row_M)]
    else:
        input_val = guess[:]

    prev_va=[input_val[:]]

    for i in range(itter):
        input_new= [0.0 for _ in range(row_M)]
        for i in range(row_M):
            sigma = 0
            for j in range(col_M):
                if j != i:
                    sigma += M[i][j]*input_val[j]
            input_new[i] = (x[i] - sigma)/M[i][i]
        prev_va.append(input_new[:])

        diff = max(abs(input_new[i]-input_val[i]) for r in range(row_M))
        if diff < precision:
            return input_new  , steps


        input_val = input_new
        steps +=1

    return input_val , steps

def CholeskyD(M):
    # M should be symmetric and positive definite
    # returns L and LT (U is LT here)
    row_M = len(M)
    col_M = len(M[0])
    M_l = [[0.0 for _ in range(col_M)] for _ in range(row_M) ]

    for i in range(row_M):
        for j in range(i+1):
            sum = 0
            for k in range(j):
                sum += M_l[i][k]*M_l[j][k]
            if i==j:
                M_l[i][j] = math.sqrt(M[i][i]- sum)
            else:
                M_l[i][j] = (M[i][j] - sum)/M_l[j][j]
    
    M_lt = [[M_l[j][i] for j in range(col_M)] for i in range(row_M)]
    
    return M_l , M_lt


def GSeidel(M,x,precision,itter,guess):
    row_M = len(M)
    row_b = len(x)
    col_M = len(M[0])
    steps = 0

    if guess is None:
        input_val = [0.0 for _ in range(row_M)]
    else:
        input_val = guess[:]

    for l in range(itter):
        prev_va=input_val[:]
        for i in range(row_M):
            sigma1 = 0
            sigma2 = 0
            for j in range(col_M):
                if j < i:
                    sigma1 += M[i][j]*input_val[j]
                if j > i :
                    sigma2 += M[i][j]*prev_va[j]

            input_val[i] = (x[i] - sigma1 - sigma2)/M[i][i]
            
        diff = max(abs(input_val[r]-prev_va[r]) for r in range(row_M))
        if diff < precision:
            return input_val  , l+1

    return input_val , itter


def Check_sym_matrix(M):
    n = len(M)
    m = len(M[0])
    for i in range(n):
        for j in range(m):
            if M[i][j] != M[j][i]:
                print('Matrix is not symmetric')
                return
    print('Matrix is symmetric')


def root_bisection(fx,a,b,delta,eps,beta):
    # fx = function whose root is to be found
    # a,b = initial guesses, should bracket the root
    # delta = function value tolerance (closenwss of f(c) to 0)
    # eps = interval size tolerance
    # beta = bracketing expansion factor

    if fx(a)*fx(b) >= 0:
        print("The root is not bracketed, let me do it for ya")
        fa,fb=fx(a),fx(b)
        s= 0
        while fa*fb >0 :
            if abs(fa) < abs(fb):
                a = a - beta*(b-a)
                fa = fx(a)
            if abs(fa) > abs(fb):
                b = b + beta*(b-a)
                fb = fx(b)
            s+=1
        a0,b0 = a,b
        print('Roots lie in the interval','[',a0,',',b0,'] and is reached in',s,'steps')
        return root_falsi(fx,a0,b0,delta,eps,beta)

    steps = 0
    while True:
        c= (a+b)/2.0
        fc = fx(c)

        if abs(fc) < delta or abs(b-a) < eps:
            print('Checking Value of f(c) is',fc)
            return c,steps
        if fc*fx(a) < 0:
            b = c
        else:      #vulnarable if fc == 0,fix it
            a = c
        steps+=1

# Root Falsi with bracketing function in it
def root_falsi(fx,a,b,delta,eps,beta):
    # fx = function whose root is to be found
    # a,b = initial guesses, should bracket the root
    # delta = function value tolerance (closenwss of f(c) to 0)
    # eps = interval size tolerance
    # beta = bracketing expansion factor

    if fx(a)*fx(b) >= 0:
        print("The root is not bracketed, let me do it for ya")
        fa,fb=fx(a),fx(b)
        s= 0
        while fa*fb >0 :
            if abs(fa) < abs(fb):
                a = a - beta*(b-a)
                fa = fx(a)
            if abs(fa) > abs(fb):
                b = b + beta*(b-a)
                fb = fx(b)
            s+=1
        a0,b0 = a,b
        print('Roots lie in the interval','[',a0,',',b0,'] and is reached in',s,'steps')
        return root_falsi(fx,a0,b0,delta,eps,beta)

    steps = 0
    while True:
        c= b - (((b-a)*fx(b))/(fx(b)-fx(a)))
        fc = fx(c)

        if abs(fc) < delta or abs(b-a) < eps:
            print('Checking Value of f(c) is',fc)
            return c,steps
        if fc*fx(a) < 0:
            b = c
        else:
            a = c
        steps+=1

#bracketing function,(integrated in bisection and falsi)
def bracket(fx,a,b,beta):
    # fx = function whose root is to be found
    # a,b = initial guesses, should bracket the root
    # beta = bracketing expansion factor

    fa,fb=fx(a),fx(b)
    steps = 0

    while fa*fb >0 :
        if abs(fa) < abs(fb):
            a = a - beta*(b-a)
            fa = fx(a)
        if abs(fa) > abs(fb):
            b = b + beta*(b-a)
            fb = fx(b)
        steps+=1
    return a,b,steps

def fix_point(f,gf,x0,delta,eps,itter):
    # f = function whose root is to be found
    # gf = g(x) function for fixed point iteration
    # x0 = initial guess
    # delta = function value tolerance (closenwss of f(c) to 0
    # eps = interval size tolerance
    # itter = max itterations (use to stop infinite loop)
    
    x1 = x0
    step = 0
    
    while step < itter:
        x2 = gf(x1)
        step+=1

        if abs(x2-x1) < eps  or abs(f(x2)) < delta:
            print('The value function at root is', f(x2))
            return x2,step
        
        x1 = x2

    print('donot converge within the given iterations')
    return None, step

def newton_raphson(f,df,x0,delta,eps,itter):
    # f = function whose root is to be found
    # df = derivative of f
    # x0 = initial guess
    # delta = function value tolerance (closenwss of f(c) to 0)
    # eps = interval size tolerance
    # itter = max itterations (use to stop infinite loop)
    
    x1 = x0
    step = 0
    
    while step<itter:
        x2 = x1 - (f(x1)/df(x1))
        step+=1

        if abs(x2-x1) < eps  or abs(f(x2)) < delta:
            print('The value function at root is', f(x2))
            return x2,step
        
        x1 = x2

    print('donot converge within the given iterations')
    return None, step


def fix_point_multi(f, gf, x0, delta, eps, itter):
    # f = function whose root is to be found
    # gf = g(x) function for fixed point iteration
    # x0 = initial guess
    # delta = function value tolerance (closenwss of f(c) to 0
    # eps = interval size tolerance
    # itter = max itterations (use to stop infinite loop)
    
    x1 = x0[:]  #copy the list of initial guess
    step = 0
    
    while step < itter:
        x2 = gf(*x1)  # unpack list into arguments, x2 is a list
        step += 1

        # compute max difference
        mod = (sum(x**2 for x in x2))**0.5
        diff = (sum((x2[i] - x1[i])**2 for i in range(len(x1))))**0.5 / mod
        f_val = f(*x2)
        f_norm = max(abs(val) for val in f_val)

        if diff < eps or f_norm < delta:
            print("The value of function at root is", f_val)
            return x2, step
        
        x1 = x2

    print("Did not converge within the given iterations")
    return None, step

def newton_raphson_multi(F, J, x0, delta, eps, itter, inv_func, matmul):
    # F = function whose root is to be found, returns a list
    # J = Jacobian of F, returns a matrix 
    # x0 = initial guess, a list
    # delta = function value tolerance (closenwss of f(c) to 0)
    # eps = interval size tolerance
    # itter = max itterations (use to stop infinite loop)
    # inv_func = function to compute inverse of a matrix
    # matmul = function to compute matrix multiplication

    x1 = x0[:]  # copy the list of initial guess
    step = 0

    while step < itter:
        f_val = F(*x1)                      # vector
        J_val = J(*x1)                      # matrix
        J_inv = inv_func(J_val)

        # reshape f_val into column vector to fit into my matrix multiplication code
        f_col = [[val] for val in f_val]

        # compute update = J_inv * f_val
        update_col = matmul(J_inv, f_col)

        # turn back to list for easier substraction
        update = [row[0] for row in update_col]

        x2 = [x1[i] - update[i] for i in range(len(x1))]
        step += 1

        #compute difference
        mod = (sum(x**2 for x in x2))**0.5
        diff = (sum((x2[i] - x1[i])**2 for i in range(len(x1))))**0.5 / mod
        f_val2 = F(*x2)
        f_norm = max(abs(val) for val in f_val2)

        if diff < eps or f_norm < delta:
            print("The value of function at root is", f_val)
            return x2, step

        x1 = x2

    print("Did not converge within the given iterations")
    return None, step

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

def lagurre(fun,x0,itter,eps,delta,syn_div,diff,fun_build):
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

