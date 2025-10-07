from My2lib import *
#Assign 4 (THURSDAY)
# name = Aditya Raj 
# Roll no - 2311013

A1 = [[4,-1,0,-1,0,0],[-1,4,-1,0,-1,0],[0,-1,4,0,0,-1],[-1,0,0,4,-1,0],[0,-1,0,-1,4,-1],[0,0,-1,0,-1,4]]
B1 = [2,1,2,2,1,2]

A2 = [[4,0,4,10,1],[0,4,2,0,1],[2,5,1,3,13],[11,3,0,1,2],[3,2,7,1,0]]
B2 = [20,15,92,51,15]


#Question 1 
print('Solution for Q1\n')
print(Check_sym_matrix(A1))

jj,ll = GSeidel(A1,B1, 1e-6,50,None)
print("By Gauss-Seidel- Solution is",jj, 'and is achived in', ll, 'steps')

kk,pp = CholeskyD(A1)
y = forward_substitution(kk,B1)
x = backward_substitution(pp, y)
print("By Cholesky- Solution is:", x)

er,tr = GJacobi(A1,B1,1e-6,50,None)
print("By Jacobi- Solution is",er, 'and is achived in',tr, 'steps')


# Question 2

# Manual alteration to make A2 diagonal dominiant
# exchanging 3rd row with 5th & 1st row with 4th
A2_m = [[11,3,0,1,2],[0,4,2,0,1],[3,2,7,1,0],[4,0,4,10,1],[2,5,1,3,13]]
B2_m = [51,15,15,20,92]


print('Solution for Q2\n')

jj1,ll1 = GSeidel(A2_m,B2_m, 1e-6,50,None)
print("By Gauss-Seidel- Solution is",jj1, 'and is achived in', ll1, 'steps')

er1,tr1 = GJacobi(A2_m,B2_m,1e-6,60,None)
print("By Jacobi- Solution is",er1, 'and is achived in',tr1, 'steps')



'''
MY SOLUTIONS 
Solution for Q1

Matrix is symmetric

By Gauss-Seidel- Solution is [0.9999997530614102, 0.9999997892247294, 0.9999999100460266, 0.9999998509593769, 0.9999998727858708, 0.9999999457079743] and is achived in 16 steps
By Cholesky- Solution is: [1.0, 0.9999999999999999, 1.0, 1.0, 1.0, 1.0]
By Jacobi- Solution is [0.9999989753998146, 0.9999985509965219, 0.9999989753998146, 0.9999989753998146, 0.9999985509965219, 0.9999989753998146] and is achived in 26 steps

Solution for Q2

By Gauss-Seidel- Solution is [2.979165086347139, 2.215599676186742, 0.21128402698819157, 0.15231700827754802, 5.715033568811629] and is achived in 12 steps       
By Jacobi- Solution is [2.9791649583226008, 2.215599258220273, 0.21128373337161171, 0.15231661140963978, 5.71503326456748] and is achived in 57 steps'''