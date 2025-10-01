# name - Aditya Raj , roll no - 2311013

# Question 1 (from Doolittle)
from My2lib import *
A = [[1,2,4], [3,8,14], [2,6,13]]

lo,uo = LU_decom(A)
res = multiply_matrices(lo,uo)

print(res)

'''My solution
L=[[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [2.0, 1.0, 1.0]] 
U =[[1, 2, 4], [0.0, 2.0, 2.0], [0.0, 0.0, 3.0]]
A =[[1.0, 2.0, 4.0], [3.0, 8.0, 14.0], [2.0, 6.0, 13.0]]'''


# Question 2
C = [[1,-1,4,0,2,9],[0,5,-2,7,8,4],[1,0,5,7,3,-2],[6,-1,2,3,0,8],[-4,2,0,5,-5,3],[0,7,-1,5,4,-2]]
D = [19,2,13,-7,-9,2]

C_l , C_u = LU_decom(C)

'''aug_1, y = GaussJordan(C_l,D)
aug_u, x = GaussJordan(C_u, y)

print(x)'''
clr = len(C_l)
y = [0.0]*clr
y[0]=D[0]
for i in range (1,clr):
    sum = 0
    for j in range(i):
        sum += C_l[i][j]*y[j]
    y[i] = D[i] - sum

x = [0.0]*clr
x[-1] = y[-1]/C_u[-1][-1]
for i in range(clr):
    sum2=0
    for j in range(i,clr):
        sum2 += C_u[-i][-j]*x[-j]
    x[-i]= (y[-i]- sum2)/ C_u[-i][-i]

print(x)

''' MY X 
[17.63350761559658, 0.4, -4.57142857142857, -2.821428571428571, 1.6989051094890513, -0.0]
'''


