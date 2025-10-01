# name = Aditya Raj , Roll - 2311013
# code for gauss jordan elemination
from My2lib import GaussJordan
# Question 1 
A = [[0,2,5],[3,-1,2],[1,-1,3]]
b = [1,-2,3]

z,s = GaussJordan(A,b)
print("Aug matrix -",z, '\nsolution -', s)

# Question 2 
C = [[1,-1,4,0,2,9],[0,5,-2,7,8,4],[1,0,5,7,3,-2],[6,-1,2,3,0,8],[-4,2,0,5,-5,3],[0,7,-1,5,4,-2]]
D = [19,2,13,-7,-9,2]

jo, jh = GaussJordan(C,D)
print("Aug matrix -",jo, '\nsolution -', jh)