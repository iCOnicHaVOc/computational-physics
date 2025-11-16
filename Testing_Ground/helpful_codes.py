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
'''
# testing the above functions
l = [0.1, 0.2, 0.3, 0.4, 0.5,6.7,7.8,8.9]
file_path = 'numSTORE.txt'  
store_list(file_path, l)
l_read = read_list(file_path)
print(l_read)  # should print the original list 
r = read_list(r'DATA\asgn0_vecC')
print(r)
'''
#PROBLEMS - always read list with r"relative\path\file.txt" or 'absolute\path\file.txt'


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

# testing the above functions
'''matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
file_path_matrix = 'F:\F_folder\DDD\VS_FILES\P346\Testing_Ground\matrixSTORE.txt'
store_matrix(file_path_matrix, matrix)'''
matrix_read = read_matrix(r'DATA\asgn0_matB')
print(matrix_read)  # should print the original matrix  '''

# efficient LU decomposition using Doolittle's method
def LU_decom_inplace(A):
    """
    In-place LU decomposition using Doolittle's method.
    The matrix A is overwritten with L and U:
    - U in the upper triangle including diagonal
    - L in the lower triangle (excluding diagonal, since L[i][i] = 1 implicitly)
    """
    n = len(A)

    for k in range(n):
        # Compute U[k][j] for j = k to n-1
        for j in range(k, n):
            sigma_u = sum(A[k][m] * A[m][j] for m in range(k))
            A[k][j] = A[k][j] - sigma_u

        # Compute L[i][k] for i = k+1 to n-1
        for i in range(k + 1, n):
            sigma_l = sum(A[i][m] * A[m][k] for m in range(k))
            A[i][k] = (A[i][k] - sigma_l) / A[k][k]

    return A  # Contains both L and U

def extract_LU(A):
    n = len(A)
    L = [[1.0 if i == j else (A[i][j] if i > j else 0.0) for j in range(n)] for i in range(n)]
    U = [[A[i][j] if i <= j else 0.0 for j in range(n)] for i in range(n)]
    return L, U

A = [[1,2,4], [3,8,14], [2,6,13]]
c = extract_LU(LU_decom_inplace(A))
print(c)

# for round of digits
result = [3.1415926535, 0.000987654321, 123456.789, 2.7182818284, 0.0000123456, 99999.9999, 1.0000000001]
rounded = [round(x, 4) for x in result]
print(rounded)