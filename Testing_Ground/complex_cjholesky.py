import cmath

def CholeskyComplex(M):
    # M should be Hermitian and positive definite
    n = len(M)
    L = [[0+0j for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i+1):
            sum_val = sum(L[i][k] * L[j][k].conjugate() for k in range(j))
            if i == j:
                val = M[i][i] - sum_val
                L[i][j] = cmath.sqrt(val)
            else:
                val = M[i][j] - sum_val
                L[i][j] = val / L[j][j].conjugate()

    # Compute L† (conjugate transpose of L)
    L_conj_T = [[L[j][i].conjugate() for j in range(n)] for i in range(n)]

    return L, L_conj_T

M = [
    [4+0j, 1+2j],
    [1-2j, 5+0j]
]

L, L_conj_T = CholeskyComplex(M)
print(L)
print(L_conj_T)