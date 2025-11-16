import numpy as np

# Get nodes and weights for 3-point Gauss-Legendre quadrature
x, w = np.polynomial.legendre.leggauss(4)

print("Nodes:", x)
print("Weights:", w)