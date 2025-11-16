from My2lib import RK4_coupled
from assign_15 import dT_dx, dz_dx

import matplotlib.pyplot as plt

def BV_shooting_plot(RK4_coupled, T_0, T_L, L, guess_low, guess_high, f1, f2, tol=1e-4, max_iter=50):
    # Tracking lists
    guess_lo = []
    guess_hi = []
    T_low = []
    T_high = []
    guess_all = []
    T_all = []

    for i in range(max_iter):
        # Solve with lower guess
        _, T_lo_val, _ = RK4_coupled(f1, f2, T_0, guess_low, 0.0, 0.1, 0.0, L)
        # Solve with higher guess
        _, T_hi_val, _ = RK4_coupled(f1, f2, T_0, guess_high, 0.0, 0.1, 0.0, L)

        # Store low/high guesses
        guess_lo.append(guess_low)
        T_low.append(T_lo_val[-1])
        guess_hi.append(guess_high)
        T_high.append(T_hi_val[-1])

        # Secant method guess
        zeta = guess_low + (guess_high - guess_low) * (T_L - T_lo_val[-1]) / (T_hi_val[-1] - T_lo_val[-1])
        x, T, z = RK4_coupled(f1, f2, T_0, zeta, 0.0, 0.1, 0.0, L)

        error = abs(T[-1] - T_L)
        print(f"Iter {i+1}: z(0) = {zeta:.6f}, T(L) = {T[-1]:.6f}, error = {error:.2e}")

        # Store intermediate guess
        guess_all.append(zeta)
        T_all.append(T[-1])

        if error < tol:
            print(f"Converged in {i+1} iterations: z(0) = {zeta:.6f}")
            break

        # Update bounds
        if T[-1] > T_L:
            guess_high = zeta
        else:
            guess_low = zeta

    # Plotting
    plt.plot(guess_lo, T_low, 'ro-', label='Lower guesses')
    plt.plot(guess_hi, T_high, 'bo-', label='Higher guesses')
    plt.plot(guess_all, T_all, 'go-', label='Intermediate guesses')
    plt.axhline(T_L, color='gray', linestyle='--', label='Target T(L)')
    plt.xlabel('Initial guess z(0)')
    plt.ylabel('Terminal Temperature T(L)')
    plt.title('Shooting Method Convergence')
    plt.legend()
    plt.grid()
    plt.show()

    return x, T, z

x, T, z = BV_shooting_plot(
    RK4_coupled,
    T_0=40,
    T_L=200,
    L=10,
    guess_low=0,
    guess_high=100,
    f1=dT_dx,
    f2=dz_dx,
    tol=1e-4,
    max_iter=50
)
from assign_16 import *
# assign 16 plot
# plotting the data and fits
import numpy as np
import matplotlib.pyplot as plt

x_vals = np.linspace(min(x), max(x), 300)
y_pow_vals = [math.exp(a_pow) * (x_val ** b_pow) for x_val in x_vals]
y_expo_vals = [math.exp(a_expo) * math.exp(b_expo * x_val) for x_val in x_vals]

fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=sigma_i, fmt='o', label='Data with error bars')
ax.plot(x_vals, y_pow_vals, label='Power Law Fit', color='red')
ax.plot(x_vals, y_expo_vals, label='Exponential Fit', color='green')

# show r^2 values on the plot
textstr = f"Power law: r^2 = {r2_pow:.4f}\nExponential: r^2 = {r2_expo:.4f}"
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Data Fitting using Linear Regression on Transformed Data')
ax.legend()
ax.grid()
plt.show()
