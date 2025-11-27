# Assignment15
# Aditya Raj, Roll no- 2311013
from My2lib import *

# Question 1

# Parameters
alpha = 0.01 # in m^-2
T_al = 20 # in deg C

def dT_dx(z):
    return z
def dz_dx(T, z):
    return alpha * (T - T_al)

# Boundary Conditions
T_0 = 40 # Temperature at x=0
T_L = 200 # Temperature at x=L
L = 10 # Length of the rod in m


t,T,z = BV_shooting(RK4_coupled,T_0,T_L,L,-10,20,dT_dx,dz_dx,1e-6,50)
for i in range(1, len(T)):
    if T[i-1] < 100 <= T[i]:
        x1, x2 = t[i-1], t[i]
        T1, T2 = T[i-1], T[i]
        x_100 = x1 + (100 - T1) * (x2 - x1) / (T2 - T1)
        break

print(f"Temperature reaches 100°C at x ≈ {x_100:.4f} meters")

Plot(t,T)

# Question 2

L = 2.0
dx = 0.1
dt = 0.004
r = dt / dx**2
N = int(L / dx) + 1  
M = 200             

# Initialize arrays
x = [i * dx for i in range(N)]
u = [0.0 for _ in range(N)]
u_new = [0.0 for _ in range(N)]

# Initial conditions - a pulse in the center
center_index = N // 2
u[center_index] = 300.0

# Store for plotting
snapshots = []
snapshot_times = []

for n in range(M):
    for i in range(1, N - 1):
        u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])
    u = u_new[:]
    if n % 20 == 0:
        snapshots.append(u[:])
        snapshot_times.append(n * dt)

# Plot snapshots
plt.figure(figsize=(8, 5))
for i, u_snap in enumerate(snapshots):
    plt.plot(x, u_snap, label=f"t={snapshot_times[i]:.2f}")
plt.xlabel("x")
plt.ylabel("Temperature u(x,t)")
plt.title("Heat Equation Evolution (Explicit Method)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
