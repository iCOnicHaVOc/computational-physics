# 3D phase space plot with time as third axis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D projection in some mpl versions
from assign_14 import yu, gu, p  # importing time, displacement, momentum arrays
import numpy as np

# ensure arrays are numpy arrays and have the same length
t = np.asarray(yu)      # time
x = np.asarray(gu)      # displacement
p_arr = np.asarray(p)   # momentum

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# parametric line in 3D: displacement (x), momentum (p), time (t)
ax.plot(x, p_arr, t, lw=1, color='tab:blue')

ax.set_xlabel('Displacement (m)')
ax.set_ylabel('Momentum (kg m/s)')
ax.set_zlabel('Time (s)')
ax.set_title('3D Phase Space (time as third axis)')
plt.tight_layout()
plt.show()
