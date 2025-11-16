
from SVD_star_denoising import *

# Create X, Y coordinate grids
x = np.arange(0, noisy_image.shape[1])
y = np.arange(0, noisy_image.shape[0])
X, Y = np.meshgrid(x, y)

Z = noisy_image  # intensities

# Plot in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='gray', edgecolor='none')

ax.set_title("3D Intensity Surface of Sliced Image")
ax.set_xlabel("Column Index (X)")
ax.set_ylabel("Row Index (Y)")
ax.set_zlabel("Intensity (Z)")

plt.show()


# 3D ORIGINAL slice in noisy_image
Z1 = noisy_image
x = np.arange(Z1.shape[1])
y = np.arange(Z1.shape[0])
X1, Y1 = np.meshgrid(x, y)

# SVD reduced slice in noise_reduced_image
Z2 = noise_reduced_image
X2, Y2 = np.meshgrid(np.arange(Z2.shape[1]), np.arange(Z2.shape[0]))

# Create side-by-side 3D figure
fig = plt.figure(figsize=(16, 7))

# ---- LEFT PLOT: ORIGINAL ----
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X1, Y1, Z1, cmap='gray', edgecolor='none')
ax1.set_title("Original Sliced Image (3D Surface)")
ax1.set_xlabel("Column Index")
ax1.set_ylabel("Row Index")
ax1.set_zlabel("Intensity")

# ---- RIGHT PLOT: SVD REDUCED ----
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X2, Y2, Z2, cmap='gray', edgecolor='none')
ax2.set_title(f"SVD Noise-Reduced Image (rank={rank})")
ax2.set_xlabel("Column Index")
ax2.set_ylabel("Row Index")
ax2.set_zlabel("Intensity")

plt.tight_layout()
plt.show()


# AMINATION


import matplotlib.animation as animation
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15,5))

# Create two plot lines
line1, = ax.plot([], [], color='black', label='Original')
line2, = ax.plot([], [], color='orange', label='Reconstructed')

# Fixed y-axis (0–1 since you normalized)
ax.set_ylim(0, 1)

# Fixed x-axis length = number of rows
ax.set_xlim(0, noisy_image.shape[0])

ax.set_xlabel("Row index")
ax.set_ylabel("Intensity (0–1)")
ax.grid(True)
ax.legend()

# Choose a fixed column
col_original = noisy_image[:, col_idx] / 255

def update(frame_rank):
    recon = svd_noise_reduction(noisy_image, frame_rank)
    col_recon = recon[:, col_idx] / 255

    line1.set_data(range(len(col_original)), col_original)
    line2.set_data(range(len(col_recon)), col_recon)

    ax.set_title(f"Column {col_idx} — Rank={frame_rank}")
    return line1, line2

# Animation object
ani = animation.FuncAnimation(
    fig,
    update,
    frames=range(1, 50, 3),
    interval=200  # interval doesn't matter for GIF speed
)

# SAVE GIF -- Slow & smooth
ani.save("svd_animation.gif", writer='pillow', fps=2)

print("Saved GIF as svd_animation.gif")
plt.close(fig)
