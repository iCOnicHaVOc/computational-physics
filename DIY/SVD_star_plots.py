import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------------------
# Load the grayscale image
# -------------------------------
image_path = "test_img1.jpg"
gay_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# -------------------------------
# Compute SVD once
# -------------------------------
U, S, Vt = np.linalg.svd(gay_image, full_matrices=False)

'''
# -------------------------------
# 1️⃣ Singular Values (Linear Scale)
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(S, linestyle='--', linewidth=2)
plt.title("Singular Values vs Index")
plt.xlabel("Index (k)")
plt.ylabel("Singular Value Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# 1B️⃣ Singular Values (Log Scale)
# -------------------------------
plt.figure(figsize=(12,7))
plt.semilogy(S, linestyle='--', linewidth=2)
plt.title("Singular Values vs Index (Log Scale)")
plt.xlabel("Index (k)")
plt.ylabel("Singular Value Magnitude (log scale)")
plt.grid(True, which='both')
plt.tight_layout()
plt.show()
'''
# -------------------------------
# 2️⃣ Cumulative Energy
# -------------------------------
energy = S**2
cumulative_energy = np.cumsum(energy) / np.sum(energy)

num_sv = len(S)
percent_k = (np.arange(num_sv) / (num_sv - 1)) * 100

# Thresholds for annotation
targets = [0.20, 0.40, 0.60, 0.80]
threshold_positions = {}

for t in targets:
    idx = np.where(cumulative_energy >= t)[0][0]
    threshold_positions[t] = idx


# Annotate
for t, idx in threshold_positions.items():
    x = percent_k[idx]
    y = cumulative_energy[idx]

    plt.axvline(x, color='red', linestyle='--', alpha=0.4)

    plt.text(
        x + 1,
        y - 0.05,
        f"{int(t*100)}% energy at k={idx} ({x:.1f}% SVs)",
        fontsize=9,
        color='blue'
    )

plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# -------------------------------
# 2B️⃣ Cumulative Energy (Log Y-axis)
# -------------------------------
plt.figure(figsize=(12,6))
plt.semilogy(percent_k, cumulative_energy, linestyle='--', linewidth=2)
plt.title("Cumulative Energy vs % of Singular Values (Log Scale)")
plt.xlabel("% of Singular Values Used")
plt.ylabel("Cumulative Energy (log scale)")
plt.grid(True, which='both')
plt.xlim(0, 20)     # <--- LIMIT X AXIS TO 50%

plt.tight_layout()
plt.show()
