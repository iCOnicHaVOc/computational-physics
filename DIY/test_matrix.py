import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ------------------------------
# 1) Create a clean 10x10 matrix
# ------------------------------
np.random.seed(0)
clean = np.random.randint(0, 256, (10, 10)).astype(np.float64)

print("Clean Matrix:")
print(clean)

# ------------------------------
# 2) Add Gaussian noise
# ------------------------------
def add_gaussian_noise(mat, sigma=30):
    noise = np.random.normal(0, sigma, mat.shape)
    noisy = mat + noise
    return np.clip(noisy, 0, 255)

noisy = add_gaussian_noise(clean, sigma=30)

print("\nNoisy Matrix:")
print(np.round(noisy))

# ------------------------------
# 3) SVD-based denoising
# ------------------------------
def svd_denoise(mat, rank):
    U, S, Vt = np.linalg.svd(mat, full_matrices=False)
    S_reduced = np.zeros_like(S)
    S_reduced[:rank] = S[:rank]
    recon = U @ np.diag(S_reduced) @ Vt
    return np.clip(recon, 0, 255)

# Choose some ranks
ranks = [1, 2, 3, 5, 8,10,13,15]

# ------------------------------
# 4) Print PSNR & SSIM for each rank
# ------------------------------
print("\n===== DENOISING METRICS =====")
for k in ranks:
    den = svd_denoise(noisy, k)
    p = psnr(clean, den, data_range=255)
    s = ssim(clean, den, data_range=255)
    print(f"Rank {k}: PSNR = {p:.2f} dB, SSIM = {s:.4f}")

# ------------------------------
# 5) Visual Grid: Clean, Noisy, Rank-k Results
# ------------------------------
plt.figure(figsize=(14, 6))

plt.subplot(2, len(ranks) + 2, 1)
plt.imshow(clean, cmap="gray")
plt.title("Clean")
plt.axis("off")

plt.subplot(2, len(ranks) + 2, 2)
plt.imshow(noisy, cmap="gray")
plt.title("Noisy")
plt.axis("off")

# Reconstructed and difference images
for i, k in enumerate(ranks):
    den = svd_denoise(noisy, k)
    diff = np.abs(clean - den)

    # Top row → reconstructed images
    plt.subplot(2, len(ranks) + 2, i + 3)
    plt.imshow(den, cmap="gray")
    plt.title(f"Rank {k}")
    plt.axis("off")

    # Bottom row → difference heatmaps
    plt.subplot(2, len(ranks) + 2, i + len(ranks) + 5)
    plt.imshow(diff, cmap='hot')
    plt.title(f"Diff k={k}")
    plt.axis("off")

plt.tight_layout()
plt.show()
