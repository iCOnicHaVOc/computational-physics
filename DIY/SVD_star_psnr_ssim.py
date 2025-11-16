
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# -------------------------------------------------------------
# 1. SVD Noise Reduction
# -------------------------------------------------------------
def svd_noise_reduction(image, rank):
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    S_reduced = np.zeros_like(S)
    S_reduced[:rank] = S[:rank]
    recon = U @ np.diag(S_reduced) @ Vt
    return np.clip(recon, 0, 255).astype(np.uint8)

# -------------------------------------------------------------
# 2. Add Gaussian Noise
# -------------------------------------------------------------
def add_gaussian_noise(image, mean=0, sigma=20):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# -------------------------------------------------------------
# 3. Load CLEAN image
# -------------------------------------------------------------
clean = cv2.imread("m46_good_done111.png", cv2.IMREAD_GRAYSCALE)
clean_f = clean.astype(np.float64)

# Make noisy image
noisy = add_gaussian_noise(clean, sigma=30)
noisy_f = noisy.astype(np.float64)

# -------------------------------------------------------------
# 4. Baseline: CLEAN vs NOISY
# -------------------------------------------------------------
psnr_noisy = psnr(clean_f, noisy_f, data_range=255)
ssim_noisy = ssim(clean_f, noisy_f, data_range=255)
print("================================")
print(" CLEAN vs NOISY")
print("================================")
print(f"PSNR = {psnr_noisy:.2f} dB")
print(f"SSIM = {ssim_noisy:.4f}\n")

# -------------------------------------------------------------
# 5. CLEAN vs SVD-DENOISED for multiple ranks
# -------------------------------------------------------------
psnr_values = []
ssim_values = []
ranks = []
for r in range(1,150):
    den = svd_noise_reduction(noisy, r).astype(np.float64)

    p = psnr(clean_f, den, data_range=255)
    s = ssim(clean_f, den, data_range=255)

    psnr_values.append(p)
    ssim_values.append(s)

    print(f"Rank {r}: PSNR = {p:.2f} dB, SSIM = {s:.4f}")
    ranks.append(r)
# -------------------------------------------------------------
# 6. Plot PSNR and SSIM vs Rank
# -------------------------------------------------------------
plt.figure(figsize=(12,5))

# PSNR
plt.subplot(1, 2, 1)
plt.plot(ranks, psnr_values, marker='o', linestyle='--')
plt.title("PSNR vs Rank")
plt.xlabel("Rank")
plt.ylabel("PSNR (dB)")
plt.grid(True)

# SSIM
plt.subplot(1, 2, 2)
plt.plot(ranks, ssim_values, marker='o', linestyle='--')
plt.title("SSIM vs Rank")
plt.xlabel("Rank")
plt.ylabel("SSIM")
plt.grid(True)

plt.tight_layout()
plt.show()


# reault
''' CLEAN vs NOISY
================================
PSNR = 21.67 dB
SSIM = 0.0201

Rank 5: PSNR = 27.05 dB, SSIM = 0.0909 
Rank 10: PSNR = 26.92 dB, SSIM = 0.0858 
Rank 15: PSNR = 26.78 dB, SSIM = 0.0809 
Rank 20: PSNR = 26.64 dB, SSIM = 0.0768 
Rank 30: PSNR = 26.39 dB, SSIM = 0.0700 
Rank 50: PSNR = 25.96 dB, SSIM = 0.0604
'''
