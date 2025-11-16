import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def svd_noise_reduction(image, rank):
    """
    Perform SVD-based noise reduction on a grayscale image.

    Parameters:
        image (ndarray): Grayscale image as a 2D NumPy array.
        rank (int): Number of singular values to keep.

    Returns:
        ndarray: Denoised grayscale image (uint8).
    """
    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(image, full_matrices=False)

    S_reduced = np.zeros_like(S) # matrix filled with 0
    S_reduced[:rank] = S[:rank]  

    # Reconstruct the image using reduced singular values
    noise_reduced_image = np.dot(U, np.dot(np.diag(S_reduced), Vt))

    # Clip pixel values to valid range [0, 255], some values may be outside the range
    noise_reduced_image = np.clip(noise_reduced_image, 0, 255)

    # Cut out decimal places to unsigned 8-bit integer for display
    return noise_reduced_image.astype(np.uint8)

def add_gaussian_noise(image, mean=0, sigma=20):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

# Load a image
image_path = "m46_good_done111.png"
noisy_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # make image gray

# cutour a portion for good reduction, and to generate column plots 
row_start, row_end = 500, 800
col_start, col_end = 380, 1300  

original_slice = noisy_image[row_start:row_end, :]
noisy_image = add_gaussian_noise(original_slice, sigma=25)
plt.imshow(noisy_image, cmap="gray"); plt.show()

# Set desired rank for noise reduction
rank = 20
# Apply SVD noise reduction
noise_reduced_image = svd_noise_reduction(noisy_image, rank)

'''
# for showing image side by side
plt.figure(figsize=(12, 6))

# LEFT: original sliced region
plt.subplot(1, 2, 1)
plt.title("Original Sliced Image")
plt.imshow(noisy_image, cmap="gray")
plt.axis("off")

# RIGHT: SVD noise reduced region
plt.subplot(1, 2, 2)
plt.title(f"SVD Noise-Reduced Image (rank={rank})")
plt.imshow(noise_reduced_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
'''

# Pick a random column to visualize intensity noise reduction 
col_idx =  650 #np.random.randint(0, noisy_image.shape[1]) 
print(f"Selected column index: {col_idx}")

col_original = noisy_image[30:, col_idx]
col_recon = noise_reduced_image[30:, col_idx]

plt.figure(figsize=(13, 5))
plt.plot(col_original,color= 'black', label='Original')
plt.plot(col_recon,color = 'orange', label=f'Reconstructed (k={rank})')
plt.title(f'Column {col_idx}: Intensity vs Row Index')
plt.xlabel('Row index')
plt.ylabel('Intensity (0-1)')
plt.legend(); plt.grid(True); plt.show()
