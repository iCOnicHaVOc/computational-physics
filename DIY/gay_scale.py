import cv2
import matplotlib.pyplot as plt

image_path = r"DATA\testing1\test_img1.jpg"
noisy_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
plt.imshow(noisy_image)
plt.show()