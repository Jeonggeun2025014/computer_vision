import cv2
import numpy as np

img = cv2.imread('ch04_Albert_Einstein.png', cv2.IMREAD_GRAYSCALE)

kernel_1 = np.array([[0, 0, 0],
                   [0, 2, 0],
                   [0, 0, 0]], dtype=np.float32)
kernel_2 = np.ones((3, 3), dtype=np.float32) / 9.0
kernel = kernel_1 - kernel_2

sharpen = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
sharpen = np.clip(sharpen, 0, 255).astype(np.uint8)
sharpen_round = np.round(sharpen).astype(np.uint8)

cv2.imshow('Original', img)
cv2.imshow('Sharpen', sharpen_round)
cv2.waitKey(0)
cv2.destroyAllWindows()