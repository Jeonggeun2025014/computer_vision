import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("ch03_1.png", cv2.IMREAD_GRAYSCALE) # 이미지 읽기
k = 15 # Box filter 크기
blur = cv2.blur(img, (k, k)) # 평균 필터 적용

kernel = np.ones((k, k)) / (k * k)


plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Box Filter")
plt.imshow(kernel, cmap='gray')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Smoothed Image")
plt.imshow(blur, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()