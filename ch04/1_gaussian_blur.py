import cv2

# 이미지 읽기
img = cv2.imread("ch04_panda.png", cv2.IMREAD_GRAYSCALE)

# 가우시안 필터 적용
blur1 = cv2.GaussianBlur(img, (5, 5), 1)
blur2 = cv2.GaussianBlur(img, (11, 11), 3)
blur3 = cv2.GaussianBlur(img, (21, 21), 5)

# 결과 출력
cv2.imshow("Original", img)
cv2.imshow("Gaussian Blur sigma=1", blur1)
cv2.imshow("Gaussian Blur sigma=3", blur2)
cv2.imshow("Gaussian Blur sigma=5", blur3)

cv2.waitKey(0)
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # 이미지 읽기
# img = cv2.imread("panda.png", cv2.IMREAD_GRAYSCALE)

# # Gaussian Filter 적용
# kernel_size = 11
# sigma = 3
# blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

# # ----------------------------
# # Gaussian Kernel 생성
# # ----------------------------
# g1d = cv2.getGaussianKernel(kernel_size, sigma)
# kernel = g1d @ g1d.T   # 2D Gaussian Kernel

# # ----------------------------
# # 결과 출력
# # ----------------------------
# plt.figure(figsize=(10,4))

# plt.subplot(1,3,1)
# plt.title("Original Image")
# plt.imshow(img, cmap='gray')
# plt.axis('off')

# plt.subplot(1,3,2)
# plt.title("Gaussian Blur")
# plt.imshow(blur, cmap='gray')
# plt.axis('off')

# plt.subplot(1,3,3)
# plt.title("Gaussian Kernel")
# plt.imshow(kernel, cmap='jet')
# plt.colorbar()
# plt.axis('off')

# plt.tight_layout()
# plt.show()
