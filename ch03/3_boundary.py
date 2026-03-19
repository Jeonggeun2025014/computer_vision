import cv2
import numpy as np
import matplotlib.pyplot as plt

# 가장자리까지 밝은 이미지 생성
img = np.zeros((120,120), dtype=np.uint8)
img[:, :60] = 255   # 왼쪽은 흰색
img[:, 60:] = 0     # 오른쪽은 검정

# 큰 평균 필터
kernel = np.ones((25,25), np.float32) / (25*25)

constant = cv2.filter2D(img,-1,kernel,borderType=cv2.BORDER_CONSTANT)
wrap = cv2.filter2D(img,-1,kernel,borderType=cv2.BORDER_WRAP)
replicate = cv2.filter2D(img,-1,kernel,borderType=cv2.BORDER_REPLICATE)
reflect = cv2.filter2D(img,-1,kernel,borderType=cv2.BORDER_REFLECT)

plt.figure(figsize=(10,6))

plt.subplot(231); plt.title("Original")
plt.imshow(img,cmap='gray'); plt.axis("off")

plt.subplot(232); plt.title("Clip (black)")
plt.imshow(constant,cmap='gray'); plt.axis("off")

plt.subplot(233); plt.title("Wrap around")
plt.imshow(wrap,cmap='gray'); plt.axis("off")

plt.subplot(234); plt.title("Copy edge")
plt.imshow(replicate,cmap='gray'); plt.axis("off")

plt.subplot(235); plt.title("Reflect")
plt.imshow(reflect,cmap='gray'); plt.axis("off")

plt.show()