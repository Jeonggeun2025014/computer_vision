import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기 (그레이스케일)
img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("이미지 로드 실패")
    exit()

# -----------------------------
# Sobel X (가로 방향 변화 → 수직 엣지)
# -----------------------------
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

# -----------------------------
# Sobel Y (세로 방향 변화 → 수평 엣지)
# -----------------------------
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# -----------------------------
# 크기 (엣지 강도)
# -----------------------------
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

# 보기 좋게 변환
sobel_x = np.abs(sobel_x)
sobel_y = np.abs(sobel_y)
sobel_mag = np.clip(sobel_mag, 0, 255)

# -----------------------------
# 출력
# -----------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,4,1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,4,2)
plt.title("Sobel X")
plt.imshow(sobel_x, cmap='gray')
plt.axis('off')

plt.subplot(1,4,3)
plt.title("Sobel Y")
plt.imshow(sobel_y, cmap='gray')
plt.axis('off')

plt.subplot(1,4,4)
plt.title("Magnitude")
plt.imshow(sobel_mag, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()