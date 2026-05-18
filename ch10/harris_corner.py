import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("inhatc_10.jpg")

if image is None:
    print("이미지 로드 실패")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

harris_corner = cv2.cornerHarris(gray, blockSize=6, ksize=3, k=0.04)
harris_corner = cv2.dilate(harris_corner, None)

result = image_rgb.copy()
result[harris_corner > 0.01 * harris_corner.max()] = [255, 0, 0]

# 4. 출력
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Harris Corners")
plt.imshow(result)
plt.axis("off")

plt.tight_layout()
plt.show()






