import cv2
import matplotlib.pyplot as plt
from networkx import edges

img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("이미지 로드 실패")
    exit()

maxVal = [150, 200, 250]
minVal = [ 50, 100, 150]

# Canny Edge Detection
edges1 = cv2.Canny(img, minVal[0], maxVal[0])
edges2 = cv2.Canny(img, minVal[1], maxVal[1])
edges3 = cv2.Canny(img, minVal[2], maxVal[2])

# -----------------------------
# 출력
# -----------------------------
plt.figure(figsize=(8,4))

plt.subplot(1,4,1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,4,2)
plt.title("Canny Edge 50")
plt.imshow(edges1, cmap='gray')
plt.axis('off')

plt.subplot(1,4,3)
plt.title("Canny Edge 100")
plt.imshow(edges2, cmap='gray')
plt.axis('off')

plt.subplot(1,4,4)
plt.title("Canny Edge 150")
plt.imshow(edges3, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()