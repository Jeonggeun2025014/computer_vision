import numpy as np
import cv2

# 7x7 임펄스 이미지
img = np.zeros((7,7), dtype=np.float32)
img[3,3] = 1

# 필터
kernel = np.array([[1,2,1],
                   [2,4,2],
                   [1,2,1]], dtype=np.float32) / 16

# 적용
out = cv2.filter2D(img, -1, kernel)

print(out)