# import cv2
# import numpy as np

# # 이미지 읽기
# img = cv2.imread("lena.png")

# if img is None:
#     print("이미지 로드 실패")
#     exit()

# # Gaussian Pyramid 생성
# gaussian_pyramid = [img]
# levels = 5

# current = img.copy()
# for i in range(levels):
#     current = cv2.pyrDown(current)   # Gaussian blur + downsampling
#     gaussian_pyramid.append(current)

# # 결과 출력
# for i, level_img in enumerate(gaussian_pyramid):
#     cv2.imshow(f"Gaussian Level {i}", level_img)
#     cv2.imwrite(f"gaussian_level_{i}.jpg", level_img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # 이미지 읽기
# img = cv2.imread("lena.png")

# if img is None:
#     print("이미지 로드 실패")
#     exit()

# levels = 4

# # -----------------------------
# # Gaussian Pyramid 생성
# # -----------------------------
# gaussian = [img]
# current = img.copy()

# for i in range(levels):
#     current = cv2.pyrDown(current)
#     gaussian.append(current)

# # -----------------------------
# # 하나의 이미지로 합치기
# # -----------------------------
# h, w = img.shape[:2]

# # 오른쪽에 쌓기 위한 캔버스
# canvas = np.zeros((h, w + w//2, 3), dtype=np.uint8)

# # 왼쪽: 원본 이미지
# canvas[0:h, 0:w] = img

# # 오른쪽: 작은 이미지들 위에서부터 쌓기
# y_offset = 0
# for i in range(1, len(gaussian)):
#     h_i, w_i = gaussian[i].shape[:2]
#     canvas[y_offset:y_offset+h_i, w:w+w_i] = gaussian[i]
#     y_offset += h_i

# # -----------------------------
# # 출력
# # -----------------------------
# cv2.imshow("Gaussian Pyramid", canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 읽기
img = cv2.imread("lena.png")

if img is None:
    print("이미지 로드 실패")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

levels = 5

# Gaussian Pyramid
G = [img]
for i in range(levels):
    G.append(cv2.pyrDown(G[i]))

# Laplacian Pyramid
L = []
for i in range(levels):
    up = cv2.pyrUp(G[i+1])
    up = cv2.resize(up, (G[i].shape[1], G[i].shape[0]))  # 크기 맞추기

    lap = G[i] - up
    L.append(lap)

h, w = img.shape[:2]    # 캔버스 생성
canvas = np.zeros((h, w + w//2, 3), dtype=np.uint8)
canvas[0:h, 0:w] = img  # 왼쪽: 원본 이미지

# 오른쪽: Gaussian과 동일한 계단 구조로 배치
x_offset = w
y_offset = 0

for i in range(1, len(G)):
    h_i, w_i = G[i].shape[:2]

    lap = L[i-1]    # Laplacian 가져오기
    lap_show = np.clip(lap + 128, 0, 255).astype(np.uint8)  # 보기 좋게 변환 (음수 처리)
    lap_show = cv2.resize(lap_show, (w_i, h_i)) # 크기 맞추기 (핵심)
    canvas[y_offset:y_offset+h_i, x_offset:x_offset+w_i] = lap_show # 배치
    y_offset += h_i // 2    # 계단 구조

# 출력
plt.figure(figsize=(8,6))
plt.imshow(canvas)
plt.title("Laplacian Pyramid (Same Layout as Gaussian)")
plt.axis('off')
plt.show()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# img = cv2.imread("lena.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# levels = 5

# # -----------------------------
# # Gaussian Pyramid
# # -----------------------------
# G = [img]
# for i in range(levels):
#     G.append(cv2.pyrDown(G[i]))

# # -----------------------------
# # Laplacian Pyramid
# # -----------------------------
# L = []
# for i in range(levels):
#     up = cv2.pyrUp(G[i+1])
#     up = cv2.resize(up, (G[i].shape[1], G[i].shape[0]))

#     lap = G[i] - up
#     L.append(lap)

# # -----------------------------
# # Gaussian 기준으로 위치 계산
# # -----------------------------
# h, w = img.shape[:2]
# canvas = np.zeros((h, w + w//2, 3), dtype=np.uint8)

# # 왼쪽: 원본 (그대로)
# canvas[0:h, 0:w] = img

# # -----------------------------
# # 오른쪽: Gaussian 위치 그대로 사용
# # -----------------------------
# x_offset = w
# y_offset = 0

# for i in range(1, len(G)):
#     h_i, w_i = G[i].shape[:2]

#     # Laplacian 넣기 (i-1 사용)
#     lap = L[i-1]

#     # 시각화용 변환 (핵심🔥)
#     lap_show = np.clip(lap + 128, 0, 255).astype(np.uint8)

#     canvas[y_offset:y_offset+h_i, x_offset:x_offset+w_i] = lap_show

#     # Gaussian과 동일한 계단 구조 유지
#     y_offset += h_i // 2

# # -----------------------------
# # 출력
# # -----------------------------
# plt.figure(figsize=(8,6))
# plt.imshow(canvas)
# plt.title("Laplacian Pyramid (Same Layout)")
# plt.axis('off')
# plt.show()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# img = cv2.imread("lena.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# levels = 5

# # -----------------------------
# # Gaussian Pyramid 생성
# # -----------------------------
# G = [img]
# for i in range(levels):
#     G.append(cv2.pyrDown(G[i]))

# # -----------------------------
# # 캔버스 생성
# # -----------------------------
# h, w = img.shape[:2]

# canvas = np.zeros((h, w + w//2, 3), dtype=np.uint8)

# # 왼쪽: 원본
# canvas[0:h, 0:w] = img

# # -----------------------------
# # 오른쪽 계단식 배치 (핵심)
# # -----------------------------
# x_offset = w
# y_offset = 0

# for i in range(1, len(G)):
#     h_i, w_i = G[i].shape[:2]

#     canvas[y_offset:y_offset+h_i, x_offset:x_offset+w_i] = G[i]

#     y_offset += h_i // 2   # 👉 계단 느낌 핵심
#     x_offset += 0          # 오른쪽 고정

# # -----------------------------
# # 출력
# # -----------------------------
# plt.figure(figsize=(8,6))
# plt.imshow(canvas)
# plt.title("Gaussian Pyramid")
# plt.axis('off')
# plt.show()