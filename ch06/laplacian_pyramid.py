import cv2
import matplotlib.pyplot as plt

img = cv2.imread("lena.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

levels = 4

# -----------------------------
# Gaussian Pyramid
# -----------------------------
G = [img]
for i in range(levels):
    G.append(cv2.pyrDown(G[i]))

# -----------------------------
# Laplacian Pyramid
# -----------------------------
L = []

for i in range(levels):
    up = cv2.pyrUp(G[i+1])
    up = cv2.resize(up, (G[i].shape[1], G[i].shape[0]))  # 핵심

    lap = G[i] - up
    L.append(lap)

# -----------------------------
# 출력
# -----------------------------
plt.figure(figsize=(12,6))

for i in range(levels):
    plt.subplot(2, levels, i+1)
    plt.title(f"G{i}")
    plt.imshow(G[i]); plt.axis('off')

    plt.subplot(2, levels, i+1+levels)
    plt.title(f"L{i}")
    plt.imshow(L[i] + 128); plt.axis('off')

plt.tight_layout()
plt.show()

# import cv2
# import matplotlib.pyplot as plt

# img = cv2.imread("lena.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # Gaussian
# g1 = cv2.pyrDown(img)
# g2 = cv2.pyrDown(g1)

# # 크기 맞추기
# up_g2 = cv2.pyrUp(g2)
# up_g2 = cv2.resize(up_g2, (g1.shape[1], g1.shape[0]))

# up_g1 = cv2.pyrUp(g1)
# up_g1 = cv2.resize(up_g1, (img.shape[1], img.shape[0]))

# # Laplacian
# l0 = img - up_g1
# l1 = g1 - up_g2

# # 출력
# plt.figure(figsize=(8,4))

# plt.subplot(1,3,1)
# plt.title("Original")
# plt.imshow(img); plt.axis('off')

# plt.subplot(1,3,2)
# plt.title("Laplacian 0")
# plt.imshow(l0 + 128); plt.axis('off')

# plt.subplot(1,3,3)
# plt.title("Laplacian 1")
# plt.imshow(l1 + 128); plt.axis('off')

# plt.show()


