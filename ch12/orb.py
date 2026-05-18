import cv2
import matplotlib.pyplot as plt

# 이미지 읽기
img = cv2.imread("inhatc_10.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("이미지 로드 실패")
    exit()

# ORB 객체 생성
orb = cv2.ORB_create(nfeatures=500)

# 특징점과 descriptor 계산
keypoints, descriptors = orb.detectAndCompute(img, None)

# 특징점 그리기
result = cv2.drawKeypoints(
    img,
    keypoints,
    None,
    color=(0, 255, 0),
    flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
)

# overlay the keypoints on the img
orb_result = cv2.drawKeypoints(img, keypoints, None, (255, 0, 0))
cv2.imshow('orb_result', orb_result)

# 출력
plt.figure(figsize=(8, 6))
plt.imshow(result, cmap="gray")
plt.title("ORB Features")
plt.axis("off")
plt.show()

print("검출된 특징점 개수:", len(keypoints))
print("Descriptor shape:", descriptors.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()