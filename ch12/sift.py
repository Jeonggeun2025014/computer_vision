import cv2
import matplotlib.pyplot as plt

# 이미지 읽기
image = cv2.imread("inhatc_10.jpg")

if image is None:
    print("이미지 로드 실패")
    exit()

# grayscale 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# SIFT 객체 생성
sift = cv2.SIFT_create()

# keypoint와 descriptor 계산
keypoints, descriptors = sift.detectAndCompute(gray, None)

print("검출된 keypoint 개수:", len(keypoints))
print("Descriptor shape:", descriptors.shape)

# keypoint 시각화
result = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# overlay the keypoints on the img
sift_result = cv2.drawKeypoints(image, keypoints, None, (255, 0, 0))
cv2.imshow('sift_result', sift_result)

# BGR → RGB 변환 후 출력
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 6))
plt.imshow(result_rgb)
plt.title("SIFT Keypoints")
plt.axis("off")
plt.show()