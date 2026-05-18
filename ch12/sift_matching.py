import cv2
import matplotlib.pyplot as plt

# 이미지 읽기
img1 = cv2.imread("inhatc_10_1.png")
img2 = cv2.imread("inhatc_10_2_rotated.png")

if img1 is None or img2 is None:
    print("이미지 로드 실패")
    exit()

# grayscale 변환
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT 생성
sift = cv2.SIFT_create()

# keypoint와 descriptor 계산
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# BFMatcher 생성
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# descriptor 매칭
matches = bf.match(des1, des2)

# 거리 기준으로 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 좋은 매칭 일부만 표시
good_matches = matches[:50]

# 매칭 결과 그리기
result = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# BGR → RGB 변환
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(14, 7))
plt.imshow(result_rgb)
plt.title("SIFT Feature Matching")
plt.axis("off")
plt.show()