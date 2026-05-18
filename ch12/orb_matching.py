import cv2
import matplotlib.pyplot as plt

# 1. 이미지 읽기
img1 = cv2.imread("inhatc_10_1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("inhatc_10_2_rotated.png", cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("이미지 로드 실패")
    exit()

# 2. ORB 객체 생성
orb = cv2.ORB_create(nfeatures=1000)

# 3. 특징점과 descriptor 계산
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

if des1 is None or des2 is None:
    print("특징점 검출 실패")
    exit()

print("Image 1 keypoints:", len(kp1))
print("Image 2 keypoints:", len(kp2))

# 4. Descriptor Matching# ORB는 binary descriptor이므로 Hamming distance 사용
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

# 거리 기준 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 좋은 매칭 일부만 선택
good_matches = matches[:50]

# 5. 매칭 결과 시각화
result = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 6. 출력
plt.figure(figsize=(14, 7))
plt.imshow(result, cmap="gray")
plt.title("ORB Feature Matching")
plt.axis("off")
plt.show()
