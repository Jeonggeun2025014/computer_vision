import cv2
import numpy as np


# --------------------------------------------------
# 0. 3x3 블록에 값을 넣는 함수
# --------------------------------------------------
def put_block(img, y, x, value, block_size=3):
    noisy = img

    h, w = img.shape[:2]
    half = block_size // 2

    y1 = max(y - half, 0)
    y2 = min(y + half + 1, h)
    x1 = max(x - half, 0)
    x2 = min(x + half + 1, w)

    noisy[y1:y2, x1:x2] = value


# --------------------------------------------------
# 1. Salt and Pepper Noise
#    3x3 영역을 0 또는 255로 변경
# --------------------------------------------------
def add_salt_and_pepper_noise(img, amount=0.03, salt_ratio=0.5, block_size=3):
    noisy = img.copy()

    h, w = img.shape[:2]

    # block_size x block_size 영역이 생기므로 개수를 줄여줌
    num_blocks = int(amount * h * w / (block_size * block_size))

    num_salt = int(num_blocks * salt_ratio)
    num_pepper = num_blocks - num_salt

    # Salt noise: 흰색 블록
    y_salt = np.random.randint(0, h, num_salt)
    x_salt = np.random.randint(0, w, num_salt)

    # Pepper noise: 검은색 블록
    y_pepper = np.random.randint(0, h, num_pepper)
    x_pepper = np.random.randint(0, w, num_pepper)

    for y, x in zip(y_salt, x_salt):
        put_block(noisy, y, x, 255, block_size)

    for y, x in zip(y_pepper, x_pepper):
        put_block(noisy, y, x, 0, block_size)

    return noisy


# --------------------------------------------------
# 2. Impulse Noise
#    3x3 영역을 임의의 밝기값으로 변경
# --------------------------------------------------
def add_impulse_noise(img, amount=0.03, block_size=3):
    noisy = img.copy()

    h, w = img.shape[:2]
    num_blocks = int(amount * h * w / (block_size * block_size))

    y_positions = np.random.randint(0, h, num_blocks)
    x_positions = np.random.randint(0, w, num_blocks)

    for y, x in zip(y_positions, x_positions):
        random_value = np.random.randint(0, 256)
        put_block(noisy, y, x, random_value, block_size)

    return noisy


# --------------------------------------------------
# 3. Gaussian Noise
#    모든 픽셀에 정규분포 형태의 노이즈를 더함
#    3x3 Gaussian blur를 적용해서 노이즈가 한 픽셀보다 조금 넓게 보이게 함
# --------------------------------------------------
def add_gaussian_noise(img, mean=0, sigma=25, kernel_size=3):
    img_float = img.astype(np.float32)

    # 1. 모든 픽셀에 Gaussian noise 생성
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)

    # 2. 노이즈를 3x3 크기로 약간 퍼지게 만듦
    noise = cv2.GaussianBlur(noise, (kernel_size, kernel_size), 0)

    # 3. blur 후 약해진 노이즈 크기를 다시 sigma 정도로 맞춤
    noise = noise - noise.mean()
    noise = noise / (noise.std() + 1e-8) * sigma

    # 4. 원본 이미지에 노이즈 추가
    noisy = img_float + noise

    noisy = np.clip(noisy, 0, 255)
    noisy = noisy.astype(np.uint8)

    return noisy


# --------------------------------------------------
# 4. 이미지 읽기
# --------------------------------------------------
img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("이미지 로드 실패")
    exit()


# --------------------------------------------------
# 5. 노이즈 추가
# --------------------------------------------------
block_size = 3

salt_pepper_img = add_salt_and_pepper_noise(
    img,
    amount=0.05,
    salt_ratio=0.5,
    block_size=block_size
)

impulse_img = add_impulse_noise(
    img,
    amount=0.05,
    block_size=block_size
)

gaussian_img = add_gaussian_noise(
    img,
    mean=0,
    sigma=15,
    kernel_size=3
)


# --------------------------------------------------
# 6. 결과 이미지 저장
# --------------------------------------------------
cv2.imwrite("gray_image.png", img)
cv2.imwrite("salt_and_pepper_noise_3x3.png", salt_pepper_img)
cv2.imwrite("impulse_noise_3x3.png", impulse_img)
cv2.imwrite("gaussian_noise_3x3.png", gaussian_img)

print("3x3 노이즈 이미지 저장 완료")