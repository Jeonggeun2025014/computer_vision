import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# 1. 실험 데이터 생성
np.random.seed(10)

# x 데이터
x = np.linspace(0, 100, 100)

# 실제 2차 함수: y = ax^2 + bx + c
true_y = -1.5 * x**2 + 120 * x - 2500

# 작은 노이즈가 포함된 데이터
noise_y = true_y + np.random.normal(0, 180, size=len(x))

# 이상치가 포함된 데이터
outlier_y = noise_y.copy()

# 일부 구간에 큰 이상치 추가
outlier_index = np.arange(52, 70)
outlier_y[outlier_index] -= np.array([
    500, 1000, 1800, 3000, 4300, 5200, 4700, 3900, 3000,
    2300, 1700, 1300, 1000, 800, 650, 500, 350, 200
])

# 입력 데이터를 sklearn 형태로 변환
X = x.reshape(-1, 1)

# 출력용 촘촘한 x 데이터
x_plot = np.linspace(0, 100, 500).reshape(-1, 1)


# 2. 2차 최소제곱법 모델
def fit_least_squares(X, y):
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression()
    )
    model.fit(X, y)
    return model


# 3. 2차 RANSAC 모델
def fit_ransac(X, y):
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=3,           # 2차 곡선 계산에 필요한 최소 점 개수
            residual_threshold=350,  # inlier 판단 기준
            max_trials=1000,
            random_state=10
        )
    )
    model.fit(X, y)
    return model


# 4. 노이즈 데이터 fitting
ls_noise_model = fit_least_squares(X, noise_y)
ransac_noise_model = fit_ransac(X, noise_y)

ls_noise_pred = ls_noise_model.predict(x_plot)
ransac_noise_pred = ransac_noise_model.predict(x_plot)


# 5. Outlier 데이터 fitting
ls_outlier_model = fit_least_squares(X, outlier_y)
ransac_outlier_model = fit_ransac(X, outlier_y)

ls_outlier_pred = ls_outlier_model.predict(x_plot)
ransac_outlier_pred = ransac_outlier_model.predict(x_plot)

# RANSAC이 판단한 inlier / outlier 정보
ransac_step = ransac_outlier_model.named_steps["ransacregressor"]
inlier_mask = ransac_step.inlier_mask_
outlier_mask = ~inlier_mask


# 6. 결과 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# ---------------- 상단: 노이즈 데이터 ----------------
# (1) 노이즈 데이터
axes[0, 0].scatter(x, noise_y, s=12, marker="+", label="Noise data")
axes[0, 0].set_title("Noise Data")
axes[0, 0].grid(True)

# (2) 노이즈 데이터 + 최소제곱법
axes[0, 1].scatter(x, noise_y, s=12, marker="+")
axes[0, 1].plot(x_plot, ls_noise_pred, linewidth=2, label="Least Squares")
axes[0, 1].set_title("Least Squares")
axes[0, 1].grid(True)
axes[0, 1].legend()

# (3) 노이즈 데이터 + RANSAC
axes[0, 2].scatter(x, noise_y, s=12, marker="+")
axes[0, 2].plot(x_plot, ransac_noise_pred, linewidth=2, label="RANSAC")
axes[0, 2].set_title("RANSAC")
axes[0, 2].grid(True)
axes[0, 2].legend()

# ---------------- 하단: Outlier 데이터 ----------------
# (4) Outlier 데이터
axes[1, 0].scatter(x, outlier_y, s=12, marker="+", label="Outlier data")
axes[1, 0].set_title("Outlier Data")
axes[1, 0].grid(True)

# (5) Outlier 데이터 + 최소제곱법
axes[1, 1].scatter(x, outlier_y, s=12, marker="+")
axes[1, 1].plot(x_plot, ls_outlier_pred, linewidth=2, label="Least Squares")
axes[1, 1].set_title("Least Squares: Affected by Outliers")
axes[1, 1].grid(True)
axes[1, 1].legend()

# (6) Outlier 데이터 + RANSAC
axes[1, 2].scatter(
    x[inlier_mask], outlier_y[inlier_mask], s=12, marker="+", label="Inliers"
)

axes[1, 2].scatter(
    x[outlier_mask], outlier_y[outlier_mask], s=20, marker="x", label="Rejected points"
)

axes[1, 2].plot(
    x_plot, ransac_outlier_pred, linewidth=2, label="RANSAC"
)

axes[1, 2].set_title("RANSAC: Robust to Outliers")
axes[1, 2].grid(True)
axes[1, 2].legend()

# 전체 제목 및 출력
fig.suptitle("Least Squares vs. RANSAC", fontsize=16)
plt.tight_layout()
plt.show()