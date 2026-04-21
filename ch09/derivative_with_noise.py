import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1. 1차원 signal 만들기
x = np.arange(0, 2000)

signal = np.zeros_like(x, dtype=np.float32)
signal[:1000] = 30
for i in range(965, 1035):
    signal[i] = 70 + i - 1000
signal[1035:] = 100

# 노이즈 추가
noise = np.random.normal(0, 2, size=x.shape)
noisy_signal = signal + noise

# 2. 그냥 미분 (엣지 검출)
edge_raw = np.diff(noisy_signal, prepend=noisy_signal[0])

# 3. Gaussian smoothing 후 미분
noisy_signal_2d = noisy_signal.reshape(1, -1)
smoothed_signal = cv2.GaussianBlur(noisy_signal.reshape(1, -1), (0, 0), sigmaX=15).flatten()

edge_smooth = np.diff(smoothed_signal, prepend=smoothed_signal[0])

# -----------------------------
# 4. 출력
# -----------------------------
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(x, noisy_signal, color='black', linewidth=1)
plt.title("Noisy Signal")
plt.ylabel("f(x)")
plt.xlim(0, 2000)
plt.grid(True, linestyle=':')

plt.subplot(3, 1, 2)
plt.plot(x, edge_raw, color='black', linewidth=0.7)
plt.title("Edge by Direct Derivative")
plt.ylabel("df/dx")
plt.xlim(0, 2000)
plt.grid(True, linestyle=':')

plt.subplot(3, 1, 3)
plt.plot(x, edge_smooth, color='black', linewidth=1)
plt.title("Edge after Gaussian Smoothing")
plt.ylabel("df/dx")
plt.xlabel("x")
plt.xlim(0, 2000)
plt.grid(True, linestyle=':')

plt.tight_layout()
plt.show()