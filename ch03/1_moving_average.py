import cv2
import numpy as np
import matplotlib.pyplot as plt

F = np.zeros((10, 10), dtype=np.float32)
F[2:7, 3:8] = 90    # 가운데 90 영역
F[5, 4] = 0     # 구멍 하나 만들기
F[8, 2] = 90    # 아래쪽 작은 점 하나

kernel = np.ones((3, 3), dtype=np.float32) / 9.0

G = cv2.filter2D(F, -1, kernel, borderType=cv2.BORDER_CONSTANT)
G_round = np.round(G).astype(int)

print("입력 영상 F[x,y]")
print(F.astype(int))
print()

print("출력 영상 G[x,y]")
print(G_round)

# ---------------------------------
# 5. 숫자를 이미지 위에 표시하는 함수
# ---------------------------------
def draw_matrix(ax, data, title, cmap='gray', vmin=0, vmax=90):
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

    rows, cols = data.shape

    # 격자선
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    # 눈금 숨기기
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)

    # 숫자 표시
    for i in range(rows):
        for j in range(cols):
            value = int(round(data[i, j]))
            color = 'white' if data[i, j] < 45 else 'black'
            ax.text(j, i, str(value), ha='center', va='center',
                    fontsize=9, color=color)

    ax.set_title(title, fontsize=16)

# ---------------------------------
# 6. 결과 시각화
# ---------------------------------
plt.figure(figsize=(10, 5))

ax1 = plt.subplot(1, 2, 1)
draw_matrix(ax1, F, "F[x, y]")

ax2 = plt.subplot(1, 2, 2)
draw_matrix(ax2, G_round, "G[x, y]")

plt.tight_layout()
plt.show()