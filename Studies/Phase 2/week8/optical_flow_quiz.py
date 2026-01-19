"""
Phase 2 - Week 8: 광류 실습 문제
================================
피라미드 LK, 파라미터 분석, Dense Flow

학습 목표:
1. 피라미드 LK 이해
2. 파라미터 영향 분석
3. Dense Flow 비교
4. VINS 파라미터 연결

실행 시간: 약 2분
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, zoom

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("=" * 70)
print("       Phase 2 - Week 8: 광류 실습 문제")
print("=" * 70)
print("\n이 실습에서는 광류의 다양한 측면을 탐구합니다.\n")

# ============================================================
# 기본 함수
# ============================================================
def create_moving_scene(size=150, shift=(15, 8)):
    """이동하는 장면 생성"""
    # 프레임 1: 여러 도형
    img1 = np.zeros((size, size), dtype=np.float32)
    
    # 사각형
    img1[30:60, 30:70] = 200
    
    # 원
    y, x = np.ogrid[:size, :size]
    circle = (x - 100)**2 + (y - 100)**2 <= 20**2
    img1[circle] = 180
    
    # 삼각형
    for i in range(25):
        img1[80+i, 40-i:40+i+1] = 160
    
    # 프레임 2: 전체 이동
    img2 = np.zeros((size, size), dtype=np.float32)
    dx, dy = shift
    
    # 간단한 이동 (wraparound 없이)
    src_y1 = max(0, -dy)
    src_y2 = min(size, size - dy)
    src_x1 = max(0, -dx)
    src_x2 = min(size, size - dx)
    
    dst_y1 = max(0, dy)
    dst_y2 = min(size, size + dy)
    dst_x1 = max(0, dx)
    dst_x2 = min(size, size + dx)
    
    img2[dst_y1:dst_y2, dst_x1:dst_x2] = img1[src_y1:src_y2, src_x1:src_x2]
    
    return img1, img2, shift

def compute_gradients(img1, img2):
    """그래디언트 계산"""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8
    
    Ix = (convolve(img1, kx) + convolve(img2, kx)) / 2
    Iy = (convolve(img1, ky) + convolve(img2, ky)) / 2
    It = img2.astype(np.float32) - img1.astype(np.float32)
    
    return Ix, Iy, It

def lk_flow(Ix, Iy, It, point, window_size=21):
    """Lucas-Kanade 광류"""
    x, y = int(point[0]), int(point[1])
    half_w = window_size // 2
    
    y_min = max(0, y - half_w)
    y_max = min(Ix.shape[0], y + half_w + 1)
    x_min = max(0, x - half_w)
    x_max = min(Ix.shape[1], x + half_w + 1)
    
    Ix_win = Ix[y_min:y_max, x_min:x_max].flatten()
    Iy_win = Iy[y_min:y_max, x_min:x_max].flatten()
    It_win = It[y_min:y_max, x_min:x_max].flatten()
    
    A = np.column_stack([Ix_win, Iy_win])
    AtA = A.T @ A
    Atb = -A.T @ It_win
    
    try:
        if np.linalg.det(AtA) < 1e-6:
            return (0, 0), False
        flow = np.linalg.solve(AtA, Atb)
        return (flow[0], flow[1]), True
    except:
        return (0, 0), False

# ============================================================
# 문제 1: 큰 움직임 문제
# ============================================================
print("\n" + "=" * 70)
print("문제 1: 큰 움직임 문제")
print("=" * 70)

print("""
🎯 Lucas-Kanade의 한계:

테일러 전개는 작은 Δx, Δy 가정
→ 큰 움직임에서 실패!

해결: 피라미드 (다중 스케일)
""")

# 다양한 이동량 테스트
shifts = [(5, 3), (10, 5), (15, 8), (25, 12)]

print("\n이동량에 따른 추적 성능:")
print("-" * 50)
print(f"{'Shift':>15} | {'Estimated':>15} | {'Error':>10}")
print("-" * 50)

for shift in shifts:
    img1, img2, true_shift = create_moving_scene(150, shift)
    Ix, Iy, It = compute_gradients(img1, img2)
    
    # 중심점에서 추적
    point = (50, 45)  # 사각형 중심
    (u, v), success = lk_flow(Ix, Iy, It, point, window_size=31)
    
    error = np.sqrt((u - true_shift[0])**2 + (v - true_shift[1])**2)
    print(f"({shift[0]:3d}, {shift[1]:3d}) | ({u:6.1f}, {v:5.1f}) | {error:10.2f}")

print("""
💡 관찰:
   큰 이동량 → 오차 증가
   해결: 피라미드 LK 필요
""")

# ============================================================
# 문제 2: 피라미드 LK
# ============================================================
print("\n" + "=" * 70)
print("문제 2: 피라미드 LK 개념")
print("=" * 70)

print("""
🎯 피라미드 LK:

Level 2 (가장 작음): 움직임 5px → 추적 가능
       │
       ▼ 확대 + 정제
Level 1: 움직임 10px → 이전 레벨 결과 + 정제
       │
       ▼ 확대 + 정제
Level 0 (원본): 움직임 20px → 최종 결과
""")

def build_pyramid(img, levels=3):
    """이미지 피라미드 생성"""
    pyramid = [img]
    for _ in range(levels - 1):
        # 축소 (0.5배)
        downsampled = zoom(pyramid[-1], 0.5, order=1)
        pyramid.append(downsampled)
    return pyramid

def pyramid_lk(img1, img2, point, levels=3, window_size=21):
    """
    피라미드 Lucas-Kanade (간단 버전)
    """
    # 피라미드 생성
    pyr1 = build_pyramid(img1, levels)
    pyr2 = build_pyramid(img2, levels)
    
    # 최상위 레벨(가장 작은)에서 시작
    scale = 2 ** (levels - 1)
    pt = (point[0] / scale, point[1] / scale)
    
    total_u, total_v = 0, 0
    
    # 상위에서 하위로
    for level in range(levels - 1, -1, -1):
        Ix, Iy, It = compute_gradients(pyr1[level], pyr2[level])
        
        # 현재 레벨에서 광류
        (u, v), success = lk_flow(Ix, Iy, It, pt, window_size)
        
        total_u += u
        total_v += v
        
        # 다음 레벨로 (2배 확대)
        if level > 0:
            pt = (pt[0] * 2 + total_u, pt[1] * 2 + total_v)
            total_u *= 2
            total_v *= 2
    
    return (total_u, total_v)

# 피라미드 vs 일반 LK 비교
shift_large = (25, 12)
img1, img2, _ = create_moving_scene(150, shift_large)

point = (50, 45)

# 일반 LK
Ix, Iy, It = compute_gradients(img1, img2)
(u_simple, v_simple), _ = lk_flow(Ix, Iy, It, point, window_size=31)

# 피라미드 LK
(u_pyr, v_pyr) = pyramid_lk(img1, img2, point, levels=3, window_size=21)

print(f"\n큰 이동량 ({shift_large}) 비교:")
print(f"  일반 LK:    ({u_simple:.1f}, {v_simple:.1f})")
print(f"  피라미드 LK: ({u_pyr:.1f}, {v_pyr:.1f})")
print(f"  실제:       ({shift_large[0]}, {shift_large[1]})")

# ============================================================
# 문제 3: 윈도우 크기 영향
# ============================================================
print("\n" + "=" * 70)
print("문제 3: 윈도우 크기 영향")
print("=" * 70)

print("""
🎯 윈도우 크기:

작은 윈도우: 정밀, 노이즈에 민감, 작은 물체
큰 윈도우: 안정적, 뭉뚱그려짐, 균일 영역 도움
""")

window_sizes = [11, 21, 31, 41, 51]
shift_test = (10, 5)
img1, img2, _ = create_moving_scene(150, shift_test)

print(f"\n윈도우 크기에 따른 결과 (실제 이동: {shift_test}):")
print("-" * 50)
print(f"{'Window':>10} | {'Estimated':>15} | {'Error':>10}")
print("-" * 50)

errors_by_window = []

for ws in window_sizes:
    Ix, Iy, It = compute_gradients(img1, img2)
    (u, v), success = lk_flow(Ix, Iy, It, (50, 45), window_size=ws)
    
    error = np.sqrt((u - shift_test[0])**2 + (v - shift_test[1])**2)
    errors_by_window.append(error)
    
    print(f"{ws:>10} | ({u:6.1f}, {v:5.1f}) | {error:10.2f}")

# ============================================================
# 문제 4: 추적 실패 감지
# ============================================================
print("\n" + "=" * 70)
print("문제 4: 추적 실패 감지")
print("=" * 70)

print("""
🎯 추적 실패 원인:
1. 플랫 영역 (그래디언트 없음)
2. 에지 (조리개 문제)
3. 폐색
4. 너무 큰 움직임

감지 방법:
- Structure Tensor 고유값 확인
- 에러 임계값 (OpenCV status/error)
""")

def check_trackability(Ix, Iy, point, window_size=21, threshold=100):
    """추적 가능성 체크"""
    x, y = int(point[0]), int(point[1])
    half_w = window_size // 2
    
    y_min = max(0, y - half_w)
    y_max = min(Ix.shape[0], y + half_w + 1)
    x_min = max(0, x - half_w)
    x_max = min(Ix.shape[1], x + half_w + 1)
    
    Ix_win = Ix[y_min:y_max, x_min:x_max].flatten()
    Iy_win = Iy[y_min:y_max, x_min:x_max].flatten()
    
    M = np.array([
        [np.sum(Ix_win**2), np.sum(Ix_win * Iy_win)],
        [np.sum(Ix_win * Iy_win), np.sum(Iy_win**2)]
    ])
    
    eigvals = np.linalg.eigvalsh(M)
    min_eig = np.min(eigvals)
    
    return min_eig > threshold, min_eig

# 다양한 위치에서 테스트
test_points = [
    ((50, 45), "Rectangle center"),
    ((35, 45), "Rectangle edge"),
    ((15, 15), "Background (flat)"),
    ((100, 100), "Circle center"),
]

print("\n추적 가능성 분석:")
print("-" * 60)

Ix, Iy, It = compute_gradients(img1, img2)

for (x, y), desc in test_points:
    trackable, min_eig = check_trackability(Ix, Iy, (x, y))
    status = "✅ Trackable" if trackable else "❌ Not trackable"
    print(f"  ({x:3d}, {y:3d}) {desc:20s}: λ_min={min_eig:8.1f} {status}")

# ============================================================
# 문제 5: 시각화
# ============================================================
print("\n" + "=" * 70)
print("문제 5: 시각화")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 프레임 비교
ax1 = axes[0, 0]
ax1.imshow(img1, cmap='gray')
ax1.set_title('Frame 1', fontsize=11)
ax1.axis('off')

ax2 = axes[0, 1]
ax2.imshow(img2, cmap='gray')
ax2.set_title(f'Frame 2 (shifted by {shift_test})', fontsize=11)
ax2.axis('off')

# 윈도우 크기 vs 오차
ax3 = axes[0, 2]
ax3.plot(window_sizes, errors_by_window, 'bo-', linewidth=2, markersize=8)
ax3.set_xlabel('Window Size', fontsize=11)
ax3.set_ylabel('Error', fontsize=11)
ax3.set_title('Window Size vs Error', fontsize=11)
ax3.grid(True, alpha=0.3)

# 추적 품질 맵
ax4 = axes[1, 0]
quality_map = np.zeros_like(img1)
for y in range(5, img1.shape[0]-5, 10):
    for x in range(5, img1.shape[1]-5, 10):
        _, min_eig = check_trackability(Ix, Iy, (x, y), window_size=15, threshold=0)
        quality_map[y-5:y+5, x-5:x+5] = min_eig

ax4.imshow(quality_map, cmap='hot')
ax4.set_title('Trackability Map (min eigenvalue)', fontsize=11)
ax4.axis('off')

# 광류 필드
ax5 = axes[1, 1]
ax5.imshow(img1, cmap='gray', alpha=0.7)

for y in range(20, img1.shape[0]-20, 15):
    for x in range(20, img1.shape[1]-20, 15):
        trackable, _ = check_trackability(Ix, Iy, (x, y), threshold=100)
        if trackable:
            (u, v), success = lk_flow(Ix, Iy, It, (x, y), window_size=21)
            if success and (abs(u) > 1 or abs(v) > 1):
                ax5.arrow(x, y, u*2, v*2, head_width=2, head_length=1, 
                         fc='red', ec='red', alpha=0.7)

ax5.set_title('Sparse Optical Flow', fontsize=11)
ax5.axis('off')

# VINS 파라미터 가이드
ax6 = axes[1, 2]
ax6.axis('off')
vins_text = """
VINS-Fusion feature_tracker 파라미터:

max_cnt: 150
  → 최대 특징점 수

min_dist: 30
  → 특징점 간 최소 거리

show_track: 1
  → 추적 시각화

flow_back: 1
  → 역방향 검증

OpenCV calcOpticalFlowPyrLK:
  winSize: (21, 21)
  maxLevel: 3
"""
ax6.text(0.1, 0.5, vins_text, fontsize=10, family='monospace',
         verticalalignment='center', transform=ax6.transAxes)
ax6.set_title('VINS Parameters', fontsize=11)

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week8/optical_flow_quiz.png', dpi=150)
print("\nOptical flow quiz saved: optical_flow_quiz.png")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 70)
print("📚 Week 8 Quiz 정리")
print("=" * 70)

print("""
✅ 문제 1: 큰 움직임 문제
   - 테일러 전개 → 작은 움직임 가정
   - 큰 이동 → 오차 증가

✅ 문제 2: 피라미드 LK
   - 다중 스케일로 큰 움직임 처리
   - 상위(작은)에서 하위(큰)로

✅ 문제 3: 윈도우 크기
   - 작음: 정밀, 민감
   - 큼: 안정, 뭉뚱그려짐

✅ 문제 4: 추적 실패
   - Structure Tensor 고유값 확인
   - 플랫/에지에서 실패

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 OpenCV 사용법:

```python
# 특징점 검출
pts = cv2.goodFeaturesToTrack(gray, 200, 0.01, 30)

# 피라미드 LK 추적
next_pts, status, err = cv2.calcOpticalFlowPyrLK(
    prev_gray, cur_gray, prev_pts, None,
    winSize=(21, 21), maxLevel=3
)

# 성공한 점만
good_new = next_pts[status == 1]
good_old = prev_pts[status == 1]
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎉 Phase 2 완료!
   8주간 컴퓨터 비전 기초를 배웠습니다!
   다음: Phase 3 - 비선형 최적화
""")

print("\n" + "=" * 70)
print("optical_flow_quiz.py 실행 완료! 🎉")
print("=" * 70)
print("\n생성된 파일:")
print("  1. optical_flow_basics.png - 광류 기초")
print("  2. optical_flow_quiz.png - 광류 분석")
