"""
Phase 2 - Week 5: 에피폴라 기하학 실습 문제
=========================================
8-point 알고리즘, E/F 추정, 제약 검증

학습 목표:
1. 8-point 알고리즘 구현
2. 노이즈 영향 분석
3. E/F 검증
4. RANSAC 필요성 이해

실행 시간: 약 2분
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("=" * 70)
print("       Phase 2 - Week 5: 에피폴라 기하학 실습 문제")
print("=" * 70)
print("\n이 실습에서는 Essential/Fundamental 행렬을 추정합니다.\n")

# ============================================================
# 기본 함수
# ============================================================
def skew_symmetric(t):
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])

def rotation_matrix_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def project_point(P_3d, R, t, K):
    P_cam = R @ P_3d + t
    p = K @ P_cam
    return p[:2] / p[2]

# 카메라 파라미터
K = np.array([
    [500,   0, 320],
    [  0, 500, 240],
    [  0,   0,   1]
], dtype=np.float64)

# Ground Truth 포즈
R_gt = rotation_matrix_y(np.radians(5))
t_gt = np.array([0.1, 0, 0])

E_gt = skew_symmetric(t_gt) @ R_gt
F_gt = np.linalg.inv(K).T @ E_gt @ np.linalg.inv(K)
F_gt = F_gt / F_gt[2, 2]

# ============================================================
# 문제 1: 8-point 알고리즘 구현
# ============================================================
print("\n" + "=" * 70)
print("문제 1: 8-point 알고리즘 구현")
print("=" * 70)

print("""
🎯 8-point 알고리즘:

에피폴라 제약 p₂ᵀ F p₁ = 0 를 선형화:

[u₂u₁, u₂v₁, u₂, v₂u₁, v₂v₁, v₂, u₁, v₁, 1] · f = 0

N개 점으로 행렬 방정식:
A · f = 0

SVD로 f 구함 (A의 null space)
""")

def eight_point_algorithm(pts1, pts2):
    """
    8-point 알고리즘으로 F 추정
    
    Args:
        pts1, pts2: (N, 2) 대응점 (N >= 8)
    
    Returns:
        F: (3, 3) Fundamental Matrix
    """
    n = len(pts1)
    
    # A 행렬 구성
    A = np.zeros((n, 9))
    for i in range(n):
        u1, v1 = pts1[i]
        u2, v2 = pts2[i]
        A[i] = [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]
    
    # SVD로 해 구하기
    U, S, Vt = np.linalg.svd(A)
    f = Vt[-1]  # 가장 작은 특이값에 해당하는 벡터
    
    F = f.reshape(3, 3)
    
    # Rank 2 강제 (가장 작은 특이값을 0으로)
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt
    
    # 정규화
    F = F / F[2, 2]
    
    return F

# 테스트 데이터 생성
np.random.seed(42)
points_3d = np.random.rand(20, 3) * 2 + np.array([0, 0, 5])

R1, t1 = np.eye(3), np.zeros(3)
R2, t2 = R_gt, t_gt

pts1 = np.array([project_point(P, R1, t1, K) for P in points_3d])
pts2 = np.array([project_point(P, R2, t2, K) for P in points_3d])

print(f"\n테스트 데이터: {len(pts1)}개 대응점")

# 8-point 적용
F_est = eight_point_algorithm(pts1, pts2)

print(f"\nGround Truth F:\n{F_gt}")
print(f"\n추정된 F:\n{F_est}")

# 차이 계산
diff = np.abs(F_est - F_gt)
print(f"\n차이 (절대값):\n{diff}")
print(f"최대 차이: {diff.max():.6f}")

# ============================================================
# 문제 2: Normalized 8-point
# ============================================================
print("\n" + "=" * 70)
print("문제 2: Normalized 8-point 알고리즘")
print("=" * 70)

print("""
🎯 정규화로 수치 안정성 향상:

1. 점 정규화: 중심=0, 평균 거리=√2
2. 정규화된 점으로 F 계산
3. 역정규화: F = T₂ᵀ · F_norm · T₁
""")

def normalize_points(pts):
    """점 정규화: 중심=0, 평균 거리=√2"""
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    
    avg_dist = np.mean(np.linalg.norm(centered, axis=1))
    scale = np.sqrt(2) / avg_dist
    
    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])
    
    normalized = (pts - mean) * scale
    
    return normalized, T

def normalized_eight_point(pts1, pts2):
    """Normalized 8-point 알고리즘"""
    # 정규화
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)
    
    # 8-point (정규화된 점으로)
    F_norm = eight_point_algorithm(pts1_norm, pts2_norm)
    
    # 역정규화
    F = T2.T @ F_norm @ T1
    F = F / F[2, 2]
    
    return F

F_norm_est = normalized_eight_point(pts1, pts2)

print(f"\nNormalized 8-point 결과:\n{F_norm_est}")

diff_norm = np.abs(F_norm_est - F_gt)
print(f"\nGT와 차이 (normalized):\n{diff_norm}")
print(f"최대 차이: {diff_norm.max():.6f}")

# ============================================================
# 문제 3: 노이즈 영향
# ============================================================
print("\n" + "=" * 70)
print("문제 3: 노이즈 영향 분석")
print("=" * 70)

print("""
🎯 목표: 노이즈가 F 추정에 미치는 영향

실제 특징점 검출/매칭에는 노이즈가 있음:
- 검출 오차 (~0.5 픽셀)
- 매칭 오류 (outlier)
""")

def evaluate_F(F, pts1, pts2):
    """에피폴라 제약 오차 계산"""
    errors = []
    for p1, p2 in zip(pts1, pts2):
        p1_h = np.array([p1[0], p1[1], 1])
        p2_h = np.array([p2[0], p2[1], 1])
        error = abs(p2_h @ F @ p1_h)
        errors.append(error)
    return np.mean(errors), np.max(errors)

noise_levels = [0, 0.5, 1.0, 2.0, 5.0]

print("\n노이즈 수준에 따른 추정 오차:")
print("-" * 60)
print(f"{'Noise (px)':>12} | {'Mean Error':>15} | {'Max Error':>15}")
print("-" * 60)

errors_by_noise = []

for noise in noise_levels:
    # 노이즈 추가
    pts1_noisy = pts1 + np.random.randn(*pts1.shape) * noise
    pts2_noisy = pts2 + np.random.randn(*pts2.shape) * noise
    
    # F 추정
    F_noisy = normalized_eight_point(pts1_noisy, pts2_noisy)
    
    # 오차 계산 (원본 점 기준)
    mean_err, max_err = evaluate_F(F_noisy, pts1, pts2)
    errors_by_noise.append(mean_err)
    
    print(f"{noise:>12.1f} | {mean_err:>15.6f} | {max_err:>15.6f}")

print("""
💡 관찰:
   - 노이즈 ↑ → 추정 오차 ↑
   - 실제로는 RANSAC으로 outlier 제거 필요
""")

# ============================================================
# 문제 4: 에피폴라 제약 시각화
# ============================================================
print("\n" + "=" * 70)
print("문제 4: 에피폴라 제약 시각화")
print("=" * 70)

# 에피폴라 선 그리기
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

img_width, img_height = 640, 480
colors = plt.cm.tab10(np.linspace(0, 1, len(pts1)))

def compute_epipolar_line(F, p):
    p_h = np.array([p[0], p[1], 1])
    l = F @ p_h
    return l / np.linalg.norm(l[:2])

def line_to_points(line, w):
    a, b, c = line
    if abs(b) > 1e-6:
        x0, x1 = 0, w
        y0 = -(a * x0 + c) / b
        y1 = -(a * x1 + c) / b
        return (x0, y0), (x1, y1)
    else:
        return (0, 0), (w, 0)

# 이미지 1: 점
ax1 = axes[0]
ax1.set_xlim([0, img_width])
ax1.set_ylim([img_height, 0])
ax1.set_title('Image 1: Points', fontsize=12)

for i, (p, c) in enumerate(zip(pts1, colors)):
    ax1.scatter(p[0], p[1], color=c, s=80, zorder=5)
    ax1.annotate(f'{i}', (p[0]+5, p[1]-5), fontsize=8)

ax1.grid(True, alpha=0.3)
ax1.set_xlabel('u'); ax1.set_ylabel('v')

# 이미지 2: 점 + 에피폴라 선
ax2 = axes[1]
ax2.set_xlim([0, img_width])
ax2.set_ylim([img_height, 0])
ax2.set_title('Image 2: Points + Epipolar Lines', fontsize=12)

for i, (p1, p2, c) in enumerate(zip(pts1, pts2, colors)):
    # 에피폴라 선
    l2 = compute_epipolar_line(F_gt, p1)
    pt1, pt2 = line_to_points(l2, img_width)
    ax2.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=c, alpha=0.4, linewidth=1)
    
    # 대응점
    ax2.scatter(p2[0], p2[1], color=c, s=80, zorder=5)

ax2.grid(True, alpha=0.3)
ax2.set_xlabel('u'); ax2.set_ylabel('v')

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week5/epipolar_constraint.png', dpi=150)
print("\nEpipolar constraint saved: epipolar_constraint.png")

# ============================================================
# 문제 5: Outlier 영향
# ============================================================
print("\n" + "=" * 70)
print("문제 5: Outlier 영향")
print("=" * 70)

print("""
🎯 Outlier = 잘못된 매칭

소수의 outlier도 F 추정을 크게 왜곡!
→ RANSAC 필수
""")

# Outlier 추가
n_outliers = 5
n_total = len(pts1) + n_outliers

pts1_with_outliers = np.vstack([
    pts1,
    np.random.rand(n_outliers, 2) * np.array([img_width, img_height])
])
pts2_with_outliers = np.vstack([
    pts2,
    np.random.rand(n_outliers, 2) * np.array([img_width, img_height])
])

# Outlier 포함 추정
F_with_outliers = normalized_eight_point(pts1_with_outliers, pts2_with_outliers)

# 비교
mean_err_clean, _ = evaluate_F(F_gt, pts1, pts2)
mean_err_outlier, _ = evaluate_F(F_with_outliers, pts1, pts2)

print(f"\n결과 비교 (원본 점 기준):")
print(f"  Clean F 오차:        {mean_err_clean:.6f}")
print(f"  Outlier 포함 F 오차: {mean_err_outlier:.6f}")
print(f"  → {mean_err_outlier / max(mean_err_clean, 1e-10):.1f}배 악화!")

print("""
💡 RANSAC 사용법 (OpenCV):

```python
F, mask = cv2.findFundamentalMat(
    pts1, pts2,
    method=cv2.FM_RANSAC,
    ransacReprojThreshold=3.0
)
# mask[i] = 1: inlier
```
""")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 70)
print("📚 Week 5 Quiz 정리")
print("=" * 70)

print("""
✅ 문제 1: 8-point 알고리즘
   - 선형 방정식 Af=0
   - SVD로 해 구함
   - Rank 2 강제

✅ 문제 2: Normalized 8-point
   - 정규화로 수치 안정성 ↑
   - 중심=0, 평균거리=√2

✅ 문제 3: 노이즈 영향
   - 노이즈 ↑ → 오차 ↑
   - 실제 데이터는 항상 노이즈 존재

✅ 문제 4: 에피폴라 제약
   - 대응점은 에피폴라 선 위
   - 시각화로 검증

✅ 문제 5: Outlier 영향
   - 소수 outlier도 치명적
   - RANSAC 필수!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 실전 가이드:

1. 항상 RANSAC 사용 (cv2.FM_RANSAC)
2. 최소 8점 이상 필요
3. 점 분포가 다양해야 (한 곳에 몰리면 불안정)
4. threshold: 1~3 픽셀 권장

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 다음: Week 6 - 포즈 추정 (E → R, t 분해)
""")

print("\n" + "=" * 70)
print("epipolar_quiz.py 실행 완료! 🎉")
print("=" * 70)
print("\n생성된 파일:")
print("  1. epipolar_lines.png - 에피폴라 선 시각화")
print("  2. epipolar_constraint.png - 에피폴라 제약 시각화")
