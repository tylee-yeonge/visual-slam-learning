"""
Phase 2 - Week 5: 에피폴라 기하학 기초
=====================================
Essential/Fundamental 행렬, 에피폴라 선

학습 목표:
1. E와 F 행렬 이해
2. 에피폴라 제약 확인
3. 에피폴라 선 시각화
4. 8-point 알고리즘 개념

실행 시간: 약 1분
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("=" * 70)
print("        Phase 2 - Week 5: 에피폴라 기하학 기초")
print("=" * 70)
print("\n💡 이 실습에서는 두 뷰 사이의 기하학적 관계를 배웁니다.\n")

# ============================================================
# Part 1: 기본 개념 복습
# ============================================================
print("\n" + "=" * 70)
print("Part 1: 기본 개념 복습")
print("=" * 70)

print("""
🎯 에피폴라 기하학 = 두 카메라 사이의 기하학적 관계

핵심 공식:
- Essential Matrix:    x₂ᵀ E x₁ = 0  (정규화 좌표)
- Fundamental Matrix:  p₂ᵀ F p₁ = 0  (픽셀 좌표)

관계:
- E = [t]× R
- F = K₂⁻ᵀ E K₁⁻¹
""")

# 카메라 파라미터
K = np.array([
    [500,   0, 320],
    [  0, 500, 240],
    [  0,   0,   1]
], dtype=np.float64)

print("카메라 내부 파라미터 K:")
print(K)

# ============================================================
# Part 2: Essential Matrix 구성
# ============================================================
print("\n" + "=" * 70)
print("Part 2: Essential Matrix 구성")
print("=" * 70)

print("""
🎯 E = [t]× R

[t]× = skew-symmetric matrix (반대칭 행렬)
""")

def skew_symmetric(t):
    """벡터 → 반대칭 행렬"""
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])

def rotation_matrix_x(theta):
    """X축 회전"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rotation_matrix_y(theta):
    """Y축 회전"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotation_matrix_z(theta):
    """Z축 회전"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# 카메라 2의 포즈 (카메라 1 기준)
# 5도 Y축 회전 + 오른쪽으로 0.1m 이동
R = rotation_matrix_y(np.radians(5))
t = np.array([0.1, 0, 0])  # 오른쪽 이동

print(f"\n카메라 상대 포즈:")
print(f"R (5° Y축 회전):\n{R}")
print(f"t: {t}")

# Essential Matrix 계산
t_skew = skew_symmetric(t)
E = t_skew @ R

print(f"\n[t]× (skew-symmetric):\n{t_skew}")
print(f"\nEssential Matrix E = [t]× R:\n{E}")

# E의 특성 확인
U, S, Vt = np.linalg.svd(E)
print(f"\nE의 특이값: {S}")
print(f"  → 두 값이 비슷하고, 하나는 0에 가까움 (rank 2)")

# ============================================================
# Part 3: Fundamental Matrix 계산
# ============================================================
print("\n" + "=" * 70)
print("Part 3: Fundamental Matrix 계산")
print("=" * 70)

print("""
🎯 F = K⁻ᵀ E K⁻¹

E: 정규화 좌표 사용
F: 픽셀 좌표 사용
""")

K_inv = np.linalg.inv(K)
F = K_inv.T @ E @ K_inv

print(f"\nFundamental Matrix F:\n{F}")

# F를 정규화 (f₃₃ = 1 또는 ||F|| = 1)
F = F / F[2, 2]
print(f"\n정규화된 F (F[2,2]=1):\n{F}")

# ============================================================
# Part 4: 에피폴라 제약 검증
# ============================================================
print("\n" + "=" * 70)
print("Part 4: 에피폴라 제약 검증")
print("=" * 70)

print("""
🎯 에피폴라 제약: x₂ᵀ E x₁ = 0

올바른 대응점이면 이 값이 0에 가까워야!
""")

def project_point(P_3d, R, t, K):
    """3D 점을 2D 픽셀로 투영"""
    P_cam = R @ P_3d + t
    p = K @ P_cam
    return p[:2] / p[2]

def pixel_to_normalized(p, K):
    """픽셀 → 정규화 좌표"""
    K_inv = np.linalg.inv(K)
    p_h = np.array([p[0], p[1], 1])
    x = K_inv @ p_h
    return x

# 임의의 3D 점들 생성
np.random.seed(42)
points_3d = np.random.rand(10, 3) * 2 + np.array([0, 0, 5])  # 카메라 앞

# 두 카메라에서 투영
R1, t1 = np.eye(3), np.zeros(3)  # 카메라 1 (원점)
R2, t2 = R, t                      # 카메라 2

pts1 = np.array([project_point(P, R1, t1, K) for P in points_3d])
pts2 = np.array([project_point(P, R2, t2, K) for P in points_3d])

print("3D 점 → 두 이미지에 투영:")
print(f"  점 개수: {len(points_3d)}")
print(f"  이미지 1: {pts1[0]}")
print(f"  이미지 2: {pts2[0]}")

# 에피폴라 제약 확인
print("\n에피폴라 제약 p₂ᵀ F p₁ (0에 가까워야):")
for i in range(5):
    p1 = np.array([pts1[i, 0], pts1[i, 1], 1])
    p2 = np.array([pts2[i, 0], pts2[i, 1], 1])
    
    constraint = p2 @ F @ p1
    print(f"  점 {i}: {constraint:.6f}")

print("\n✅ 모든 값이 0에 매우 가까움 → 에피폴라 제약 만족!")

# ============================================================
# Part 5: 에피폴라 선
# ============================================================
print("\n" + "=" * 70)
print("Part 5: 에피폴라 선")
print("=" * 70)

print("""
🎯 에피폴라 선 = 대응점이 존재할 수 있는 선

l₂ = F · p₁   (p₁에 대응하는 l₂)
l₁ = Fᵀ · p₂  (p₂에 대응하는 l₁)
""")

def compute_epipolar_line(F, p, direction='forward'):
    """에피폴라 선 계산"""
    p_h = np.array([p[0], p[1], 1])
    if direction == 'forward':
        l = F @ p_h  # l₂ = F · p₁
    else:
        l = F.T @ p_h  # l₁ = Fᵀ · p₂
    return l / np.linalg.norm(l[:2])  # 정규화

def line_to_points(line, img_width):
    """ax + by + c = 0 → 두 점"""
    a, b, c = line
    if abs(b) > 1e-6:
        x0, x1 = 0, img_width
        y0 = -(a * x0 + c) / b
        y1 = -(a * x1 + c) / b
    else:
        x0, x1 = -c / a, -c / a
        y0, y1 = 0, 480
    return (x0, y0), (x1, y1)

# 몇 개 점에 대해 에피폴라 선 계산
print("\n에피폴라 선 예시:")
for i in range(3):
    l2 = compute_epipolar_line(F, pts1[i], 'forward')
    print(f"  p₁[{i}] = {pts1[i]} → l₂ = [{l2[0]:.4f}, {l2[1]:.4f}, {l2[2]:.4f}]")
    
    # 대응점이 선 위에 있는지 확인
    p2_h = np.array([pts2[i, 0], pts2[i, 1], 1])
    distance = abs(l2 @ p2_h) / np.linalg.norm(l2[:2])
    print(f"       p₂[{i}]와 l₂ 사이 거리: {distance:.4f} px")

# ============================================================
# Part 6: 시각화
# ============================================================
print("\n" + "=" * 70)
print("Part 6: 시각화")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

img_width, img_height = 640, 480

# 왼쪽: 이미지 1 + 에피폴라 선
ax1 = axes[0]
ax1.set_xlim([0, img_width])
ax1.set_ylim([img_height, 0])
ax1.set_title('Image 1: Points', fontsize=12)

# 점들
colors = plt.cm.tab10(np.linspace(0, 1, len(pts1)))
for i, (p, c) in enumerate(zip(pts1, colors)):
    ax1.scatter(p[0], p[1], color=c, s=100, zorder=5)
    ax1.annotate(f'{i}', (p[0]+5, p[1]-5), fontsize=9)

ax1.set_xlabel('u (pixels)')
ax1.set_ylabel('v (pixels)')
ax1.grid(True, alpha=0.3)

# 오른쪽: 이미지 2 + 에피폴라 선
ax2 = axes[1]
ax2.set_xlim([0, img_width])
ax2.set_ylim([img_height, 0])
ax2.set_title('Image 2: Points + Epipolar Lines', fontsize=12)

# 에피폴라 선 및 점
for i, (p1, p2, c) in enumerate(zip(pts1, pts2, colors)):
    # 에피폴라 선
    l2 = compute_epipolar_line(F, p1, 'forward')
    pt1, pt2 = line_to_points(l2, img_width)
    ax2.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=c, alpha=0.5, linewidth=1)
    
    # 대응점
    ax2.scatter(p2[0], p2[1], color=c, s=100, zorder=5)
    ax2.annotate(f'{i}', (p2[0]+5, p2[1]-5), fontsize=9)

ax2.set_xlabel('u (pixels)')
ax2.set_ylabel('v (pixels)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week5/epipolar_lines.png', dpi=150)
print("\nEpipolar lines saved: epipolar_lines.png")
print("  → 각 p₂ 점이 해당 에피폴라 선 위에 있음!")

# ============================================================
# Part 7: 에피폴 (Epipole)
# ============================================================
print("\n" + "=" * 70)
print("Part 7: 에피폴 계산")
print("=" * 70)

print("""
🎯 에피폴 = 다른 카메라 중심의 투영점

e₂ = F의 오른쪽 null space
e₁ = F의 왼쪽 null space (Fᵀ의 null space)
""")

# 에피폴 계산 (SVD의 마지막 열/행)
U, S, Vt = np.linalg.svd(F)
e2 = Vt[-1]  # 오른쪽 null space
e2 = e2 / e2[2]  # 정규화

U, S, Vt = np.linalg.svd(F.T)
e1 = Vt[-1]
e1 = e1 / e1[2]

print(f"\n에피폴:")
print(f"  e₁ (이미지 1): [{e1[0]:.1f}, {e1[1]:.1f}]")
print(f"  e₂ (이미지 2): [{e2[0]:.1f}, {e2[1]:.1f}]")

print("""
💡 해석:
   - 에피폴은 모든 에피폴라 선이 만나는 점
   - 에피폴이 이미지 밖에 있으면 → 에피폴라 선이 평행에 가까움
""")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 70)
print("📚 Week 5 Basics 정리")
print("=" * 70)

print("""
✅ Part 1-2: Essential Matrix
   - E = [t]× R
   - 정규화 좌표 사용
   - 5 DOF, rank 2

✅ Part 3: Fundamental Matrix
   - F = K⁻ᵀ E K⁻¹
   - 픽셀 좌표 사용
   - 7 DOF

✅ Part 4: 에피폴라 제약
   - p₂ᵀ F p₁ = 0
   - 올바른 매칭 검증

✅ Part 5-6: 에피폴라 선
   - l₂ = F·p₁
   - 대응점은 선 위에 존재

✅ Part 7: 에피폴
   - 모든 에피폴라 선의 교점
   - F의 null space

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 핵심 메시지:
   에피폴라 제약 = 두 뷰 사이의 기하학적 관계
   이를 통해 카메라 포즈 (R, t)를 추정할 수 있음!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 다음: epipolar_quiz.py → Week 6: 포즈 추정 (R, t)
""")

print("\n" + "=" * 70)
print("epipolar_basics.py 실행 완료! 🎉")
print("=" * 70)
