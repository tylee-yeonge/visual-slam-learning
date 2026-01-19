"""
Phase 2 - Week 6: 포즈 추정 실습 문제
====================================
4가지 해 분석, 스케일 모호성, 회전 검증

학습 목표:
1. 4가지 해 비교
2. 스케일 모호성 이해
3. 회전 행렬 검증
4. 포즈 추정 실패 케이스

실행 시간: 약 2분
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("=" * 70)
print("       Phase 2 - Week 6: 포즈 추정 실습 문제")
print("=" * 70)
print("\n이 실습에서는 E → R, t 분해의 다양한 측면을 탐구합니다.\n")

# ============================================================
# 기본 함수
# ============================================================
def skew_symmetric(t):
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])

def rotation_matrix(axis, theta):
    """축-각도 회전 행렬"""
    if axis == 'x':
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:  # z
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def decompose_essential(E):
    """E에서 4가지 (R, t) 해 추출"""
    U, S, Vt = np.linalg.svd(E)
    
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    R1 = U @ W.T @ Vt
    R2 = U @ W @ Vt
    t1 = U[:, 2]
    t2 = -U[:, 2]
    
    # det = 1 보장
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    
    return [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]

K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)

# ============================================================
# 문제 1: 4가지 해 분석
# ============================================================
print("\n" + "=" * 70)
print("문제 1: 4가지 해 기하학적 의미")
print("=" * 70)

print("""
🎯 4가지 해는 무엇을 의미하는가?

1. (R₁, +t): 정상 해
2. (R₁, -t): t 반전 (카메라 반대 방향)
3. (R₂, +t): 180° 회전된 해
4. (R₂, -t): 둘 다 반전

→ Cheirality Check로 구분!
""")

# GT 포즈
R_gt = rotation_matrix('y', np.radians(15))
t_gt = np.array([0.3, 0.1, 0.05])
t_gt = t_gt / np.linalg.norm(t_gt)

E = skew_symmetric(t_gt) @ R_gt
solutions = decompose_essential(E)

print("4가지 해 분석:")
print("-" * 60)

for i, (R, t) in enumerate(solutions):
    # R 분석
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    
    # t 방향 분석
    t_dir = t / np.linalg.norm(t)
    t_gt_dir = t_gt / np.linalg.norm(t_gt)
    t_dot = np.dot(t_dir, t_gt_dir)
    
    print(f"\n[해 {i+1}]")
    print(f"  회전 각도: {np.degrees(angle):.1f}°")
    print(f"  t 방향 cos: {t_dot:.4f}")
    print(f"  t와 GT 정렬: {'✅ 같음' if t_dot > 0.9 else '❌ 다름' if t_dot < -0.9 else '⚠️ 다른 방향'}")

# ============================================================
# 문제 2: 스케일 모호성
# ============================================================
print("\n" + "=" * 70)
print("문제 2: 스케일 모호성 분석")
print("=" * 70)

print("""
🎯 E에서 t의 스케일(크기)은 복원 불가!

증명: t와 λt는 같은 E를 생성
E = [t]× R = λ[t/λ]× R (스케일만 다름)
""")

# 다양한 스케일로 E 생성
scales = [0.1, 0.5, 1.0, 2.0, 5.0]
t_base = np.array([1, 0, 0])

print("\n다양한 스케일의 t:")
print("-" * 50)

E_matrices = []
for scale in scales:
    t_scaled = t_base * scale
    E_scaled = skew_symmetric(t_scaled) @ R_gt
    
    # 정규화된 E
    E_norm = E_scaled / np.linalg.norm(E_scaled)
    E_matrices.append(E_norm)
    
    print(f"  |t| = {scale:.1f}: E[0,0] = {E_scaled[0,0]:.4f}, ||E|| = {np.linalg.norm(E_scaled):.4f}")

# 정규화 후 비교
print("\n정규화 후 E 비교:")
for i in range(1, len(scales)):
    diff = np.linalg.norm(E_matrices[i] - E_matrices[0])
    print(f"  E(scale={scales[i]}) vs E(scale={scales[0]}): diff = {diff:.6f}")

print("""
💡 결론:
   정규화된 E는 모두 동일!
   → t의 방향만 알 수 있고, 크기는 알 수 없음
   → Monocular SLAM의 근본적 한계
""")

# ============================================================
# 문제 3: 회전 행렬 검증
# ============================================================
print("\n" + "=" * 70)
print("문제 3: 회전 행렬 검증")
print("=" * 70)

print("""
🎯 올바른 회전 행렬 조건:
1. 직교성: RᵀR = I
2. 행렬식: det(R) = 1 (반사 아님)
""")

def validate_rotation(R, name="R"):
    """회전 행렬 검증"""
    # 직교성
    RtR = R.T @ R
    ortho_error = np.linalg.norm(RtR - np.eye(3))
    
    # 행렬식
    det = np.linalg.det(R)
    
    is_valid = ortho_error < 1e-6 and np.isclose(det, 1.0)
    
    print(f"\n{name}:")
    print(f"  RᵀR - I: {ortho_error:.6f}")
    print(f"  det(R) = {det:.6f}")
    print(f"  Valid: {'✅' if is_valid else '❌'}")
    
    return is_valid

# 정상 회전
print("정상 회전 행렬:")
validate_rotation(R_gt, "R_gt")

# 노이즈로 손상된 회전
R_noisy = R_gt + np.random.randn(3, 3) * 0.01
print("\n노이즈 추가된 행렬:")
validate_rotation(R_noisy, "R_noisy")

# SVD로 복구
U, S, Vt = np.linalg.svd(R_noisy)
R_fixed = U @ Vt
if np.linalg.det(R_fixed) < 0:
    R_fixed = U @ np.diag([1, 1, -1]) @ Vt

print("\nSVD로 복구:")
validate_rotation(R_fixed, "R_fixed")

# ============================================================
# 문제 4: 실패 케이스
# ============================================================
print("\n" + "=" * 70)
print("문제 4: 포즈 추정 실패 케이스")
print("=" * 70)

print("""
🎯 포즈 추정이 실패하는 경우:

1. 순수 회전 (t ≈ 0)
2. 공면 점들 (co-planar)
3. 노이즈/outlier
""")

# Case 1: 순수 회전
print("\n[Case 1] 순수 회전 (t = 0):")
R_pure = rotation_matrix('z', np.radians(30))
t_pure = np.array([0, 0, 0])

try:
    E_pure = skew_symmetric(t_pure) @ R_pure
    print(f"  E = \n{E_pure}")
    print(f"  E ≈ 0? {np.allclose(E_pure, 0)}")
    print("  → E가 0이면 분해 불가!")
except:
    print("  → E 정의 불가")

# Case 2: 아주 작은 이동
print("\n[Case 2] 아주 작은 이동 (t ≈ 0):")
t_small = np.array([1e-6, 0, 0])
E_small = skew_symmetric(t_small) @ R_gt
U, S_small, Vt = np.linalg.svd(E_small)
print(f"  특이값: {S_small}")
print(f"  → 특이값이 너무 작으면 수치적으로 불안정")

# ============================================================
# 문제 5: 시각화
# ============================================================
print("\n" + "=" * 70)
print("문제 5: 4가지 해 시각화")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 각 해에 대한 카메라 방향 시각화
titles = ['Solution 1 (R1, +t)', 'Solution 2 (R1, -t)', 
          'Solution 3 (R2, +t)', 'Solution 4 (R2, -t)']

for idx, ((R, t), ax) in enumerate(zip(solutions, axes.flatten())):
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    
    # 카메라 1 (원점)
    ax.scatter([0], [0], c='blue', s=100, marker='^', label='Cam 1')
    ax.arrow(0, 0, 0.5, 0, head_width=0.1, head_length=0.05, fc='blue', ec='blue')
    
    # 카메라 2
    cam2_pos = -R.T @ t
    ax.scatter([cam2_pos[0]], [cam2_pos[2]], c='red', s=100, marker='^', label='Cam 2')
    
    # 카메라 2 방향
    z_dir = R[2, :2] # XZ 평면에서 Z 방향
    ax.arrow(cam2_pos[0], cam2_pos[2], z_dir[0]*0.5, z_dir[1]*0.5,
             head_width=0.1, head_length=0.05, fc='red', ec='red')
    
    # 3D 점 (예시)
    ax.scatter([0, 0.5, -0.3], [3, 4, 2.5], c='green', s=50, marker='o', alpha=0.5, label='3D Points')
    
    ax.set_title(titles[idx], fontsize=11)
    ax.set_xlabel('X'); ax.set_ylabel('Z')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week6/four_solutions.png', dpi=150)
print("\nFour solutions saved: four_solutions.png")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 70)
print("📚 Week 6 Quiz 정리")
print("=" * 70)

print("""
✅ 문제 1: 4가지 해
   - R₁/R₂: 180° 차이
   - ±t: 방향 반대
   - Cheirality로 구분

✅ 문제 2: 스케일 모호성
   - t의 크기는 복원 불가
   - 방향만 알 수 있음
   - IMU 융합으로 해결 (VINS)

✅ 문제 3: 회전 검증
   - RᵀR = I, det(R) = 1
   - SVD로 복구 가능

✅ 문제 4: 실패 케이스
   - 순수 회전: E ≈ 0
   - 작은 이동: 수치 불안정

✅ 문제 5: 시각화
   - 4가지 해의 기하학 확인

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 OpenCV 사용법:

```python
import cv2

# E 계산
E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC)

# R, t 복원 (자동 Cheirality Check)
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 다음: Week 7 - 삼각측량과 PnP
""")

print("\n" + "=" * 70)
print("pose_estimation_quiz.py 실행 완료! 🎉")
print("=" * 70)
print("\n생성된 파일:")
print("  1. pose_estimation.png - 포즈 추정 결과")
print("  2. four_solutions.png - 4가지 해 시각화")
