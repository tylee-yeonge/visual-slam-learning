"""
Phase 2 - Week 6: 포즈 추정 기초
================================
E → R, t 분해, Cheirality Check

학습 목표:
1. SVD로 E 분해
2. 4가지 해 도출
3. Cheirality Check
4. 올바른 포즈 선택

실행 시간: 약 1분
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("=" * 70)
print("        Phase 2 - Week 6: 포즈 추정 기초")
print("=" * 70)
print("\n💡 이 실습에서는 E에서 R, t를 분해합니다.\n")

# ============================================================
# Part 1: 기본 설정
# ============================================================
print("\n" + "=" * 70)
print("Part 1: 기본 설정")
print("=" * 70)

def skew_symmetric(t):
    """벡터 → 반대칭 행렬"""
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])

def rotation_matrix_y(theta):
    """Y축 회전"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

# 카메라 파라미터
K = np.array([
    [500,   0, 320],
    [  0, 500, 240],
    [  0,   0,   1]
], dtype=np.float64)

# Ground Truth 포즈
R_gt = rotation_matrix_y(np.radians(10))  # 10도 회전
t_gt = np.array([0.5, 0.1, 0.2])           # 이동
t_gt = t_gt / np.linalg.norm(t_gt)         # 단위 벡터

print("Ground Truth 포즈:")
print(f"R:\n{R_gt}")
print(f"t: {t_gt}")

# Essential Matrix
E_gt = skew_symmetric(t_gt) @ R_gt
print(f"\nEssential Matrix E:\n{E_gt}")

# ============================================================
# Part 2: SVD 분해
# ============================================================
print("\n" + "=" * 70)
print("Part 2: E의 SVD 분해")
print("=" * 70)

print("""
🎯 E = U · Σ · Vᵀ

E의 특성:
- 두 특이값이 같음 (σ, σ, 0)
- rank = 2
""")

U, S, Vt = np.linalg.svd(E_gt)

print(f"U:\n{U}")
print(f"\n특이값: {S}")
print(f"\nVᵀ:\n{Vt}")

# 특이값 확인
print(f"\n✅ 첫 두 특이값 비슷? {np.isclose(S[0], S[1], rtol=0.1)}")
print(f"✅ 세 번째 특이값 ≈ 0? {np.isclose(S[2], 0, atol=1e-10)}")

# ============================================================
# Part 3: R, t 분해 (4가지 해)
# ============================================================
print("\n" + "=" * 70)
print("Part 3: 4가지 (R, t) 해 도출")
print("=" * 70)

print("""
🎯 W 행렬 사용:

W = [0 -1 0; 1 0 0; 0 0 1]

R₁ = U · Wᵀ · Vᵀ,  R₂ = U · W · Vᵀ
t₁ = +U₃,          t₂ = -U₃
""")

# W 행렬 (90도 Z축 회전)
W = np.array([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]
])

# 4가지 해 계산
R1 = U @ W.T @ Vt
R2 = U @ W @ Vt
t1 = U[:, 2]   # U의 3번째 열
t2 = -U[:, 2]

# 회전행렬 검증 (det = 1)
def fix_rotation(R):
    """det(R) = -1이면 부호 수정"""
    if np.linalg.det(R) < 0:
        return -R
    return R

R1 = fix_rotation(R1)
R2 = fix_rotation(R2)

solutions = [
    (R1, t1, "R1, +t"),
    (R1, t2, "R1, -t"),
    (R2, t1, "R2, +t"),
    (R2, t2, "R2, -t"),
]

print("\n4가지 (R, t) 해:")
for i, (R, t, name) in enumerate(solutions):
    det_R = np.linalg.det(R)
    print(f"\n[해 {i+1}] {name}")
    print(f"  det(R) = {det_R:.4f}")
    print(f"  t = {t}")

# ============================================================
# Part 4: Cheirality Check
# ============================================================
print("\n" + "=" * 70)
print("Part 4: Cheirality Check")
print("=" * 70)

print("""
🎯 3D 점이 두 카메라 앞에 있어야 (Z > 0)

삼각측량 후 Z 좌표 확인!
""")

def triangulate_point(P1, P2, p1, p2):
    """DLT 삼각측량 (간단 버전)"""
    A = np.array([
        p1[0] * P1[2] - P1[0],
        p1[1] * P1[2] - P1[1],
        p2[0] * P2[2] - P2[0],
        p2[1] * P2[2] - P2[1]
    ])
    
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def project_point(P_3d, R, t, K):
    """3D → 2D 투영"""
    P_cam = R @ P_3d + t
    p = K @ P_cam
    return p[:2] / p[2]

def cheirality_check(R, t, pts1, pts2, K):
    """
    Cheirality Check: Z > 0인 점 비율 반환
    """
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    
    count = 0
    for p1, p2 in zip(pts1, pts2):
        # 삼각측량
        X = triangulate_point(P1, P2, np.append(p1, 1), np.append(p2, 1))
        
        # 카메라 1에서 Z > 0?
        if X[2] > 0:
            # 카메라 2에서 Z > 0?
            X_cam2 = R @ X + t
            if X_cam2[2] > 0:
                count += 1
    
    return count / len(pts1)

# 테스트 점 생성
np.random.seed(42)
points_3d = np.random.rand(20, 3) * 2 + np.array([0, 0, 5])

R1_cam, t1_cam = np.eye(3), np.zeros(3)
R2_cam, t2_cam = R_gt, t_gt

pts1 = np.array([project_point(P, R1_cam, t1_cam, K) for P in points_3d])
pts2 = np.array([project_point(P, R2_cam, t2_cam, K) for P in points_3d])

# 각 해에 대해 Cheirality Check
print("\nCheirality Check 결과:")
print("-" * 50)

best_solution = None
best_ratio = 0

for i, (R, t, name) in enumerate(solutions):
    ratio = cheirality_check(R, t, pts1, pts2, K)
    status = "✅ BEST" if ratio > 0.9 else "❌"
    print(f"[해 {i+1}] {name}: {ratio*100:.1f}% Z>0 {status}")
    
    if ratio > best_ratio:
        best_ratio = ratio
        best_solution = (R, t, name)

print(f"\n🎯 선택된 해: {best_solution[2]}")

# ============================================================
# Part 5: Ground Truth와 비교
# ============================================================
print("\n" + "=" * 70)
print("Part 5: Ground Truth와 비교")
print("=" * 70)

R_est, t_est, _ = best_solution

print("추정된 R vs Ground Truth R:")
print(f"  추정: \n{R_est}")
print(f"  GT:   \n{R_gt}")
print(f"  차이: {np.linalg.norm(R_est - R_gt):.6f}")

print("\n추정된 t vs Ground Truth t:")
print(f"  추정: {t_est}")
print(f"  GT:   {t_gt}")
# t는 방향만 비교 (부호가 다를 수 있음)
dot = abs(np.dot(t_est, t_gt))
print(f"  방향 유사도 (|cos|): {dot:.6f}")

if dot > 0.99:
    print("  ✅ 방향 일치!")
else:
    print("  ⚠️ 방향 차이 있음")

# ============================================================
# Part 6: 시각화
# ============================================================
print("\n" + "=" * 70)
print("Part 6: 시각화")
print("=" * 70)

fig = plt.figure(figsize=(12, 5))

# 3D 시각화
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('Camera Poses and 3D Points', fontsize=12)

# 3D 점
ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
            c='blue', s=20, alpha=0.6, label='3D Points')

# 카메라 1 (원점)
ax1.scatter([0], [0], [0], c='red', s=100, marker='^', label='Camera 1')

# 카메라 2 (추정된 위치)
cam2_pos = -R_est.T @ t_est
ax1.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], 
            c='green', s=100, marker='^', label='Camera 2 (est)')

# 방향 벡터
scale = 1.0
ax1.quiver(0, 0, 0, 0, 0, scale, color='red', alpha=0.5)
ax1.quiver(cam2_pos[0], cam2_pos[1], cam2_pos[2], 
           R_est[2, 0]*scale, R_est[2, 1]*scale, R_est[2, 2]*scale,
           color='green', alpha=0.5)

ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.legend()

# 4가지 해 비교
ax2 = fig.add_subplot(122)
ax2.set_title('Cheirality Check Results', fontsize=12)

names = [s[2] for s in solutions]
ratios = [cheirality_check(s[0], s[1], pts1, pts2, K) * 100 for s in solutions]
colors = ['green' if r > 90 else 'red' for r in ratios]

ax2.barh(names, ratios, color=colors)
ax2.set_xlabel('Points with Z > 0 (%)')
ax2.set_xlim([0, 105])
ax2.axvline(x=90, color='gray', linestyle='--', alpha=0.5)

for i, v in enumerate(ratios):
    ax2.text(v + 2, i, f'{v:.1f}%', va='center')

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week6/pose_estimation.png', dpi=150)
print("\nPose estimation saved: pose_estimation.png")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 70)
print("📚 Week 6 Basics 정리")
print("=" * 70)

print("""
✅ Part 1-2: E의 SVD 분해
   - E = U Σ Vᵀ
   - Σ = diag(σ, σ, 0)

✅ Part 3: 4가지 해 도출
   - R = U·W(ᵀ)·Vᵀ
   - t = ±U₃

✅ Part 4: Cheirality Check
   - 삼각측량 후 Z > 0 확인
   - 두 카메라 모두에서

✅ Part 5-6: 검증
   - GT와 비교
   - 시각화

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 핵심 메시지:
   E → 4가지 (R, t) 해
   Cheirality Check로 유일한 올바른 해 선택!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 다음: pose_estimation_quiz.py → Week 7: 삼각측량과 PnP
""")

print("\n" + "=" * 70)
print("pose_estimation_basics.py 실행 완료! 🎉")
print("=" * 70)
