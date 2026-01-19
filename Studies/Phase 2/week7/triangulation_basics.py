"""
Phase 2 - Week 7: 삼각측량 기초
===============================
DLT 삼각측량, 정확도 분석

학습 목표:
1. DLT 삼각측량 구현
2. 베이스라인 효과 이해
3. 재투영 오차 계산
4. 깊이 불확실성 분석

실행 시간: 약 1분
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("=" * 70)
print("        Phase 2 - Week 7: 삼각측량 기초")
print("=" * 70)
print("\n💡 이 실습에서는 2D 대응점에서 3D 점을 복원합니다.\n")

# ============================================================
# Part 1: 기본 설정
# ============================================================
print("\n" + "=" * 70)
print("Part 1: 기본 설정")
print("=" * 70)

# 카메라 파라미터
K = np.array([
    [500,   0, 320],
    [  0, 500, 240],
    [  0,   0,   1]
], dtype=np.float64)

def rotation_matrix_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

# 두 카메라 포즈
R1, t1 = np.eye(3), np.zeros(3)          # 카메라 1: 원점
R2 = rotation_matrix_y(np.radians(10))   # 카메라 2: 10도 회전
t2 = np.array([0.5, 0, 0])               # 오른쪽으로 0.5m

print("카메라 1: 원점")
print(f"카메라 2: R = {np.degrees(10):.1f}° Y회전, t = {t2}")

# 투영 행렬
P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
P2 = K @ np.hstack([R2, t2.reshape(3, 1)])

print(f"\nP1 (3×4):\n{P1}")
print(f"\nP2 (3×4):\n{P2}")

# ============================================================
# Part 2: DLT 삼각측량 구현
# ============================================================
print("\n" + "=" * 70)
print("Part 2: DLT 삼각측량 구현")
print("=" * 70)

print("""
🎯 DLT 삼각측량:

p × (P·X) = 0 → 선형 시스템 A·X = 0

A = [u₁P₁³ᵀ - P₁¹ᵀ]
    [v₁P₁³ᵀ - P₁²ᵀ]
    [u₂P₂³ᵀ - P₂¹ᵀ]
    [v₂P₂³ᵀ - P₂²ᵀ]

SVD로 해: 가장 작은 특이값에 해당하는 벡터
""")

def triangulate_dlt(P1, P2, p1, p2):
    """
    DLT 삼각측량
    
    Args:
        P1, P2: (3, 4) 투영 행렬
        p1, p2: (2,) 이미지 좌표 [u, v]
    
    Returns:
        X: (3,) 3D 좌표
    """
    u1, v1 = p1
    u2, v2 = p2
    
    A = np.array([
        u1 * P1[2] - P1[0],
        v1 * P1[2] - P1[1],
        u2 * P2[2] - P2[0],
        v2 * P2[2] - P2[1]
    ])
    
    # SVD
    _, _, Vt = np.linalg.svd(A)
    X_homo = Vt[-1]
    
    # 동차 좌표 → 유클리드
    X = X_homo[:3] / X_homo[3]
    
    return X

def project(P, X):
    """3D → 2D 투영"""
    X_homo = np.append(X, 1)
    p_homo = P @ X_homo
    return p_homo[:2] / p_homo[2]

# 테스트: 알려진 3D 점
X_gt = np.array([0.3, -0.2, 5.0])

# 두 카메라에 투영
p1 = project(P1, X_gt)
p2 = project(P2, X_gt)

print(f"\nGround Truth 3D 점: {X_gt}")
print(f"카메라 1 투영: {p1}")
print(f"카메라 2 투영: {p2}")

# 삼각측량으로 복원
X_reconstructed = triangulate_dlt(P1, P2, p1, p2)

print(f"\n복원된 3D 점: {X_reconstructed}")
print(f"오차: {np.linalg.norm(X_reconstructed - X_gt):.6f}")

# ============================================================
# Part 3: 노이즈 영향
# ============================================================
print("\n" + "=" * 70)
print("Part 3: 노이즈 영향")
print("=" * 70)

print("""
🎯 실제 이미지 점에는 노이즈가 있음
→ 삼각측량 오차 발생
""")

noise_levels = [0, 0.5, 1.0, 2.0, 5.0]

print("\n노이즈 수준에 따른 삼각측량 오차:")
print("-" * 50)
print(f"{'Noise (px)':>12} | {'3D Error (m)':>15} | {'Reproj Error (px)':>18}")
print("-" * 50)

errors_by_noise = []

for noise in noise_levels:
    errors = []
    reproj_errors = []
    
    for _ in range(100):
        # 노이즈 추가
        p1_noisy = p1 + np.random.randn(2) * noise
        p2_noisy = p2 + np.random.randn(2) * noise
        
        # 삼각측량
        X_est = triangulate_dlt(P1, P2, p1_noisy, p2_noisy)
        
        # 3D 오차
        error_3d = np.linalg.norm(X_est - X_gt)
        errors.append(error_3d)
        
        # 재투영 오차
        p1_reproj = project(P1, X_est)
        error_reproj = np.linalg.norm(p1_reproj - p1)
        reproj_errors.append(error_reproj)
    
    mean_error = np.mean(errors)
    mean_reproj = np.mean(reproj_errors)
    errors_by_noise.append(mean_error)
    
    print(f"{noise:>12.1f} | {mean_error:>15.4f} | {mean_reproj:>18.4f}")

# ============================================================
# Part 4: 베이스라인 효과
# ============================================================
print("\n" + "=" * 70)
print("Part 4: 베이스라인 효과")
print("=" * 70)

print("""
🎯 베이스라인(카메라 간 거리)이 삼각측량 정확도에 미치는 영향

좁은 베이스라인 → 깊이 불확실성 증가
넓은 베이스라인 → 정확도 향상 (매칭 어려움)
""")

baselines = [0.1, 0.3, 0.5, 1.0, 2.0]
noise_fixed = 1.0  # 1픽셀 노이즈

print(f"\n베이스라인에 따른 오차 (노이즈={noise_fixed}px, 깊이=5m):")
print("-" * 50)
print(f"{'Baseline (m)':>12} | {'3D Error (m)':>15} | {'삼각측량 각도':>15}")
print("-" * 50)

errors_by_baseline = []

for baseline in baselines:
    # 베이스라인 조정
    t2_new = np.array([baseline, 0, 0])
    P2_new = K @ np.hstack([R2, t2_new.reshape(3, 1)])
    
    # 새 투영 계산
    p2_new = project(P2_new, X_gt)
    
    # 삼각측량 각도 계산
    # 두 광선 사이 각도
    ray1 = X_gt - t1
    ray2 = X_gt - t2_new
    cos_angle = np.dot(ray1, ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    errors = []
    for _ in range(100):
        p1_noisy = p1 + np.random.randn(2) * noise_fixed
        p2_noisy = p2_new + np.random.randn(2) * noise_fixed
        
        X_est = triangulate_dlt(P1, P2_new, p1_noisy, p2_noisy)
        errors.append(np.linalg.norm(X_est - X_gt))
    
    mean_error = np.mean(errors)
    errors_by_baseline.append(mean_error)
    print(f"{baseline:>12.2f} | {mean_error:>15.4f} | {angle:>14.1f}°")

print("""
💡 관찰:
   베이스라인 ↑ → 삼각측량 각도 ↑ → 오차 ↓
   권장: 5° 이상 삼각측량 각도
""")

# ============================================================
# Part 5: 여러 점 삼각측량
# ============================================================
print("\n" + "=" * 70)
print("Part 5: 여러 점 삼각측량")
print("=" * 70)

# 3D 점 생성
np.random.seed(42)
points_3d_gt = np.random.rand(20, 3) * np.array([2, 2, 3]) + np.array([-1, -1, 4])

# 투영
pts1 = np.array([project(P1, X) for X in points_3d_gt])
pts2 = np.array([project(P2, X) for X in points_3d_gt])

# 노이즈 추가
pts1_noisy = pts1 + np.random.randn(*pts1.shape) * 0.5
pts2_noisy = pts2 + np.random.randn(*pts2.shape) * 0.5

# 삼각측량
points_3d_est = np.array([
    triangulate_dlt(P1, P2, p1, p2) 
    for p1, p2 in zip(pts1_noisy, pts2_noisy)
])

# 오차 분석
errors = np.linalg.norm(points_3d_est - points_3d_gt, axis=1)

print(f"점 개수: {len(points_3d_gt)}")
print(f"평균 3D 오차: {np.mean(errors):.4f} m")
print(f"최대 3D 오차: {np.max(errors):.4f} m")
print(f"최소 3D 오차: {np.min(errors):.4f} m")

# ============================================================
# Part 6: 시각화
# ============================================================
print("\n" + "=" * 70)
print("Part 6: 시각화")
print("=" * 70)

fig = plt.figure(figsize=(14, 5))

# 3D 점 시각화
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(points_3d_gt[:, 0], points_3d_gt[:, 1], points_3d_gt[:, 2],
            c='blue', s=30, label='Ground Truth', alpha=0.6)
ax1.scatter(points_3d_est[:, 0], points_3d_est[:, 1], points_3d_est[:, 2],
            c='red', s=30, label='Reconstructed', alpha=0.6)

# 카메라 표시
ax1.scatter([0], [0], [0], c='green', s=100, marker='^', label='Cam 1')
cam2_pos = -R2.T @ t2
ax1.scatter([cam2_pos[0]], [cam2_pos[1]], [cam2_pos[2]], 
            c='orange', s=100, marker='^', label='Cam 2')

ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.set_title('3D Reconstruction', fontsize=11)
ax1.legend(fontsize=8)

# 베이스라인 vs 오차
ax2 = fig.add_subplot(132)
ax2.plot(baselines, errors_by_baseline, 'bo-', linewidth=2, markersize=8)
ax2.set_xlabel('Baseline (m)', fontsize=11)
ax2.set_ylabel('Mean 3D Error (m)', fontsize=11)
ax2.set_title('Baseline vs Error', fontsize=11)
ax2.grid(True, alpha=0.3)

# 노이즈 vs 오차
ax3 = fig.add_subplot(133)
ax3.plot(noise_levels, errors_by_noise, 'ro-', linewidth=2, markersize=8)
ax3.set_xlabel('Pixel Noise (px)', fontsize=11)
ax3.set_ylabel('Mean 3D Error (m)', fontsize=11)
ax3.set_title('Noise vs Error', fontsize=11)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week7/triangulation_analysis.png', dpi=150)
print("\nTriangulation analysis saved: triangulation_analysis.png")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 70)
print("📚 Week 7 Triangulation 정리")
print("=" * 70)

print("""
✅ Part 1-2: DLT 삼각측량
   - A·X = 0 선형 시스템
   - SVD로 해 구함

✅ Part 3: 노이즈 영향
   - 노이즈 ↑ → 3D 오차 ↑
   - 재투영 오차로 품질 평가

✅ Part 4: 베이스라인 효과
   - 좁은 베이스라인 → 깊이 불확실
   - 권장: 5° 이상 삼각측량 각도

✅ Part 5-6: 다중 점 분석
   - 전체 정확도 통계
   - 시각화

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 핵심 메시지:
   삼각측량 = 2D-2D → 3D
   베이스라인과 노이즈가 정확도 좌우!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 다음: pnp_quiz.py → Week 8: 광류
""")

print("\n" + "=" * 70)
print("triangulation_basics.py 실행 완료! 🎉")
print("=" * 70)
