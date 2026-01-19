"""
Phase 2 - Week 7: PnP 실습 문제
==============================
PnP 구현, RANSAC, 재투영 오차

학습 목표:
1. PnP 원리 이해
2. 재투영 오차 계산
3. RANSAC 필요성
4. 다양한 PnP 알고리즘

실행 시간: 약 2분
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("=" * 70)
print("       Phase 2 - Week 7: PnP 실습 문제")
print("=" * 70)
print("\n이 실습에서는 3D-2D 대응에서 카메라 포즈를 추정합니다.\n")

# ============================================================
# 기본 함수
# ============================================================
def rotation_matrix(axis, theta):
    if axis == 'x':
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def project(K, R, t, X):
    """3D → 2D 투영"""
    X_cam = R @ X + t
    p = K @ X_cam
    return p[:2] / p[2]

K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)

# ============================================================
# 문제 1: PnP 기본 원리
# ============================================================
print("\n" + "=" * 70)
print("문제 1: PnP 기본 원리")
print("=" * 70)

print("""
🎯 PnP (Perspective-n-Point):

주어진 것:
- N개 3D 점 (맵에서 알려진 좌표)
- 해당 점들의 2D 투영 (현재 이미지)
- 카메라 내부 파라미터 K

구하는 것:
- 카메라 포즈 (R, t)
""")

# Ground Truth 포즈
R_gt = rotation_matrix('y', np.radians(20)) @ rotation_matrix('x', np.radians(10))
t_gt = np.array([1.0, 0.5, 0.2])

print("Ground Truth 포즈:")
print(f"R:\n{R_gt}")
print(f"t: {t_gt}")

# 3D 점 생성 (맵 좌표)
np.random.seed(42)
object_points = np.random.rand(20, 3) * 4 + np.array([-2, -2, 5])

# 2D 점 생성 (이미지 투영)
image_points = np.array([project(K, R_gt, t_gt, X) for X in object_points])

print(f"\n3D 점 개수: {len(object_points)}")
print(f"2D 점 범위: X [{image_points[:, 0].min():.0f}, {image_points[:, 0].max():.0f}], "
      f"Y [{image_points[:, 1].min():.0f}, {image_points[:, 1].max():.0f}]")

# ============================================================
# 문제 2: DLT PnP 구현
# ============================================================
print("\n" + "=" * 70)
print("문제 2: 간단한 PnP (DLT)")
print("=" * 70)

print("""
🎯 DLT 방식: P = K[R|t] 직접 추정 후 분해

p = P · X (동차)
→ 선형 시스템으로 P 추정
→ P에서 K, R, t 분해
""")

def pnp_dlt(object_points, image_points, K):
    """
    DLT PnP (간단 버전)
    6점 이상 필요
    """
    n = len(object_points)
    
    # A 행렬 구성
    A = []
    for i in range(n):
        X, Y, Z = object_points[i]
        u, v = image_points[i]
        
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    
    A = np.array(A)
    
    # SVD
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    
    # P에서 K[R|t] 분해
    # P = K[R|t] → K⁻¹P = [R|t]
    K_inv = np.linalg.inv(K)
    M = K_inv @ P
    
    R = M[:, :3]
    t = M[:, 3]
    
    # R 직교화
    U, S, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    
    # 스케일 조정
    scale = np.mean(S)
    t = t / scale
    
    return R, t

# DLT PnP 실행
R_est, t_est = pnp_dlt(object_points, image_points, K)

print("\nDLT PnP 결과:")
print(f"R_est:\n{R_est}")
print(f"t_est: {t_est}")

# 오차
R_error = np.linalg.norm(R_est - R_gt, 'fro')
t_error = np.linalg.norm(t_est - t_gt)

print(f"\n오차:")
print(f"  R 오차 (Frobenius): {R_error:.6f}")
print(f"  t 오차: {t_error:.6f}")

# ============================================================
# 문제 3: 재투영 오차
# ============================================================
print("\n" + "=" * 70)
print("문제 3: 재투영 오차 계산")
print("=" * 70)

print("""
🎯 재투영 오차 = 품질 측정 지표

1. 추정된 R, t로 3D → 2D 재투영
2. 원래 2D점과 비교
3. 거리의 평균/RMS = 재투영 오차
""")

def compute_reprojection_error(R, t, object_points, image_points, K):
    """재투영 오차 계산"""
    errors = []
    for X, p_obs in zip(object_points, image_points):
        p_proj = project(K, R, t, X)
        error = np.linalg.norm(p_proj - p_obs)
        errors.append(error)
    return np.array(errors)

# 재투영 오차
errors_gt = compute_reprojection_error(R_gt, t_gt, object_points, image_points, K)
errors_est = compute_reprojection_error(R_est, t_est, object_points, image_points, K)

print(f"\nGround Truth 재투영 오차: {np.mean(errors_gt):.6f} px")
print(f"추정 포즈 재투영 오차:   {np.mean(errors_est):.4f} px")

# ============================================================
# 문제 4: 노이즈 영향
# ============================================================
print("\n" + "=" * 70)
print("문제 4: 노이즈 영향")
print("=" * 70)

noise_levels = [0, 0.5, 1.0, 2.0, 5.0]

print("\n노이즈에 따른 PnP 오차:")
print("-" * 60)
print(f"{'Noise (px)':>12} | {'R Error':>12} | {'t Error':>12} | {'Reproj (px)':>12}")
print("-" * 60)

for noise in noise_levels:
    # 노이즈 추가
    image_points_noisy = image_points + np.random.randn(*image_points.shape) * noise
    
    # PnP
    R_n, t_n = pnp_dlt(object_points, image_points_noisy, K)
    
    # 오차
    R_err = np.linalg.norm(R_n - R_gt, 'fro')
    t_err = np.linalg.norm(t_n - t_gt)
    reproj = np.mean(compute_reprojection_error(R_n, t_n, object_points, image_points, K))
    
    print(f"{noise:>12.1f} | {R_err:>12.6f} | {t_err:>12.4f} | {reproj:>12.4f}")

# ============================================================
# 문제 5: Outlier와 RANSAC
# ============================================================
print("\n" + "=" * 70)
print("문제 5: Outlier 영향")
print("=" * 70)

print("""
🎯 Outlier = 잘못된 2D-3D 대응

소수 outlier도 PnP를 크게 왜곡!
→ RANSAC 필수
""")

# Outlier 추가
n_outliers = 5
outlier_indices = np.random.choice(len(image_points), n_outliers, replace=False)
image_points_outlier = image_points.copy()
image_points_outlier[outlier_indices] += np.random.randn(n_outliers, 2) * 100

# Outlier 있는 PnP
R_out, t_out = pnp_dlt(object_points, image_points_outlier, K)

print(f"\n정상 데이터:")
print(f"  R 오차: {np.linalg.norm(R_est - R_gt, 'fro'):.6f}")
print(f"  t 오차: {np.linalg.norm(t_est - t_gt):.4f}")

print(f"\nOutlier 포함 ({n_outliers}개):")
print(f"  R 오차: {np.linalg.norm(R_out - R_gt, 'fro'):.6f}")
print(f"  t 오차: {np.linalg.norm(t_out - t_gt):.4f}")

print("""
💡 OpenCV PnP RANSAC 사용법:

```python
import cv2

success, rvec, tvec, inliers = cv2.solvePnPRansac(
    object_points, image_points, K, None,
    reprojectionError=3.0,
    confidence=0.99
)

R, _ = cv2.Rodrigues(rvec)
```
""")

# ============================================================
# 문제 6: 시각화
# ============================================================
print("\n" + "=" * 70)
print("문제 6: 시각화")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 2D 점과 재투영
ax1 = axes[0]
ax1.scatter(image_points[:, 0], image_points[:, 1], c='blue', s=50, label='Observed', alpha=0.7)
reproj_pts = np.array([project(K, R_est, t_est, X) for X in object_points])
ax1.scatter(reproj_pts[:, 0], reproj_pts[:, 1], c='red', s=50, marker='x', label='Reprojected', alpha=0.7)
for obs, rep in zip(image_points, reproj_pts):
    ax1.plot([obs[0], rep[0]], [obs[1], rep[1]], 'g-', alpha=0.3)
ax1.set_xlim([0, 640]); ax1.set_ylim([480, 0])
ax1.set_title('Reprojection (Blue=Obs, Red=Est)', fontsize=11)
ax1.legend(); ax1.grid(True, alpha=0.3)

# 3D 뷰
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], 
            c='blue', s=30, label='3D Points')

# 카메라 위치
cam_gt = -R_gt.T @ t_gt
cam_est = -R_est.T @ t_est
ax2.scatter([cam_gt[0]], [cam_gt[1]], [cam_gt[2]], c='green', s=100, marker='^', label='GT Cam')
ax2.scatter([cam_est[0]], [cam_est[1]], [cam_est[2]], c='red', s=100, marker='^', label='Est Cam')

ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
ax2.set_title('3D View', fontsize=11)
ax2.legend(fontsize=8)

# 재투영 오차 분포
ax3 = axes[2]
ax3.hist(errors_est, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
ax3.axvline(np.mean(errors_est), color='red', linestyle='--', label=f'Mean: {np.mean(errors_est):.2f}px')
ax3.set_xlabel('Reprojection Error (px)', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('Error Distribution', fontsize=11)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week7/pnp_analysis.png', dpi=150)
print("\nPnP analysis saved: pnp_analysis.png")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 70)
print("📚 Week 7 PnP Quiz 정리")
print("=" * 70)

print("""
✅ 문제 1: PnP 원리
   - 3D-2D 대응 → 카메라 포즈
   - 최소 6점 (DLT), 3점 (P3P)

✅ 문제 2: DLT PnP
   - P = K[R|t] 추정
   - SVD + 직교화

✅ 문제 3: 재투영 오차
   - 품질 평가 지표
   - < 1 px 이면 좋음

✅ 문제 4: 노이즈 영향
   - 노이즈 ↑ → 오차 ↑

✅ 문제 5: Outlier
   - 소수도 치명적
   - solvePnPRansac 필수

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 SLAM 파이프라인:

1. 초기화: E 분해 + 삼각측량 → 초기 맵
2. 추적: PnP로 새 프레임 포즈 추정
3. 맵 확장: 새 점 삼각측량

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 다음: Week 8 - 광류 (Optical Flow)
""")

print("\n" + "=" * 70)
print("pnp_quiz.py 실행 완료! 🎉")
print("=" * 70)
print("\n생성된 파일:")
print("  1. triangulation_analysis.png - 삼각측량 분석")
print("  2. pnp_analysis.png - PnP 분석")
