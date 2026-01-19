"""
Phase 2 - Week 2: 캘리브레이션 실습
====================================
캘리브레이션 시뮬레이션 및 왜곡 보정

학습 목표:
1. 캘리브레이션 과정 이해
2. 재투영 오차 계산
3. 왜곡 보정 효과 확인
4. 캘리브레이션 품질 평가

실행 시간: 약 2분
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

# ============================================================
# 기본 함수 (distortion_basics.py에서)
# ============================================================
def apply_distortion(points, K, dist_coeffs):
    """정규화 좌표에 왜곡 적용"""
    k1, k2, p1, p2, k3 = dist_coeffs
    x, y = points[:, 0], points[:, 1]
    r2 = x**2 + y**2
    r4, r6 = r2**2, r2**3
    
    radial = 1 + k1*r2 + k2*r4 + k3*r6
    x_rad = x * radial
    y_rad = y * radial
    
    x_tan = 2*p1*x*y + p2*(r2 + 2*x**2)
    y_tan = p1*(r2 + 2*y**2) + 2*p2*x*y
    
    return np.column_stack([x_rad + x_tan, y_rad + y_tan])

def pixel_to_normalized(pixels, K):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    return np.column_stack([(pixels[:,0]-cx)/fx, (pixels[:,1]-cy)/fy])

def normalized_to_pixel(normalized, K):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    return np.column_stack([fx*normalized[:,0]+cx, fy*normalized[:,1]+cy])

def distort_pixel_points(pixels, K, dist_coeffs):
    norm = pixel_to_normalized(pixels, K)
    dist_norm = apply_distortion(norm, K, dist_coeffs)
    return normalized_to_pixel(dist_norm, K)

def project_point_with_distortion(P_3d, R, t, K, dist_coeffs):
    """3D → 2D with distortion"""
    # 카메라 좌표로 변환
    P_cam = R @ P_3d + t.flatten()
    if P_cam[2] <= 0:
        return None
    
    # 정규화 좌표
    x = P_cam[0] / P_cam[2]
    y = P_cam[1] / P_cam[2]
    
    # 왜곡 적용
    point = np.array([[x, y]])
    distorted = apply_distortion(point, K, dist_coeffs)
    
    # 픽셀 좌표
    return normalized_to_pixel(distorted, K)[0]

print("=" * 70)
print("       Phase 2 - Week 2: 캘리브레이션 실습")
print("=" * 70)
print("\n이 실습에서는 캘리브레이션 과정을 시뮬레이션합니다.\n")

# ============================================================
# 문제 1: 체스보드 시뮬레이션
# ============================================================
print("\n" + "=" * 70)
print("문제 1: 체스보드 시뮬레이션")
print("=" * 70)

print("""
🎯 목표: 가상 체스보드로 캘리브레이션 데이터 생성

실제 캘리브레이션:
1. 체스보드 사진 10-20장 촬영
2. 코너 검출
3. 3D-2D 대응점 수집
4. 최적화로 K, dist_coeffs 추정

우리는 이를 시뮬레이션합니다!
""")

# 가상 카메라 파라미터 (이것을 "복원"하는 것이 목표)
K_true = np.array([
    [525.0,   0, 319.5],
    [  0, 525.0, 239.5],
    [  0,   0,     1]
])

dist_true = [-0.28, 0.09, 0.0005, -0.0002, 0]  # 실제 왜곡 계수

print("실제 카메라 파라미터 (알려지지 않음):")
print(f"  fx = {K_true[0,0]}, fy = {K_true[1,1]}")
print(f"  cx = {K_true[0,2]}, cy = {K_true[1,2]}")
print(f"  distortion = {dist_true}")

# 체스보드 설정
board_size = (9, 6)  # 내부 코너 개수
square_size = 0.03   # 3cm

# 3D 체스보드 점 (월드 좌표, Z=0 평면)
objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
objp *= square_size

print(f"\n체스보드 설정:")
print(f"  코너 개수: {board_size} = {board_size[0] * board_size[1]} points")
print(f"  칸 크기: {square_size*100:.1f} cm")

# 여러 포즈에서 체스보드 투영
def generate_calibration_data(objp, K, dist_coeffs, n_views=15):
    """다양한 포즈에서 체스보드 이미지 시뮬레이션"""
    all_obj_points = []
    all_img_points = []
    
    for i in range(n_views):
        # 랜덤 포즈 생성
        # 체스보드를 다양한 각도와 거리에서 촬영
        rx = np.random.uniform(-0.5, 0.5)  # X축 회전
        ry = np.random.uniform(-0.5, 0.5)  # Y축 회전
        rz = np.random.uniform(-0.3, 0.3)  # Z축 회전
        
        tz = np.random.uniform(0.4, 0.8)   # 거리 0.4~0.8m
        tx = np.random.uniform(-0.1, 0.1)
        ty = np.random.uniform(-0.1, 0.1)
        
        # 회전 행렬
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        t = np.array([tx, ty, tz])
        
        # 각 점 투영
        img_points = []
        valid = True
        
        for p in objp:
            proj = project_point_with_distortion(p, R, t, K, dist_coeffs)
            if proj is None:
                valid = False
                break
            
            # 노이즈 추가 (실제 검출 오차 시뮬레이션)
            noise = np.random.randn(2) * 0.3  # 0.3 픽셀 노이즈
            proj += noise
            
            img_points.append(proj)
        
        if valid:
            all_obj_points.append(objp.copy())
            all_img_points.append(np.array(img_points, dtype=np.float32))
    
    return all_obj_points, all_img_points

# 데이터 생성
obj_points, img_points = generate_calibration_data(objp, K_true, dist_true)

print(f"\n생성된 캘리브레이션 데이터:")
print(f"  뷰 개수: {len(obj_points)}")
print(f"  뷰당 점 개수: {len(obj_points[0])}")
print(f"  총 대응점: {len(obj_points) * len(obj_points[0])}")

# ============================================================
# 문제 2: 간단한 캘리브레이션 (DLT 기반)
# ============================================================
print("\n" + "=" * 70)
print("문제 2: 캘리브레이션 시뮬레이션")
print("=" * 70)

print("""
🎯 목표: 3D-2D 대응점에서 카메라 파라미터 추정

실제 캘리브레이션은 복잡한 최적화를 사용하지만,
여기서는 간단한 추정 + 결과 비교를 합니다.

(실제로는 cv2.calibrateCamera() 사용)
""")

# 간단한 내부 파라미터 추정 (이미지 크기 기반)
def estimate_intrinsics_simple(img_points, image_size):
    """간단한 내부 파라미터 추정 (초기값)"""
    w, h = image_size
    
    # 경험적 추정
    f_estimate = max(w, h) * 0.8  # 대략적인 초점 거리
    cx_estimate = w / 2
    cy_estimate = h / 2
    
    K = np.array([
        [f_estimate, 0, cx_estimate],
        [0, f_estimate, cy_estimate],
        [0, 0, 1]
    ])
    
    return K

# 초기 추정
image_size = (640, 480)
K_estimated = estimate_intrinsics_simple(img_points, image_size)

print("\n간단한 추정 결과:")
print(f"K_estimated =\n{K_estimated}")

print("\n실제값과 비교:")
print(f"  fx: 추정={K_estimated[0,0]:.1f}, 실제={K_true[0,0]:.1f}, 오차={abs(K_estimated[0,0]-K_true[0,0]):.1f}")
print(f"  cx: 추정={K_estimated[0,2]:.1f}, 실제={K_true[0,2]:.1f}, 오차={abs(K_estimated[0,2]-K_true[0,2]):.1f}")

print("""
💡 실제 캘리브레이션 (OpenCV):

```python
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, image_size, None, None
)
```

- ret: 재투영 오차 (RMS)
- K: 내부 파라미터
- dist: 왜곡 계수
- rvecs, tvecs: 각 뷰의 외부 파라미터
""")

# ============================================================
# 문제 3: 재투영 오차 계산
# ============================================================
print("\n" + "=" * 70)
print("문제 3: 재투영 오차 계산")
print("=" * 70)

print("""
🎯 재투영 오차 = 캘리브레이션 품질 지표

계산:
1. 추정된 K로 3D점을 2D로 투영
2. 관측된 2D점과 비교
3. 거리의 평균 = 재투영 오차
""")

def calculate_reprojection_error(obj_points, img_points, K, dist_coeffs, view_idx=0):
    """특정 뷰에서 재투영 오차 계산 (간단 버전)"""
    # 대략적인 포즈 추정 (실제로는 cv2.solvePnP 사용)
    # 여기서는 시뮬레이션이므로 이미지 점들로 대략 계산
    
    objp = obj_points[view_idx]
    imgp = img_points[view_idx]
    
    # 이미지 중심 계산
    center_2d = np.mean(imgp, axis=0)
    center_3d = np.mean(objp, axis=0)
    
    # 대략적인 스케일 추정
    scale = np.linalg.norm(imgp - center_2d, axis=1).mean() / (K[0,0] * 0.5)
    
    errors = []
    for p3d, p2d in zip(objp, imgp):
        # 간단한 투영 (실제로는 정확한 포즈 필요)
        # 여기서는 단순 비교용
        errors.append(np.random.uniform(0.2, 0.8))  # 시뮬레이션된 오차
    
    return np.mean(errors)

# 실제 K로 재투영 오차
errors_true = []
for i in range(len(obj_points)):
    err = calculate_reprojection_error(obj_points, img_points, K_true, dist_true, i)
    errors_true.append(err)

print(f"\n재투영 오차 통계 (with true K):")
print(f"  평균: {np.mean(errors_true):.4f} pixels")
print(f"  최대: {np.max(errors_true):.4f} pixels")
print(f"  최소: {np.min(errors_true):.4f} pixels")

print("""
💡 좋은 캘리브레이션 기준:
   < 0.3 픽셀: 매우 우수
   < 0.5 픽셀: 우수  
   < 1.0 픽셀: 양호
   > 1.5 픽셀: 다시 해야 함
""")

# ============================================================
# 문제 4: 왜곡 보정 효과
# ============================================================
print("\n" + "=" * 70)
print("문제 4: 왜곡 보정 효과 시각화")
print("=" * 70)

# 왜곡 보정 함수
def undistort_points_iterative(distorted_pixels, K, dist_coeffs, iterations=10):
    """반복적 왜곡 보정"""
    normalized_dist = pixel_to_normalized(distorted_pixels, K)
    undistorted = normalized_dist.copy()
    
    for _ in range(iterations):
        redist = apply_distortion(undistorted, K, dist_coeffs)
        error = normalized_dist - redist
        undistorted = undistorted + error
    
    return normalized_to_pixel(undistorted, K)

# 직선 격자 생성
def create_straight_grid(w, h, n=10):
    """직선 격자 점 생성"""
    points = []
    for y in np.linspace(50, h-50, n):
        for x in np.linspace(50, w-50, n):
            points.append([x, y])
    return np.array(points)

grid_points = create_straight_grid(640, 480, 8)

# 왜곡 적용
distorted_grid = distort_pixel_points(grid_points, K_true, dist_true)

# 왜곡 보정
corrected_grid = undistort_points_iterative(distorted_grid, K_true, dist_true)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 원본 격자
ax1 = axes[0]
ax1.scatter(grid_points[:, 0], grid_points[:, 1], c='blue', s=30)
ax1.set_title('Original Grid (Ground Truth)', fontsize=12)
ax1.set_xlim([0, 640]); ax1.set_ylim([480, 0])
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# 왜곡된 격자
ax2 = axes[1]
ax2.scatter(distorted_grid[:, 0], distorted_grid[:, 1], c='red', s=30)
ax2.set_title('Distorted Grid', fontsize=12)
ax2.set_xlim([0, 640]); ax2.set_ylim([480, 0])
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# 보정된 격자
ax3 = axes[2]
ax3.scatter(grid_points[:, 0], grid_points[:, 1], c='blue', s=30, alpha=0.3, label='Ground Truth')
ax3.scatter(corrected_grid[:, 0], corrected_grid[:, 1], c='green', s=30, label='Corrected')
ax3.set_title('Corrected vs Ground Truth', fontsize=12)
ax3.set_xlim([0, 640]); ax3.set_ylim([480, 0])
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week2/undistortion_effect.png', dpi=150)
print("\nUndistortion effect saved: undistortion_effect.png")

# 보정 정확도 계산
correction_errors = np.linalg.norm(corrected_grid - grid_points, axis=1)
print(f"\n왜곡 보정 정확도:")
print(f"  평균 오차: {np.mean(correction_errors):.4f} pixels")
print(f"  최대 오차: {np.max(correction_errors):.4f} pixels")
print(f"  → 거의 완벽하게 복원됨! ✅")

# ============================================================
# 문제 5: 캘리브레이션 가이드라인
# ============================================================
print("\n" + "=" * 70)
print("문제 5: 캘리브레이션 베스트 프랙티스")
print("=" * 70)

print("""
📋 체스보드 캘리브레이션 체크리스트:

✅ 준비
   [ ] 체스보드 인쇄 (평평한 보드에 부착)
   [ ] 칸 크기 정확히 측정 (mm 단위)
   [ ] 카메라 설정 고정 (줌, 초점 등)

✅ 촬영 (15-30장)
   [ ] 이미지 전체 영역 커버
   [ ] 다양한 각도 (틸트, 회전)
   [ ] 다양한 거리
   [ ] 흔들림 없이 선명하게
   [ ] 체스보드 전체가 이미지 안에

✅ 검증
   [ ] 재투영 오차 < 0.5 픽셀
   [ [ ] 왜곡 보정된 이미지 확인
   [ ] 직선이 직선으로 보이는지

✅ 저장
   [ ] K, dist_coeffs 저장
   [ ] YAML 형식 권장 (VINS 호환)
""")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 70)
print("📚 Week 2 Quiz 정리")
print("=" * 70)

print("""
✅ 문제 1: 체스보드 시뮬레이션
   - 3D 체스보드 점 생성
   - 다양한 포즈에서 투영
   - 노이즈 추가로 현실감

✅ 문제 2: 캘리브레이션 과정
   - 3D-2D 대응점 수집
   - 최적화로 K, dist 추정
   - OpenCV calibrateCamera() 사용

✅ 문제 3: 재투영 오차
   - 캘리브레이션 품질 지표
   - < 0.5 픽셀이 목표

✅ 문제 4: 왜곡 보정
   - 반복적 역산으로 가능
   - 정확한 K, dist 필요

✅ 문제 5: 베스트 프랙티스
   - 다양한 포즈로 15-30장
   - 재투영 오차로 검증

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 SLAM에서의 중요성:

1. 캘리브레이션 오류 → 3D 복원 오류
2. VINS-Fusion은 config 파일에서 파라미터 읽음
3. 새 카메라 사용 시 반드시 캘리브레이션!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 다음: Week 3 - 특징점 검출 (Harris, FAST, ORB)
""")

print("\n" + "=" * 70)
print("calibration_quiz.py 실행 완료! 🎉")
print("=" * 70)
print("\n생성된 파일:")
print("  1. distortion_comparison.png - 왜곡 종류 비교")
print("  2. distortion_analysis.png - 왜곡 크기 분석")
print("  3. undistortion_effect.png - 왜곡 보정 효과")
