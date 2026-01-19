"""
Phase 2 - Week 2: 렌즈 왜곡 기초
================================
왜곡 모델 구현 및 시각화

학습 목표:
1. 방사 왜곡 이해 및 구현
2. 접선 왜곡 이해
3. 왜곡 보정 원리
4. OpenCV 왜곡 함수 사용

실행 시간: 약 1분
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

print("=" * 70)
print("        Phase 2 - Week 2: 렌즈 왜곡 기초")
print("=" * 70)
print("\n💡 이 실습에서는 카메라 렌즈 왜곡을 이해하고 시각화합니다.\n")

# ============================================================
# Part 1: 왜곡 모델 정의
# ============================================================
print("\n" + "=" * 70)
print("Part 1: 왜곡 모델 정의")
print("=" * 70)

print("""
📷 렌즈 왜곡의 두 종류:

1. 방사 왜곡 (Radial Distortion)
   - 렌즈 곡률로 인해 발생
   - 중심에서 멀수록 심함
   - 계수: k1, k2, k3

2. 접선 왜곡 (Tangential Distortion)
   - 렌즈-센서 정렬 오류로 발생
   - 보통 작음
   - 계수: p1, p2

OpenCV 왜곡 계수 형식:
   dist_coeffs = [k1, k2, p1, p2, k3]
""")

# 카메라 파라미터
image_width = 640
image_height = 480
K = np.array([
    [500,  0, 320],
    [ 0, 500, 240],
    [ 0,   0,   1]
])

# 다양한 왜곡 계수
distortion_types = {
    "No Distortion": [0, 0, 0, 0, 0],
    "Barrel (k1=-0.3)": [-0.3, 0, 0, 0, 0],
    "Pincushion (k1=0.3)": [0.3, 0, 0, 0, 0],
    "Barrel + k2": [-0.3, 0.1, 0, 0, 0],
    "Tangential": [0, 0, 0.01, 0.01, 0],
    "Fisheye-like": [-0.4, 0.2, 0, 0, -0.05],
}

print("다양한 왜곡 계수:")
print("-" * 60)
for name, coeffs in distortion_types.items():
    print(f"{name:25s}: {coeffs}")

# ============================================================
# Part 2: 왜곡 함수 구현
# ============================================================
print("\n" + "=" * 70)
print("Part 2: 왜곡 함수 구현")
print("=" * 70)

print("""
수학적 모델:

1. 정규화 좌표 계산: (x, y) = ((u-cx)/fx, (v-cy)/fy)
2. 거리 계산: r² = x² + y²
3. 방사 왜곡:
   x' = x(1 + k1·r² + k2·r⁴ + k3·r⁶)
   y' = y(1 + k1·r² + k2·r⁴ + k3·r⁶)
4. 접선 왜곡:
   x'' = x' + 2·p1·x·y + p2·(r² + 2·x²)
   y'' = y' + p1·(r² + 2·y²) + 2·p2·x·y
5. 픽셀 복원: (u', v') = (fx·x'' + cx, fy·y'' + cy)
""")

def apply_distortion(points, K, dist_coeffs):
    """
    정규화 좌표에 왜곡 적용
    
    Args:
        points: (N, 2) 정규화 좌표 [x, y]
        K: 내부 파라미터 행렬
        dist_coeffs: [k1, k2, p1, p2, k3]
    
    Returns:
        distorted_points: (N, 2) 왜곡된 정규화 좌표
    """
    k1, k2, p1, p2, k3 = dist_coeffs
    
    x = points[:, 0]
    y = points[:, 1]
    
    # r² 계산
    r2 = x**2 + y**2
    r4 = r2**2
    r6 = r2**3
    
    # 방사 왜곡
    radial = 1 + k1*r2 + k2*r4 + k3*r6
    x_radial = x * radial
    y_radial = y * radial
    
    # 접선 왜곡
    x_tangential = 2*p1*x*y + p2*(r2 + 2*x**2)
    y_tangential = p1*(r2 + 2*y**2) + 2*p2*x*y
    
    # 최종 왜곡 좌표
    x_dist = x_radial + x_tangential
    y_dist = y_radial + y_tangential
    
    return np.column_stack([x_dist, y_dist])

def pixel_to_normalized(pixels, K):
    """픽셀 좌표 → 정규화 좌표"""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (pixels[:, 0] - cx) / fx
    y = (pixels[:, 1] - cy) / fy
    
    return np.column_stack([x, y])

def normalized_to_pixel(normalized, K):
    """정규화 좌표 → 픽셀 좌표"""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = fx * normalized[:, 0] + cx
    v = fy * normalized[:, 1] + cy
    
    return np.column_stack([u, v])

def distort_pixel_points(pixels, K, dist_coeffs):
    """픽셀 좌표에 왜곡 적용"""
    # 픽셀 → 정규화
    normalized = pixel_to_normalized(pixels, K)
    
    # 왜곡 적용
    distorted_normalized = apply_distortion(normalized, K, dist_coeffs)
    
    # 정규화 → 픽셀
    distorted_pixels = normalized_to_pixel(distorted_normalized, K)
    
    return distorted_pixels

# 테스트
test_points = np.array([[400, 300], [600, 100], [100, 400]])
dist_test = [-0.3, 0.1, 0, 0, 0]

distorted = distort_pixel_points(test_points, K, dist_test)

print("\n왜곡 적용 테스트:")
print("-" * 50)
for orig, dist in zip(test_points, distorted):
    shift = np.linalg.norm(dist - orig)
    print(f"원본: ({orig[0]:6.1f}, {orig[1]:6.1f}) → "
          f"왜곡: ({dist[0]:6.1f}, {dist[1]:6.1f})  Δ={shift:.1f}px")

# ============================================================
# Part 3: 격자 왜곡 시각화
# ============================================================
print("\n" + "=" * 70)
print("Part 3: 격자 왜곡 시각화")
print("=" * 70)

# 직선 격자 생성
def create_grid(w, h, spacing=50):
    """직선 격자 생성"""
    lines = []
    
    # 수평선
    for y in range(0, h+1, spacing):
        line = np.array([[x, y] for x in range(0, w+1, 5)])
        lines.append(line)
    
    # 수직선
    for x in range(0, w+1, spacing):
        line = np.array([[x, y] for y in range(0, h+1, 5)])
        lines.append(line)
    
    return lines

grid_lines = create_grid(image_width, image_height)

# 다양한 왜곡 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, coeffs) in enumerate(distortion_types.items()):
    ax = axes[idx]
    ax.set_title(name, fontsize=12)
    ax.set_xlim([0, image_width])
    ax.set_ylim([image_height, 0])
    ax.set_aspect('equal')
    
    # 격자 그리기
    for line in grid_lines:
        if np.all(coeffs == 0):
            # 왜곡 없음
            distorted_line = line
        else:
            # 왜곡 적용
            distorted_line = distort_pixel_points(line, K, coeffs)
        
        ax.plot(distorted_line[:, 0], distorted_line[:, 1], 
               'b-', linewidth=0.5, alpha=0.7)
    
    # 중심점 표시
    ax.scatter([K[0, 2]], [K[1, 2]], c='red', s=50, marker='+', linewidths=2)
    ax.set_xlabel('u (pixels)')
    ax.set_ylabel('v (pixels)')

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week2/distortion_comparison.png', dpi=150)
print("\nDistortion comparison saved: distortion_comparison.png")

# ============================================================
# Part 4: 왜곡 크기 분석
# ============================================================
print("\n" + "=" * 70)
print("Part 4: 왜곡 크기 분석")
print("=" * 70)

print("""
왜곡은 중심에서 멀어질수록 커집니다!
이미지 각 위치에서 왜곡 크기를 분석해봅시다.
""")

# 이미지 여러 위치에서 왜곡 크기 계산
def analyze_distortion(K, dist_coeffs, image_size):
    """이미지 각 위치에서 왜곡 크기 분석"""
    w, h = image_size
    cx, cy = K[0, 2], K[1, 2]
    
    # 샘플 포인트 생성
    x = np.linspace(0, w, 20)
    y = np.linspace(0, h, 20)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.flatten(), yy.flatten()])
    
    # 왜곡 적용
    distorted = distort_pixel_points(points, K, dist_coeffs)
    
    # 왜곡 크기 (픽셀 이동 거리)
    displacement = np.linalg.norm(distorted - points, axis=1)
    
    # 중심에서의 거리
    distance_from_center = np.sqrt((points[:, 0] - cx)**2 + 
                                    (points[:, 1] - cy)**2)
    
    return distance_from_center, displacement

# 배럴 왜곡 분석
barrel_coeffs = [-0.3, 0.1, 0, 0, 0]
distances, displacements = analyze_distortion(K, barrel_coeffs, (image_width, image_height))

# 히트맵 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 왜곡 크기 vs 거리
ax1 = axes[0]
ax1.scatter(distances, displacements, alpha=0.6)
ax1.set_xlabel('Distance from Center (pixels)', fontsize=11)
ax1.set_ylabel('Distortion Magnitude (pixels)', fontsize=11)
ax1.set_title('Distortion vs. Distance from Center', fontsize=12)
ax1.grid(True, alpha=0.3)

# 왜곡 히트맵
ax2 = axes[1]
x = np.linspace(0, image_width, 50)
y = np.linspace(0, image_height, 50)
xx, yy = np.meshgrid(x, y)
points_grid = np.column_stack([xx.flatten(), yy.flatten()])
distorted_grid = distort_pixel_points(points_grid, K, barrel_coeffs)
displacement_grid = np.linalg.norm(distorted_grid - points_grid, axis=1)
displacement_map = displacement_grid.reshape(50, 50)

im = ax2.imshow(displacement_map, extent=[0, image_width, image_height, 0],
                cmap='hot', aspect='equal')
ax2.set_xlabel('u (pixels)', fontsize=11)
ax2.set_ylabel('v (pixels)', fontsize=11)
ax2.set_title('Distortion Magnitude Heatmap', fontsize=12)
plt.colorbar(im, ax=ax2, label='Displacement (pixels)')

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week2/distortion_analysis.png', dpi=150)
print("Distortion analysis saved: distortion_analysis.png")

print(f"\nBarrel distortion (k1={barrel_coeffs[0]}) analysis:")
print(f"  Center distortion: {displacements.min():.1f} pixels")
print(f"  Corner distortion: {displacements.max():.1f} pixels")
print(f"  → Corner is {displacements.max()/max(displacements.min(), 0.1):.1f}x more distorted!")

# ============================================================
# Part 5: OpenCV와 비교
# ============================================================
print("\n" + "=" * 70)
print("Part 5: OpenCV 함수 소개")
print("=" * 70)

print("""
OpenCV는 왜곡 관련 함수를 제공합니다:

1. cv2.undistort(img, K, dist_coeffs)
   - 이미지 전체 왜곡 보정
   - 간단하지만 느림

2. cv2.initUndistortRectifyMap() + cv2.remap()
   - 미리 매핑 테이블 생성
   - 반복 사용 시 빠름

3. cv2.undistortPoints(points, K, dist_coeffs)
   - 점 좌표만 보정

4. cv2.projectPoints(objPoints, rvec, tvec, K, dist_coeffs)
   - 3D → 2D 투영 + 왜곡 적용

예시:
```python
import cv2

# 이미지 왜곡 보정
undistorted_img = cv2.undistort(distorted_img, K, dist_coeffs)

# 맵 사용 (더 빠름)
mapx, mapy = cv2.initUndistortRectifyMap(
    K, dist_coeffs, None, K, (w, h), cv2.CV_32FC1
)
undistorted_img = cv2.remap(distorted_img, mapx, mapy, cv2.INTER_LINEAR)
```
""")

# ============================================================
# Part 6: 왜곡 보정 시뮬레이션
# ============================================================
print("\n" + "=" * 70)
print("Part 6: 왜곡 보정 시뮬레이션")
print("=" * 70)

print("""
왜곡 보정 = 왜곡의 역변환

문제: 왜곡 함수는 forward mapping (정상→왜곡)
보정: 필요한 것은 inverse mapping (왜곡→정상)

해결: 반복적 역산 또는 Look-up Table (OpenCV 방식)
""")

def undistort_points_iterative(distorted_pixels, K, dist_coeffs, iterations=10):
    """
    반복적 방법으로 왜곡 보정
    (간단한 구현 - OpenCV는 더 정교함)
    """
    # 초기 추정: 왜곡 좌표 = 정상 좌표라고 가정
    normalized_dist = pixel_to_normalized(distorted_pixels, K)
    undistorted = normalized_dist.copy()
    
    # 반복적 개선
    for _ in range(iterations):
        # 현재 추정에 왜곡 적용
        redist = apply_distortion(undistorted, K, dist_coeffs)
        
        # 오차 계산
        error = normalized_dist - redist
        
        # 추정 업데이트
        undistorted = undistorted + error
    
    return normalized_to_pixel(undistorted, K)

# 테스트
original_points = np.array([
    [100, 100], [540, 100], [100, 380], [540, 380], [320, 240]
])

# 왜곡 적용
distorted_points = distort_pixel_points(original_points, K, barrel_coeffs)

# 왜곡 보정
recovered_points = undistort_points_iterative(distorted_points, K, barrel_coeffs)

print("\n왜곡 → 보정 테스트:")
print("-" * 65)
print(f"{'Original':>15} | {'Distorted':>15} | {'Recovered':>15} | {'Error':>8}")
print("-" * 65)
for orig, dist, recov in zip(original_points, distorted_points, recovered_points):
    error = np.linalg.norm(recov - orig)
    print(f"({orig[0]:5.0f},{orig[1]:5.0f}) | "
          f"({dist[0]:5.1f},{dist[1]:5.1f}) | "
          f"({recov[0]:5.1f},{recov[1]:5.1f}) | "
          f"{error:6.4f} px")

print("\n✅ 반복적 방법으로 왜곡 보정 가능!")
print("   OpenCV는 이를 미리 계산된 맵으로 빠르게 처리")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 70)
print("📚 Week 2 Basics 정리")
print("=" * 70)

print("""
✅ Part 1: 왜곡 종류
   - 방사 왜곡: k1, k2, k3 (중심에서 멀수록 심함)
   - 접선 왜곡: p1, p2 (렌즈 정렬 오류)

✅ Part 2: 왜곡 함수
   - 정규화 좌표에서 왜곡 적용
   - r² = x² + y² 기반

✅ Part 3: 격자 시각화
   - 배럴: k1 < 0 (광각)
   - 핀쿠션: k1 > 0 (망원)

✅ Part 4: 왜곡 분석
   - 모서리 왜곡이 중심보다 훨씬 큼
   - 히트맵으로 분포 확인

✅ Part 5: OpenCV 함수
   - cv2.undistort(), cv2.remap()
   - cv2.undistortPoints()

✅ Part 6: 왜곡 보정
   - 반복적 역산으로 가능
   - OpenCV는 Look-up Table 사용

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 핵심 메시지:
   왜곡 보정은 SLAM의 필수 전처리!
   캘리브레이션으로 정확한 계수를 측정해야 함

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 다음: calibration_quiz.py → Week 3: 특징점 검출
""")

print("\n" + "=" * 70)
print("distortion_basics.py 실행 완료! 🎉")
print("=" * 70)
