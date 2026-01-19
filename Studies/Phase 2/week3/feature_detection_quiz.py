"""
Phase 2 - Week 3: 특징점 검출 실습 문제
======================================
파라미터 튜닝, 알고리즘 비교, NMS 구현

학습 목표:
1. 파라미터가 검출에 미치는 영향
2. Non-maximum Suppression 구현
3. 알고리즘 성능 비교
4. 특징점 분포 분석

실행 시간: 약 2분
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, maximum_filter

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("=" * 70)
print("       Phase 2 - Week 3: 특징점 검출 실습 문제")
print("=" * 70)
print("\n이 실습에서는 특징점 검출을 더 깊이 탐구합니다.\n")

# ============================================================
# 기본 함수
# ============================================================
def harris_corner_detector(image, k=0.04, window_size=3, threshold=0.01):
    """Harris 코너 검출"""
    Ix = np.zeros_like(image)
    Iy = np.zeros_like(image)
    Ix[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2
    Iy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2
    
    Ixx = uniform_filter(Ix * Ix, size=window_size)
    Iyy = uniform_filter(Iy * Iy, size=window_size)
    Ixy = uniform_filter(Ix * Iy, size=window_size)
    
    det = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy
    R = det - k * trace * trace
    
    return R

def create_test_image_with_noise(size=200, noise_level=0):
    """노이즈가 있는 테스트 이미지"""
    img = np.ones((size, size), dtype=np.float32) * 128
    
    # 크기에 맞게 스케일 조정
    if size >= 200:
        img[40:80, 40:100] = 200
        for i in range(40):
            if 100+i < size and 120+i < size:
                img[100+i, max(0, 120-i):min(size, 120+i+1)] = 50
        
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 0:
                    y1, y2 = 130+i*15, min(size, 130+(i+1)*15)
                    x1, x2 = 130+j*15, min(size, 130+(j+1)*15)
                    if y1 < size and x1 < size:
                        img[y1:y2, x1:x2] = 30
    else:
        # 작은 이미지용 간단 패턴
        img[size//5:size//3, size//5:size//2] = 200
        img[size//2:size*2//3, size//3:size*2//3] = 50
    
    if noise_level > 0:
        img += np.random.randn(size, size) * noise_level
        img = np.clip(img, 0, 255)
    
    return img

# ============================================================
# 문제 1: Non-maximum Suppression
# ============================================================
print("\n" + "=" * 70)
print("문제 1: Non-maximum Suppression (NMS)")
print("=" * 70)

print("""
🎯 목표: 밀집된 코너들 중 최대값만 남기기

문제:
- Harris 응답이 코너 주변에서 높음
- 여러 픽셀이 같은 코너로 검출됨
- 하나의 코너 = 하나의 점만 필요!

해결: NMS
- 지역 윈도우 내 최대값만 유지
- 나머지는 억제
""")

def non_maximum_suppression(response, window_size=5, threshold=0.01):
    """
    Non-maximum Suppression 구현
    
    Args:
        response: Harris 응답 맵
        window_size: 지역 최대 검색 윈도우
        threshold: 응답 임계값 (비율)
    
    Returns:
        corners: (N, 2) 코너 좌표 [(x, y), ...]
    """
    # 지역 최대 필터
    local_max = maximum_filter(response, size=window_size)
    
    # 지역 최대이면서 임계값 이상인 점
    thresh_value = threshold * response.max()
    
    corners_mask = (response == local_max) & (response > thresh_value)
    
    # 좌표 추출
    coords = np.argwhere(corners_mask)
    corners = [(c[1], c[0]) for c in coords]  # (x, y)
    
    return corners

# 테스트
test_img = create_test_image_with_noise(200, noise_level=0)
R = harris_corner_detector(test_img, threshold=0.01)

# NMS 전후 비교
corners_before = np.argwhere(R > 0.01 * R.max())
corners_after = non_maximum_suppression(R, window_size=7, threshold=0.01)

print(f"\nNMS 효과:")
print(f"  NMS 전: {len(corners_before)} 점")
print(f"  NMS 후: {len(corners_after)} 점")
print(f"  제거된 중복: {len(corners_before) - len(corners_after)} 점")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax1 = axes[0]
ax1.imshow(test_img, cmap='gray')
ax1.set_title('Original Image', fontsize=12)
ax1.axis('off')

ax2 = axes[1]
ax2.imshow(test_img, cmap='gray')
for y, x in corners_before[:100]:
    ax2.plot(x, y, 'r.', markersize=3)
ax2.set_title(f'Before NMS ({len(corners_before)} points)', fontsize=12)
ax2.axis('off')

ax3 = axes[2]
ax3.imshow(test_img, cmap='gray')
for x, y in corners_after:
    ax3.plot(x, y, 'g.', markersize=8)
ax3.set_title(f'After NMS ({len(corners_after)} points)', fontsize=12)
ax3.axis('off')

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week3/nms_comparison.png', dpi=150)
print("NMS comparison saved: nms_comparison.png")

# ============================================================
# 문제 2: 파라미터 튜닝
# ============================================================
print("\n" + "=" * 70)
print("문제 2: 파라미터 튜닝 효과")
print("=" * 70)

print("""
🎯 목표: threshold가 검출에 미치는 영향 분석

threshold ↑ → 적은 검출, 강한 코너만
threshold ↓ → 많은 검출, 약한 코너도
""")

thresholds = [0.001, 0.01, 0.05, 0.1]
detection_counts = []

print("\nThreshold에 따른 검출 수:")
print("-" * 40)

for thresh in thresholds:
    corners = non_maximum_suppression(R, window_size=7, threshold=thresh)
    detection_counts.append(len(corners))
    print(f"  threshold = {thresh:.3f}: {len(corners):4d} corners")

# 시각화
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, thresh in enumerate(thresholds):
    corners = non_maximum_suppression(R, window_size=7, threshold=thresh)
    
    ax = axes[idx]
    ax.imshow(test_img, cmap='gray')
    for x, y in corners:
        ax.plot(x, y, 'r.', markersize=5)
    ax.set_title(f'threshold = {thresh}\n({len(corners)} corners)', fontsize=11)
    ax.axis('off')

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week3/threshold_tuning.png', dpi=150)
print("Threshold tuning saved: threshold_tuning.png")

# ============================================================
# 문제 3: 노이즈 강건성
# ============================================================
print("\n" + "=" * 70)
print("문제 3: 노이즈에 대한 강건성")
print("=" * 70)

print("""
🎯 목표: 노이즈가 특징점 검출에 미치는 영향

실제 카메라 이미지:
- 센서 노이즈 존재
- 조명 변화
- 모션 블러

강건한 검출기가 필요!
""")

noise_levels = [0, 10, 30, 50]

print("\n노이즈 수준에 따른 검출:")
print("-" * 40)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for idx, noise in enumerate(noise_levels):
    # 노이즈 이미지 생성
    noisy_img = create_test_image_with_noise(200, noise_level=noise)
    
    # Harris 검출
    R_noisy = harris_corner_detector(noisy_img)
    corners = non_maximum_suppression(R_noisy, window_size=7, threshold=0.01)
    
    print(f"  noise = {noise:2d}: {len(corners):4d} corners")
    
    # 시각화
    axes[0, idx].imshow(noisy_img, cmap='gray')
    axes[0, idx].set_title(f'Noise = {noise}', fontsize=11)
    axes[0, idx].axis('off')
    
    axes[1, idx].imshow(noisy_img, cmap='gray')
    for x, y in corners:
        axes[1, idx].plot(x, y, 'r.', markersize=4)
    axes[1, idx].set_title(f'{len(corners)} corners', fontsize=11)
    axes[1, idx].axis('off')

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week3/noise_robustness.png', dpi=150)
print("Noise robustness saved: noise_robustness.png")

# ============================================================
# 문제 4: 특징점 분포
# ============================================================
print("\n" + "=" * 70)
print("문제 4: 특징점 분포 분석")
print("=" * 70)

print("""
🎯 목표: 이미지 전체에 균일하게 분포시키기

문제: 특징점이 한 곳에 몰림
해결: 그리드 기반 검출 또는 최소 거리 제약

VINS 파라미터: min_dist = 30 (픽셀)
""")

def enforce_min_distance(corners, min_dist=30):
    """최소 거리 제약 적용"""
    if len(corners) == 0:
        return []
    
    selected = [corners[0]]
    
    for c in corners[1:]:
        too_close = False
        for s in selected:
            dist = np.sqrt((c[0] - s[0])**2 + (c[1] - s[1])**2)
            if dist < min_dist:
                too_close = True
                break
        
        if not too_close:
            selected.append(c)
    
    return selected

# 최소 거리 적용
R_clean = harris_corner_detector(test_img)
corners_all = non_maximum_suppression(R_clean, window_size=5, threshold=0.01)

# 응답 강도로 정렬 (강한 것부터)
corners_sorted = sorted(corners_all, 
                        key=lambda c: R_clean[c[1], c[0]], 
                        reverse=True)

corners_spaced = enforce_min_distance(corners_sorted, min_dist=20)

print(f"\n최소 거리 제약:")
print(f"  적용 전: {len(corners_all)} points")
print(f"  min_dist=20 적용 후: {len(corners_spaced)} points")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.imshow(test_img, cmap='gray')
for x, y in corners_all:
    ax1.plot(x, y, 'r.', markersize=5)
ax1.set_title(f'Without Min Distance ({len(corners_all)} pts)', fontsize=12)
ax1.axis('off')

ax2 = axes[1]
ax2.imshow(test_img, cmap='gray')
for x, y in corners_spaced:
    ax2.plot(x, y, 'g.', markersize=8)
ax2.set_title(f'With Min Distance 20px ({len(corners_spaced)} pts)', fontsize=12)
ax2.axis('off')

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week3/uniform_distribution.png', dpi=150)
print("Uniform distribution saved: uniform_distribution.png")

# ============================================================
# 문제 5: 알고리즘 비교
# ============================================================
print("\n" + "=" * 70)
print("문제 5: 알고리즘 속도 비교")
print("=" * 70)

print("""
🎯 목표: Harris vs FAST 속도 비교

FAST가 빠른 이유:
1. 간단한 비교 연산만 사용
2. Early exit (4점 테스트로 빠르게 제외)
3. 행렬 연산 없음
""")

import time

# 속도 테스트
test_sizes = [100, 200, 300]
harris_times = []
simple_times = []

print("\n이미지 크기별 처리 시간:")
print("-" * 50)
print(f"{'Size':>10} | {'Harris (ms)':>15} | {'Simple (ms)':>15}")
print("-" * 50)

for size in test_sizes:
    img = create_test_image_with_noise(size, 0)
    
    # Harris
    start = time.time()
    for _ in range(10):
        R = harris_corner_detector(img)
    harris_time = (time.time() - start) / 10 * 1000
    harris_times.append(harris_time)
    
    # Simple comparison (simulating FAST concept)
    start = time.time()
    for _ in range(10):
        # 간단한 비교 연산 시뮬레이션
        diff = np.abs(img[:-2, 1:-1] - img[2:, 1:-1])
    simple_time = (time.time() - start) / 10 * 1000
    simple_times.append(simple_time)
    
    print(f"{size:>10} | {harris_time:>15.2f} | {simple_time:>15.2f}")

print("\n💡 실제 OpenCV FAST는 이보다 훨씬 빠릅니다!")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 70)
print("📚 Week 3 Quiz 정리")
print("=" * 70)

print("""
✅ 문제 1: NMS
   - 지역 최대값만 유지
   - 중복 검출 제거
   
✅ 문제 2: 파라미터 튜닝
   - threshold ↑ → 적은 검출, 강한 코너
   - threshold ↓ → 많은 검출, 약한 코너도
   
✅ 문제 3: 노이즈 강건성
   - 노이즈 ↑ → 거짓 검출 ↑
   - 전처리(가우시안) 도움
   
✅ 문제 4: 균일 분포
   - 최소 거리 제약 (min_dist)
   - VINS: 30 픽셀
   
✅ 문제 5: 속도 비교
   - FAST >> Harris
   - 실시간 SLAM = FAST

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 SLAM 파라미터 가이드:

| 상황 | threshold | min_dist | max_features |
|------|-----------|----------|--------------|
| 텍스처 풍부 | 높임 | 넓힘 | 줄임 |
| 텍스처 부족 | 낮춤 | 좁힘 | 늘림 |
| 빠른 움직임 | 낮춤 | - | 늘림 |
| 느린 움직임 | 높임 | - | 줄임 |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 다음: Week 4 - 특징점 매칭 (Brute-Force, FLANN, RANSAC)
""")

print("\n" + "=" * 70)
print("feature_detection_quiz.py 실행 완료! 🎉")
print("=" * 70)
print("\n생성된 파일:")
print("  1. feature_detection_comparison.png - Harris/FAST 비교")
print("  2. nms_comparison.png - NMS 전후")
print("  3. threshold_tuning.png - 파라미터 효과")
print("  4. noise_robustness.png - 노이즈 강건성")
print("  5. uniform_distribution.png - 균일 분포")
