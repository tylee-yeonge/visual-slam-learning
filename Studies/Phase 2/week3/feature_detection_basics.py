"""
Phase 2 - Week 3: 특징점 검출 기초
==================================
Harris, FAST, ORB 구현 및 비교

학습 목표:
1. Harris 코너 검출 이해
2. FAST 알고리즘 이해
3. ORB 디스크립터 이해
4. OpenCV로 특징점 검출

실행 시간: 약 1분
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

print("=" * 70)
print("        Phase 2 - Week 3: 특징점 검출 기초")
print("=" * 70)
print("\n💡 이 실습에서는 이미지에서 특징점을 검출하는 방법을 배웁니다.\n")

# ============================================================
# Part 1: 테스트 이미지 생성
# ============================================================
print("\n" + "=" * 70)
print("Part 1: 테스트 이미지 생성")
print("=" * 70)

def create_test_image(size=200):
    """특징점 검출 테스트용 이미지 생성"""
    img = np.ones((size, size), dtype=np.float32) * 128
    
    # 사각형 (코너 4개)
    img[40:80, 40:100] = 200
    
    # 삼각형
    for i in range(40):
        img[100+i, 120-i:120+i+1] = 50
    
    # 원
    y, x = np.ogrid[:size, :size]
    center = (150, 60)
    radius = 25
    mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    img[mask] = 220
    
    # 체스보드 패턴
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                img[130+i*15:130+(i+1)*15, 130+j*15:130+(j+1)*15] = 30
    
    return img

test_image = create_test_image()

print("테스트 이미지 생성 완료!")
print(f"  크기: {test_image.shape}")
print(f"  포함 도형: 사각형, 삼각형, 원, 체스보드")

# ============================================================
# Part 2: Harris 코너 검출
# ============================================================
print("\n" + "=" * 70)
print("Part 2: Harris 코너 검출 구현")
print("=" * 70)

print("""
🎯 Harris Corner Detector

핵심 아이디어:
- 윈도우를 이동시킬 때 모든 방향으로 밝기 변화 → 코너!
- Structure Tensor M의 고유값으로 판단

수식:
    R = det(M) - k * trace(M)²
    
    R > threshold → 코너
""")

def harris_corner_detector(image, k=0.04, window_size=3, threshold=0.01):
    """
    Harris 코너 검출 구현
    
    Args:
        image: 그레이스케일 이미지
        k: Harris 파라미터 (0.04~0.06)
        window_size: 스무딩 윈도우 크기
        threshold: 응답 임계값 (비율)
    
    Returns:
        R: 코너 응답 맵
        corners: 코너 좌표 [(x, y), ...]
    """
    # 그래디언트 계산 (Sobel)
    Ix = np.zeros_like(image)
    Iy = np.zeros_like(image)
    
    # 간단한 Sobel 필터
    Ix[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2
    Iy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2
    
    # Structure Tensor 요소
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    # 가우시안 스무딩 (간단 버전: 박스 필터)
    from scipy.ndimage import uniform_filter
    Sxx = uniform_filter(Ixx, size=window_size)
    Syy = uniform_filter(Iyy, size=window_size)
    Sxy = uniform_filter(Ixy, size=window_size)
    
    # Harris 응답
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    R = det - k * trace * trace
    
    # 임계값 적용
    R_normalized = R / (R.max() + 1e-10)
    corner_mask = R_normalized > threshold
    
    # 좌표 추출
    corners = np.argwhere(corner_mask)
    corners = [(c[1], c[0]) for c in corners]  # (x, y) 형식
    
    return R, corners

# Harris 검출 실행
R_harris, harris_corners = harris_corner_detector(test_image, threshold=0.1)

print(f"\nHarris 검출 결과:")
print(f"  검출된 코너 수: {len(harris_corners)}")
print(f"  응답 범위: [{R_harris.min():.2f}, {R_harris.max():.2f}]")

# ============================================================
# Part 3: FAST 코너 검출 구현
# ============================================================
print("\n" + "=" * 70)
print("Part 3: FAST 코너 검출 구현")
print("=" * 70)

print("""
🎯 FAST (Features from Accelerated Segment Test)

핵심 아이디어:
- 중심 픽셀 주위 16개 픽셀 검사
- 연속 N개(보통 9~12)가 모두 밝거나 어두우면 → 코너!

장점: 매우 빠름 (VINS에서 사용)
""")

def fast_corner_detector(image, threshold=20, n_contiguous=9):
    """
    간단한 FAST 코너 검출 구현
    
    Args:
        image: 그레이스케일 이미지
        threshold: 밝기 차이 임계값
        n_contiguous: 연속해야 하는 픽셀 수
    
    Returns:
        corners: 코너 좌표 [(x, y), ...]
    """
    # Bresenham 원 상의 16픽셀 오프셋
    circle_offsets = [
        (0, -3), (1, -3), (2, -2), (3, -1),
        (3, 0), (3, 1), (2, 2), (1, 3),
        (0, 3), (-1, 3), (-2, 2), (-3, 1),
        (-3, 0), (-3, -1), (-2, -2), (-1, -3)
    ]
    
    corners = []
    h, w = image.shape
    
    for y in range(3, h - 3):
        for x in range(3, w - 3):
            center = float(image[y, x])
            
            # 16픽셀 밝기 수집
            circle_values = []
            for dx, dy in circle_offsets:
                circle_values.append(float(image[y + dy, x + dx]))
            
            # 밝은지/어두운지 판단
            brighter = [v > center + threshold for v in circle_values]
            darker = [v < center - threshold for v in circle_values]
            
            # 연속 N개 체크 (원형이므로 2배로 확장)
            brighter_ext = brighter + brighter
            darker_ext = darker + darker
            
            is_corner = False
            
            # 연속 n_contiguous개 밝은 픽셀?
            for i in range(16):
                if all(brighter_ext[i:i + n_contiguous]):
                    is_corner = True
                    break
            
            # 연속 n_contiguous개 어두운 픽셀?
            if not is_corner:
                for i in range(16):
                    if all(darker_ext[i:i + n_contiguous]):
                        is_corner = True
                        break
            
            if is_corner:
                corners.append((x, y))
    
    return corners

# FAST 검출 실행 (간단 버전이라 느릴 수 있음)
print("\nFAST 검출 중... (시뮬레이션)")
fast_corners = fast_corner_detector(test_image, threshold=30, n_contiguous=9)

print(f"FAST 검출 결과:")
print(f"  검출된 코너 수: {len(fast_corners)}")

# ============================================================
# Part 4: 디스크립터 개념
# ============================================================
print("\n" + "=" * 70)
print("Part 4: 디스크립터 개념")
print("=" * 70)

print("""
🎯 디스크립터 (Descriptor)

특징점 위치만으로는 매칭 불가!
→ 주변 패턴을 숫자(벡터)로 표현

BRIEF 디스크립터:
- 이진 벡터 (0/1)
- 빠른 계산 & 매칭

ORB = FAST + BRIEF + 회전 불변성
""")

def compute_simple_descriptor(image, keypoint, patch_size=7):
    """
    간단한 디스크립터 계산 (학습용)
    
    실제로는 BRIEF, ORB 등 사용
    """
    x, y = keypoint
    half = patch_size // 2
    
    h, w = image.shape
    if x < half or x >= w - half or y < half or y >= h - half:
        return None
    
    # 패치 추출
    patch = image[y - half:y + half + 1, x - half:x + half + 1]
    
    # 간단한 설명자: 패치 정규화
    desc = patch.flatten()
    desc = (desc - desc.mean()) / (desc.std() + 1e-10)
    
    return desc

# 예시 디스크립터 계산
if harris_corners:
    sample_point = harris_corners[0]
    sample_desc = compute_simple_descriptor(test_image, sample_point)
    
    print(f"\n샘플 디스크립터 (점 {sample_point}):")
    if sample_desc is not None:
        print(f"  차원: {len(sample_desc)}")
        print(f"  값 범위: [{sample_desc.min():.2f}, {sample_desc.max():.2f}]")
        print(f"  처음 10개: {sample_desc[:10]}")

# ============================================================
# Part 5: 시각화
# ============================================================
print("\n" + "=" * 70)
print("Part 5: 시각화")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# 원본 이미지
ax1 = axes[0, 0]
ax1.imshow(test_image, cmap='gray')
ax1.set_title('Original Image', fontsize=12)
ax1.axis('off')

# Harris 응답 맵
ax2 = axes[0, 1]
im = ax2.imshow(R_harris, cmap='hot')
ax2.set_title('Harris Response Map', fontsize=12)
plt.colorbar(im, ax=ax2, fraction=0.046)
ax2.axis('off')

# Harris 코너
ax3 = axes[1, 0]
ax3.imshow(test_image, cmap='gray')
for x, y in harris_corners[:50]:  # 최대 50개
    ax3.plot(x, y, 'r.', markersize=5)
ax3.set_title(f'Harris Corners ({len(harris_corners)} detected)', fontsize=12)
ax3.axis('off')

# FAST 코너
ax4 = axes[1, 1]
ax4.imshow(test_image, cmap='gray')
for x, y in fast_corners[:50]:  # 최대 50개
    ax4.plot(x, y, 'g.', markersize=5)
ax4.set_title(f'FAST Corners ({len(fast_corners)} detected)', fontsize=12)
ax4.axis('off')

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week3/feature_detection_comparison.png', dpi=150)
print("\nVisualization saved: feature_detection_comparison.png")

# ============================================================
# Part 6: OpenCV 사용법
# ============================================================
print("\n" + "=" * 70)
print("Part 6: OpenCV 사용법 (참고)")
print("=" * 70)

print("""
📖 OpenCV 특징점 검출 예시:

```python
import cv2

# 이미지 로드
gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# === FAST ===
fast = cv2.FastFeatureDetector_create(threshold=20)
keypoints_fast = fast.detect(gray)

# === ORB (FAST + BRIEF) ===
orb = cv2.ORB_create(nfeatures=500)
keypoints_orb, descriptors = orb.detectAndCompute(gray, None)

# === Harris ===
harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# 결과 시각화
img_with_kp = cv2.drawKeypoints(gray, keypoints_fast, None, 
                                 color=(0, 255, 0))
cv2.imshow('Features', img_with_kp)
```

파라미터:
- FAST threshold: 높이면 적은 검출, 낮추면 많은 검출
- ORB nfeatures: 검출할 최대 특징점 수
- Harris k: 보통 0.04~0.06
""")

# ============================================================
# Part 7: SLAM에서의 활용
# ============================================================
print("\n" + "=" * 70)
print("Part 7: SLAM에서의 활용")
print("=" * 70)

print("""
💡 VINS-Fusion feature_tracker:

1. 새 프레임 수신
2. FAST로 특징점 검출
3. 기존 특징점을 Lucas-Kanade로 추적
4. 추적 실패한 점 제거
5. 특징점 개수 유지 위해 새로 검출

핵심 파라미터 (VINS config):
```yaml
max_cnt: 150          # 최대 특징점 수
min_dist: 30          # 특징점 간 최소 거리
F_threshold: 1.0      # FAST 임계값
```

💡 ORB-SLAM3:

1. 새 프레임 수신
2. ORB로 특징점 + 디스크립터 계산
3. 기존 맵 포인트와 디스크립터 매칭
4. PnP로 포즈 추정
5. 새 맵 포인트 생성
""")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 70)
print("📚 Week 3 Basics 정리")
print("=" * 70)

print("""
✅ Part 1-2: Harris Corner
   - Structure Tensor의 고유값 분석
   - R = det(M) - k·trace(M)²

✅ Part 3: FAST
   - 16픽셀 원에서 연속 N개 검사
   - 매우 빠름 → VINS 사용

✅ Part 4: 디스크립터
   - 특징점 주변 패턴을 벡터로
   - BRIEF: 이진, ORB: FAST+BRIEF

✅ Part 5-6: 시각화 & OpenCV
   - cv2.FastFeatureDetector_create()
   - cv2.ORB_create()

✅ Part 7: SLAM 활용
   - VINS: FAST + KLT
   - ORB-SLAM: ORB + 매칭

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 핵심 메시지:
   특징점 = SLAM의 눈
   빠른 검출(FAST) + 고유 표현(디스크립터) = 추적/매칭 가능!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 다음: feature_detection_quiz.py → Week 4: 특징점 매칭
""")

print("\n" + "=" * 70)
print("feature_detection_basics.py 실행 완료! 🎉")
print("=" * 70)
