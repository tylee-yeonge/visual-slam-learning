"""
Phase 2 - Week 4: 특징점 매칭 실습 문제
======================================
RANSAC, 성능 비교, Homography 추정

학습 목표:
1. RANSAC 원리 이해
2. 매칭 성능 평가
3. Homography 추정
4. outlier 제거 효과

실행 시간: 약 2분
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("=" * 70)
print("       Phase 2 - Week 4: 특징점 매칭 실습 문제")
print("=" * 70)
print("\n이 실습에서는 RANSAC과 Homography 추정을 배웁니다.\n")

# ============================================================
# 기본 함수
# ============================================================
def hamming_distance(a, b):
    return np.sum(a != b)

def brute_force_match(desc1, desc2):
    matches = []
    for i, d1 in enumerate(desc1):
        best_idx, best_dist = -1, float('inf')
        for j, d2 in enumerate(desc2):
            dist = hamming_distance(d1, d2)
            if dist < best_dist:
                best_dist, best_idx = dist, j
        matches.append((i, best_idx, best_dist))
    return matches

def knn_match(desc1, desc2, k=2):
    matches = []
    for i, d1 in enumerate(desc1):
        distances = [(j, hamming_distance(d1, d2)) for j, d2 in enumerate(desc2)]
        distances.sort(key=lambda x: x[1])
        matches.append((i, distances[:k]))
    return matches

def apply_ratio_test(knn_matches, ratio=0.75):
    good = []
    for qi, top_k in knn_matches:
        if len(top_k) >= 2 and top_k[1][1] > 0:
            if top_k[0][1] / top_k[1][1] < ratio:
                good.append((qi, top_k[0][0], top_k[0][1]))
    return good

# ============================================================
# 문제 1: Homography 이해
# ============================================================
print("\n" + "=" * 70)
print("문제 1: Homography (평면 변환)")
print("=" * 70)

print("""
🎯 Homography = 평면 → 평면 변환

예시:
- 포스터 촬영 각도 변화
- 바닥/벽면의 뷰 변화
- 지도/문서 스캔

수식:
    [u']   [h11 h12 h13] [u]
  s [v'] = [h21 h22 h23] [v]  
    [1 ]   [h31 h32 h33] [1]
    
    → 8 DOF (h33=1로 정규화)
    → 최소 4점 필요
""")

def apply_homography(H, points):
    """Homography 적용"""
    ones = np.ones((len(points), 1))
    pts_h = np.hstack([points, ones])  # 동차 좌표
    
    transformed = pts_h @ H.T
    transformed /= transformed[:, 2:3]  # 정규화
    
    return transformed[:, :2]

def create_homography(rotation=0, translation=(0,0), scale=1.0):
    """간단한 Homography 생성"""
    theta = np.radians(rotation)
    c, s = np.cos(theta), np.sin(theta)
    
    H = np.array([
        [scale * c, -scale * s, translation[0]],
        [scale * s,  scale * c, translation[1]],
        [0,          0,         1]
    ])
    return H

# Homography 테스트
H_test = create_homography(rotation=15, translation=(50, 30), scale=1.1)

print("\n예시 Homography:")
print(H_test)

pts1 = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
pts2 = apply_homography(H_test, pts1)

print(f"\n변환 전: \n{pts1}")
print(f"변환 후: \n{pts2}")

# ============================================================
# 문제 2: RANSAC 구현
# ============================================================
print("\n" + "=" * 70)
print("문제 2: RANSAC 구현")
print("=" * 70)

print("""
🎯 RANSAC = Random Sample Consensus

알고리즘:
1. 랜덤하게 최소 샘플 선택 (Homography: 4점)
2. 모델 추정
3. 모든 점에 모델 적용
4. inlier 개수 세기
5. 반복하여 최고 모델 선택
""")

def estimate_homography_dlt(src_pts, dst_pts):
    """
    4점으로 Homography 추정 (DLT)
    
    간단한 구현 - 실제로는 SVD 사용
    """
    if len(src_pts) != 4:
        return None
    
    A = []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    
    A = np.array(A)
    
    try:
        _, _, Vh = np.linalg.svd(A)
        h = Vh[-1]
        H = h.reshape(3, 3)
        return H / H[2, 2]
    except:
        return None

def ransac_homography(pts1, pts2, threshold=3.0, max_iters=1000):
    """
    RANSAC으로 Homography 추정
    
    Args:
        pts1, pts2: (N, 2) 대응점
        threshold: inlier 판단 임계값 (픽셀)
        max_iters: 최대 반복 횟수
    
    Returns:
        best_H: 최적 Homography
        inlier_mask: inlier 마스크
    """
    n_points = len(pts1)
    best_H = None
    best_inliers = 0
    best_mask = None
    
    for _ in range(max_iters):
        # 1. 랜덤 4점 선택
        indices = np.random.choice(n_points, 4, replace=False)
        src = pts1[indices]
        dst = pts2[indices]
        
        # 2. Homography 추정
        H = estimate_homography_dlt(src, dst)
        if H is None:
            continue
        
        # 3. 모든 점에 적용
        projected = apply_homography(H, pts1)
        
        # 4. inlier 판단
        errors = np.linalg.norm(projected - pts2, axis=1)
        inlier_mask = errors < threshold
        n_inliers = np.sum(inlier_mask)
        
        # 5. 최고 결과 저장
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_H = H
            best_mask = inlier_mask
    
    return best_H, best_mask

# 테스트 데이터 생성
def generate_matching_data(n_inliers=40, n_outliers=15, H=None):
    """매칭 데이터 생성 (inlier + outlier)"""
    if H is None:
        H = create_homography(rotation=10, translation=(30, 20), scale=1.05)
    
    # inlier: 정확한 대응
    pts1_inlier = np.random.rand(n_inliers, 2) * 200 + 50
    pts2_inlier = apply_homography(H, pts1_inlier)
    pts2_inlier += np.random.randn(n_inliers, 2) * 1.0  # 약간 노이즈
    
    # outlier: 잘못된 대응
    pts1_outlier = np.random.rand(n_outliers, 2) * 200 + 50
    pts2_outlier = np.random.rand(n_outliers, 2) * 200 + 50  # 랜덤
    
    pts1 = np.vstack([pts1_inlier, pts1_outlier])
    pts2 = np.vstack([pts2_inlier, pts2_outlier])
    
    # Ground truth
    gt_mask = np.zeros(n_inliers + n_outliers, dtype=bool)
    gt_mask[:n_inliers] = True
    
    return pts1, pts2, gt_mask, H

# 데이터 생성
pts1, pts2, gt_mask, H_true = generate_matching_data(n_inliers=40, n_outliers=15)

print(f"\n테스트 데이터:")
print(f"  총 매칭 수: {len(pts1)}")
print(f"  실제 inlier: {np.sum(gt_mask)}")
print(f"  실제 outlier: {np.sum(~gt_mask)}")

# RANSAC 실행
H_est, ransac_mask = ransac_homography(pts1, pts2, threshold=5.0, max_iters=500)

print(f"\nRANSAC 결과:")
print(f"  검출된 inlier: {np.sum(ransac_mask)}")
print(f"  True Positive: {np.sum(ransac_mask & gt_mask)}")
print(f"  False Positive: {np.sum(ransac_mask & ~gt_mask)}")

# ============================================================
# 문제 3: RANSAC 파라미터 영향
# ============================================================
print("\n" + "=" * 70)
print("문제 3: RANSAC 파라미터 영향")
print("=" * 70)

print("""
🎯 주요 파라미터:
1. threshold: inlier 판단 거리 (픽셀)
2. max_iters: 반복 횟수
3. min_samples: 샘플 크기 (Homography=4)
""")

# threshold 영향
print("\nThreshold에 따른 결과:")
print("-" * 50)
print(f"{'Threshold':>12} | {'Inliers':>10} | {'Precision':>12}")
print("-" * 50)

for thresh in [1.0, 3.0, 5.0, 10.0, 20.0]:
    _, mask = ransac_homography(pts1, pts2, threshold=thresh, max_iters=500)
    precision = np.sum(mask & gt_mask) / np.sum(mask) * 100 if np.sum(mask) > 0 else 0
    print(f"{thresh:>12.1f} | {np.sum(mask):>10} | {precision:>11.1f}%")

print("""
💡 관찰:
   - 낮은 threshold: 적은 inlier, 높은 정밀도
   - 높은 threshold: 많은 inlier, outlier 포함 가능
   - 보통 1~3 픽셀 권장
""")

# ============================================================
# 문제 4: 전체 파이프라인
# ============================================================
print("\n" + "=" * 70)
print("문제 4: 전체 매칭 파이프라인")
print("=" * 70)

print("""
🎯 파이프라인:
   특징점 → 디스크립터 → BF/KNN → Ratio Test → RANSAC → inlier
""")

# 시뮬레이션: 전체 파이프라인
n_features = 100

# 디스크립터 생성 (시뮬레이션)
desc1 = np.random.randint(0, 2, (n_features, 32))
desc2 = desc1.copy()

# 일부 변형 (노이즈)
for i in range(n_features):
    noise_bits = np.random.randint(0, 32, np.random.randint(0, 5))
    for b in noise_bits:
        desc2[i, b] = 1 - desc2[i, b]

# 인덱스 섞기
perm = np.random.permutation(n_features)
desc2 = desc2[perm]

# Ground truth
gt = {i: np.where(perm == i)[0][0] for i in range(n_features)}

# 단계별 결과
print("\n단계별 결과:")
print("-" * 60)

# 1. BF 매칭
bf_matches = brute_force_match(desc1, desc2)
bf_correct = sum(1 for i, j, _ in bf_matches if gt[i] == j)
print(f"1. BF 매칭: {len(bf_matches)} matches, {bf_correct} correct ({bf_correct/len(bf_matches)*100:.1f}%)")

# 2. KNN + Ratio Test
knn_matches = knn_match(desc1, desc2, k=2)
ratio_matches = apply_ratio_test(knn_matches, ratio=0.75)
ratio_correct = sum(1 for i, j, _ in ratio_matches if gt[i] == j)
print(f"2. Ratio Test: {len(ratio_matches)} matches, {ratio_correct} correct ({ratio_correct/len(ratio_matches)*100 if ratio_matches else 0:.1f}%)")

# 3. 기하학적 검증 (시뮬레이션)
# 실제로는 pts1, pts2 좌표로 RANSAC
ransac_result = int(len(ratio_matches) * 0.9)  # 시뮬레이션
print(f"3. After RANSAC (sim): ~{ransac_result} inliers")

# ============================================================
# 문제 5: 시각화
# ============================================================
print("\n" + "=" * 70)
print("문제 5: RANSAC 결과 시각화")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. 모든 대응점
ax1 = axes[0]
ax1.scatter(pts1[:, 0], pts1[:, 1], c='blue', s=30, label='Image 1')
ax1.scatter(pts2[:, 0] + 300, pts2[:, 1], c='red', s=30, label='Image 2')
for p1, p2 in zip(pts1, pts2):
    ax1.plot([p1[0], p2[0] + 300], [p1[1], p2[1]], 'gray', alpha=0.3, linewidth=0.5)
ax1.axvline(x=300, color='black', linestyle='--')
ax1.set_title(f'All Matches ({len(pts1)})', fontsize=11)
ax1.set_xlim([0, 600]); ax1.set_ylim([300, 0])
ax1.axis('off')

# 2. Ground Truth
ax2 = axes[1]
ax2.scatter(pts1[:, 0], pts1[:, 1], c='blue', s=30)
ax2.scatter(pts2[:, 0] + 300, pts2[:, 1], c='red', s=30)
for i, (p1, p2) in enumerate(zip(pts1, pts2)):
    color = 'green' if gt_mask[i] else 'red'
    ax2.plot([p1[0], p2[0] + 300], [p1[1], p2[1]], color, alpha=0.5, linewidth=1)
ax2.axvline(x=300, color='black', linestyle='--')
ax2.set_title(f'Ground Truth\n(Green=Inlier, Red=Outlier)', fontsize=11)
ax2.set_xlim([0, 600]); ax2.set_ylim([300, 0])
ax2.axis('off')

# 3. RANSAC 결과
ax3 = axes[2]
ax3.scatter(pts1[:, 0], pts1[:, 1], c='blue', s=30)
ax3.scatter(pts2[:, 0] + 300, pts2[:, 1], c='red', s=30)
for i, (p1, p2) in enumerate(zip(pts1, pts2)):
    if ransac_mask[i]:
        ax3.plot([p1[0], p2[0] + 300], [p1[1], p2[1]], 'green', alpha=0.7, linewidth=1)
ax3.axvline(x=300, color='black', linestyle='--')
ax3.set_title(f'RANSAC Inliers ({np.sum(ransac_mask)})', fontsize=11)
ax3.set_xlim([0, 600]); ax3.set_ylim([300, 0])
ax3.axis('off')

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week4/ransac_result.png', dpi=150)
print("\nRANSAC result saved: ransac_result.png")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 70)
print("📚 Week 4 Quiz 정리")
print("=" * 70)

print("""
✅ 문제 1: Homography
   - 평면 → 평면 변환
   - 8 DOF, 4점 필요

✅ 문제 2: RANSAC 구현
   - 랜덤 샘플 → 모델 → inlier 세기
   - 최고 모델 선택

✅ 문제 3: 파라미터 영향
   - threshold: 1~3 픽셀 권장
   - max_iters: 500~2000

✅ 문제 4: 전체 파이프라인
   - BF/KNN → Ratio Test → RANSAC
   - 단계별 필터링

✅ 문제 5: 시각화
   - inlier/outlier 구분
   - RANSAC 효과 확인

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 OpenCV 사용법:

```python
# Homography 추정 with RANSAC
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

# Fundamental Matrix with RANSAC  
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 다음: Week 5 - 에피폴라 기하학 (Essential/Fundamental Matrix)
""")

print("\n" + "=" * 70)
print("matching_quiz.py 실행 완료! 🎉")
print("=" * 70)
print("\n생성된 파일:")
print("  1. matching_comparison.png - 매칭 필터링 비교")
print("  2. ransac_result.png - RANSAC 결과")
