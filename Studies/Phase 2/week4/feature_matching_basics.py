"""
Phase 2 - Week 4: 특징점 매칭 기초
==================================
Brute-Force, Ratio Test, 매칭 시각화

학습 목표:
1. 디스크립터 비교 이해
2. Brute-Force 매칭 구현
3. Ratio Test 적용
4. 매칭 결과 분석

실행 시간: 약 1분
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("=" * 70)
print("        Phase 2 - Week 4: 특징점 매칭 기초")
print("=" * 70)
print("\n💡 이 실습에서는 특징점 매칭의 기본을 배웁니다.\n")

# ============================================================
# Part 1: 디스크립터와 거리 함수
# ============================================================
print("\n" + "=" * 70)
print("Part 1: 디스크립터와 거리 함수")
print("=" * 70)

print("""
🎯 디스크립터 = 특징점 주변 패턴을 숫자로 표현

거리 함수:
- 이진 디스크립터 (ORB/BRIEF): 해밍 거리 (XOR)
- 실수 디스크립터 (SIFT): 유클리드 거리 (L2)
""")

def hamming_distance(a, b):
    """해밍 거리: 다른 비트 수"""
    return np.sum(a != b)

def euclidean_distance(a, b):
    """유클리드 거리: L2 norm"""
    return np.sqrt(np.sum((a - b) ** 2))

# 이진 디스크립터 예시
desc_a = np.array([1, 0, 1, 1, 0, 1, 0, 0])
desc_b = np.array([1, 0, 0, 1, 0, 1, 0, 1])
desc_c = np.array([0, 1, 0, 0, 1, 0, 1, 1])

print("\n이진 디스크립터 예시 (8비트):")
print(f"  A = {desc_a}")
print(f"  B = {desc_b}")
print(f"  C = {desc_c}")
print(f"\n해밍 거리:")
print(f"  d(A, B) = {hamming_distance(desc_a, desc_b)}  (비슷)")
print(f"  d(A, C) = {hamming_distance(desc_a, desc_c)}  (다름)")

# 실수 디스크립터 예시
desc_float_a = np.array([0.2, 0.8, 0.1, 0.5])
desc_float_b = np.array([0.3, 0.7, 0.2, 0.4])
desc_float_c = np.array([0.9, 0.1, 0.8, 0.2])

print(f"\n실수 디스크립터 예시 (4D):")
print(f"  A = {desc_float_a}")
print(f"  B = {desc_float_b}")
print(f"  C = {desc_float_c}")
print(f"\n유클리드 거리:")
print(f"  d(A, B) = {euclidean_distance(desc_float_a, desc_float_b):.4f}  (비슷)")
print(f"  d(A, C) = {euclidean_distance(desc_float_a, desc_float_c):.4f}  (다름)")

# ============================================================
# Part 2: Brute-Force 매칭 구현
# ============================================================
print("\n" + "=" * 70)
print("Part 2: Brute-Force 매칭 구현")
print("=" * 70)

print("""
🎯 Brute-Force: 모든 쌍 비교하여 가장 가까운 것 찾기

복잡도: O(N × M)
- N: 이미지 1의 특징점 수
- M: 이미지 2의 특징점 수
""")

def brute_force_match(desc1, desc2, distance_fn='hamming'):
    """
    Brute-Force 매칭
    
    Args:
        desc1: (N, D) 첫 번째 이미지 디스크립터
        desc2: (M, D) 두 번째 이미지 디스크립터
        distance_fn: 'hamming' 또는 'euclidean'
    
    Returns:
        matches: [(idx1, idx2, distance), ...]
    """
    dist_func = hamming_distance if distance_fn == 'hamming' else euclidean_distance
    
    matches = []
    for i, d1 in enumerate(desc1):
        best_idx = -1
        best_dist = float('inf')
        
        for j, d2 in enumerate(desc2):
            dist = dist_func(d1, d2)
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        
        matches.append((i, best_idx, best_dist))
    
    return matches

# 시뮬레이션용 디스크립터 생성
def generate_descriptors(n_features, dim=32, binary=True):
    """랜덤 디스크립터 생성"""
    if binary:
        return np.random.randint(0, 2, (n_features, dim))
    else:
        return np.random.randn(n_features, dim)

def generate_matching_pair(n_features, dim=32, noise_level=3, outlier_ratio=0.2):
    """
    매칭 쌍 시뮬레이션
    - 일부는 정상 매칭 (inlier)
    - 일부는 노이즈로 인한 변형
    - 일부는 잘못된 매칭 (outlier)
    """
    # 원본 디스크립터
    desc1 = np.random.randint(0, 2, (n_features, dim))
    
    # 변환된 디스크립터 (노이즈 추가)
    n_inliers = int(n_features * (1 - outlier_ratio))
    n_outliers = n_features - n_inliers
    
    # inlier: 약간 변형
    desc2_inlier = desc1[:n_inliers].copy()
    for i in range(n_inliers):
        noise_bits = np.random.randint(0, dim, noise_level)
        for bit in noise_bits:
            desc2_inlier[i, bit] = 1 - desc2_inlier[i, bit]  # 비트 플립
    
    # outlier: 완전히 다른 디스크립터
    desc2_outlier = np.random.randint(0, 2, (n_outliers, dim))
    
    # 섞기
    desc2 = np.vstack([desc2_inlier, desc2_outlier])
    shuffle_idx = np.random.permutation(n_features)
    desc2 = desc2[shuffle_idx]
    
    # Ground truth: 처음 n_inliers개는 대응점 있음
    ground_truth = {}
    for i in range(n_inliers):
        new_idx = np.where(shuffle_idx == i)[0][0]
        ground_truth[i] = new_idx
    
    return desc1, desc2, ground_truth

# 테스트
n_features = 50
desc1, desc2, gt = generate_matching_pair(n_features, dim=32, noise_level=3, outlier_ratio=0.2)

print(f"\n시뮬레이션 설정:")
print(f"  특징점 수: {n_features}")
print(f"  디스크립터 차원: 32 (binary)")
print(f"  outlier 비율: 20%")

# Brute-Force 매칭
matches = brute_force_match(desc1, desc2, 'hamming')

# 정확도 계산
correct = sum(1 for i, j, _ in matches if i in gt and gt[i] == j)
accuracy = correct / len(gt) * 100

print(f"\nBrute-Force 매칭 결과:")
print(f"  총 매칭 수: {len(matches)}")
print(f"  정답 수: {correct} / {len(gt)}")
print(f"  정확도: {accuracy:.1f}%")

# ============================================================
# Part 3: KNN 매칭과 Ratio Test
# ============================================================
print("\n" + "=" * 70)
print("Part 3: KNN 매칭과 Ratio Test")
print("=" * 70)

print("""
🎯 Lowe's Ratio Test

문제: 가장 가까운 점이 진짜 매칭인지 확신 불가
해결: 1순위와 2순위 거리 비교

ratio = 1순위 거리 / 2순위 거리

ratio < 0.75 → 수락 (1순위가 확연히 가까움)
ratio >= 0.75 → 거부 (모호함)
""")

def knn_match(desc1, desc2, k=2, distance_fn='hamming'):
    """
    KNN 매칭 (상위 k개 반환)
    """
    dist_func = hamming_distance if distance_fn == 'hamming' else euclidean_distance
    
    matches = []
    for i, d1 in enumerate(desc1):
        distances = []
        for j, d2 in enumerate(desc2):
            dist = dist_func(d1, d2)
            distances.append((j, dist))
        
        # 거리순 정렬
        distances.sort(key=lambda x: x[1])
        top_k = distances[:k]
        matches.append((i, top_k))
    
    return matches

def apply_ratio_test(knn_matches, ratio=0.75):
    """Ratio Test 적용"""
    good_matches = []
    
    for query_idx, top_k in knn_matches:
        if len(top_k) < 2:
            continue
        
        best_idx, best_dist = top_k[0]
        second_idx, second_dist = top_k[1]
        
        # Ratio Test
        if second_dist > 0 and best_dist / second_dist < ratio:
            good_matches.append((query_idx, best_idx, best_dist))
    
    return good_matches

# KNN 매칭
knn_matches = knn_match(desc1, desc2, k=2)

# 다양한 ratio로 테스트
print("\nRatio 값에 따른 결과:")
print("-" * 60)
print(f"{'Ratio':>8} | {'Matches':>10} | {'Correct':>10} | {'Precision':>12}")
print("-" * 60)

for ratio in [0.6, 0.7, 0.75, 0.8, 0.9, 1.0]:
    good = apply_ratio_test(knn_matches, ratio)
    correct = sum(1 for i, j, _ in good if i in gt and gt[i] == j)
    precision = correct / len(good) * 100 if good else 0
    print(f"{ratio:>8.2f} | {len(good):>10} | {correct:>10} | {precision:>11.1f}%")

print("""
💡 관찰:
   - 낮은 ratio: 적은 매칭, 높은 정밀도
   - 높은 ratio: 많은 매칭, 낮은 정밀도
   - 0.75: 좋은 균형점 (Lowe 권장)
""")

# ============================================================
# Part 4: Cross-check 매칭
# ============================================================
print("\n" + "=" * 70)
print("Part 4: Cross-check 매칭")
print("=" * 70)

print("""
🎯 Cross-check: 양방향 확인

A → B (A에서 B가 가장 가까움)
B → A (B에서 A가 가장 가까움)

둘 다 만족해야 매칭!
""")

def cross_check_match(desc1, desc2, distance_fn='hamming'):
    """Cross-check 매칭"""
    # 양방향 매칭
    matches_1to2 = brute_force_match(desc1, desc2, distance_fn)
    matches_2to1 = brute_force_match(desc2, desc1, distance_fn)
    
    # Cross-check
    good_matches = []
    for i, j, dist in matches_1to2:
        # desc1[i]의 최선이 desc2[j]이고
        # desc2[j]의 최선이 desc1[i]인지 확인
        if matches_2to1[j][1] == i:
            good_matches.append((i, j, dist))
    
    return good_matches

cross_matches = cross_check_match(desc1, desc2)
correct_cross = sum(1 for i, j, _ in cross_matches if i in gt and gt[i] == j)

print(f"\nCross-check 결과:")
print(f"  BF 매칭: {len(matches)} → Cross-check: {len(cross_matches)}")
print(f"  정확도: {correct_cross / len(cross_matches) * 100:.1f}%")

# ============================================================
# Part 5: 매칭 시각화
# ============================================================
print("\n" + "=" * 70)
print("Part 5: 매칭 시각화")
print("=" * 70)

# 가상 이미지에서 매칭 시각화
def create_keypoints(n_points, image_size=(200, 200)):
    """가상 키포인트 생성"""
    h, w = image_size
    kp = np.random.rand(n_points, 2)
    kp[:, 0] *= w
    kp[:, 1] *= h
    return kp

# 두 이미지의 키포인트
kp1 = create_keypoints(n_features)
kp2 = create_keypoints(n_features)

# Ground truth 매칭에서 키포인트 이동
for i, j in gt.items():
    kp2[j] = kp1[i] + np.random.randn(2) * 10  # 약간 이동

# Ratio Test로 좋은 매칭 선택
ratio_matches = apply_ratio_test(knn_matches, 0.75)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. 모든 매칭
ax1 = axes[0]
ax1.set_xlim([0, 400]); ax1.set_ylim([200, 0])
ax1.scatter(kp1[:, 0], kp1[:, 1], c='blue', s=20, label='Image 1')
ax1.scatter(kp2[:, 0] + 200, kp2[:, 1], c='red', s=20, label='Image 2')
for i, j, _ in matches[:30]:  # 처음 30개만
    ax1.plot([kp1[i, 0], kp2[j, 0] + 200], [kp1[i, 1], kp2[j, 1]], 
             'g-', alpha=0.3, linewidth=0.5)
ax1.axvline(x=200, color='gray', linestyle='--')
ax1.set_title(f'All BF Matches ({len(matches)})', fontsize=11)
ax1.axis('off')

# 2. Ratio Test 후
ax2 = axes[1]
ax2.set_xlim([0, 400]); ax2.set_ylim([200, 0])
ax2.scatter(kp1[:, 0], kp1[:, 1], c='blue', s=20)
ax2.scatter(kp2[:, 0] + 200, kp2[:, 1], c='red', s=20)
for i, j, _ in ratio_matches[:30]:
    color = 'g' if (i in gt and gt[i] == j) else 'r'
    ax2.plot([kp1[i, 0], kp2[j, 0] + 200], [kp1[i, 1], kp2[j, 1]], 
             f'{color}-', alpha=0.5, linewidth=1)
ax2.axvline(x=200, color='gray', linestyle='--')
ax2.set_title(f'After Ratio Test ({len(ratio_matches)})', fontsize=11)
ax2.axis('off')

# 3. Cross-check 후
ax3 = axes[2]
ax3.set_xlim([0, 400]); ax3.set_ylim([200, 0])
ax3.scatter(kp1[:, 0], kp1[:, 1], c='blue', s=20)
ax3.scatter(kp2[:, 0] + 200, kp2[:, 1], c='red', s=20)
for i, j, _ in cross_matches[:30]:
    color = 'g' if (i in gt and gt[i] == j) else 'r'
    ax3.plot([kp1[i, 0], kp2[j, 0] + 200], [kp1[i, 1], kp2[j, 1]], 
             f'{color}-', alpha=0.5, linewidth=1)
ax3.axvline(x=200, color='gray', linestyle='--')
ax3.set_title(f'After Cross-check ({len(cross_matches)})', fontsize=11)
ax3.axis('off')

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week4/matching_comparison.png', dpi=150)
print("\nMatching comparison saved: matching_comparison.png")
print("  Green = correct match, Red = incorrect match")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 70)
print("📚 Week 4 Basics 정리")
print("=" * 70)

print("""
✅ Part 1: 거리 함수
   - 해밍 거리: 이진 디스크립터 (ORB)
   - 유클리드 거리: 실수 디스크립터 (SIFT)

✅ Part 2: Brute-Force
   - 모든 쌍 비교
   - 정확하지만 느림

✅ Part 3: Ratio Test
   - 모호한 매칭 제거
   - ratio = 0.75 권장

✅ Part 4: Cross-check
   - 양방향 확인
   - 잘못된 매칭 제거

✅ Part 5: 시각화
   - 매칭 선 그리기
   - 필터링 효과 확인

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 핵심 메시지:
   단순 매칭 → Ratio Test → Cross-check/RANSAC
   필터링으로 신뢰도 향상!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 다음: matching_quiz.py → Week 5: 에피폴라 기하학
""")

print("\n" + "=" * 70)
print("feature_matching_basics.py 실행 완료! 🎉")
print("=" * 70)
