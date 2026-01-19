"""
Phase 2 - Week 8: 광류 기초
===========================
Lucas-Kanade 구현, 특징점 추적

학습 목표:
1. 밝기 항상성 이해
2. Lucas-Kanade 원리
3. 특징점 추적 구현
4. 피라미드 LK 이해

실행 시간: 약 1분
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("=" * 70)
print("        Phase 2 - Week 8: 광류 기초")
print("=" * 70)
print("\n💡 이 실습에서는 Lucas-Kanade 광류를 배웁니다.\n")

# ============================================================
# Part 1: 밝기 항상성 가정
# ============================================================
print("\n" + "=" * 70)
print("Part 1: 밝기 항상성 가정")
print("=" * 70)

print("""
🎯 밝기 항상성 (Brightness Constancy):

I(x, y, t) = I(x+Δx, y+Δy, t+Δt)

테일러 전개 후:
Iₓ·u + Iᵧ·v + Iₜ = 0

여기서:
- Iₓ, Iᵧ: 공간 그래디언트
- Iₜ: 시간 그래디언트
- u, v: 광류 (구하고자 하는 값)
""")

# 간단한 테스트 이미지
def create_moving_dot(size=100, center=(50, 50), radius=10):
    """움직이는 점 이미지 생성"""
    img = np.zeros((size, size), dtype=np.float32)
    y, x = np.ogrid[:size, :size]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    img[mask] = 255
    return img

# 두 프레임 생성
frame1 = create_moving_dot(100, (45, 50))  # 원래 위치
frame2 = create_moving_dot(100, (55, 53))  # 오른쪽+아래로 이동

print("테스트 이미지 생성:")
print(f"  프레임 1: 중심 (45, 50)")
print(f"  프레임 2: 중심 (55, 53)")
print(f"  실제 이동: Δx=10, Δy=3")

# ============================================================
# Part 2: 그래디언트 계산
# ============================================================
print("\n" + "=" * 70)
print("Part 2: 그래디언트 계산")
print("=" * 70)

print("""
🎯 그래디언트:
- Iₓ: x 방향 미분 (Sobel)
- Iᵧ: y 방향 미분 (Sobel)
- Iₜ: 시간 미분 (frame2 - frame1)
""")

def compute_gradients(img1, img2):
    """이미지 그래디언트 계산"""
    # Sobel 커널
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8
    
    # 공간 그래디언트 (두 프레임 평균)
    Ix = (convolve(img1, kx) + convolve(img2, kx)) / 2
    Iy = (convolve(img1, ky) + convolve(img2, ky)) / 2
    
    # 시간 그래디언트
    It = img2.astype(np.float32) - img1.astype(np.float32)
    
    return Ix, Iy, It

Ix, Iy, It = compute_gradients(frame1, frame2)

print(f"Ix 범위: [{Ix.min():.2f}, {Ix.max():.2f}]")
print(f"Iy 범위: [{Iy.min():.2f}, {Iy.max():.2f}]")
print(f"It 범위: [{It.min():.2f}, {It.max():.2f}]")

# ============================================================
# Part 3: Lucas-Kanade 구현
# ============================================================
print("\n" + "=" * 70)
print("Part 3: Lucas-Kanade 구현")
print("=" * 70)

print("""
🎯 Lucas-Kanade:

윈도우 내 모든 픽셀이 같은 광류를 가정

A·[u,v]ᵀ = b

(AᵀA)·[u,v]ᵀ = Aᵀb

해: [u,v]ᵀ = (AᵀA)⁻¹·Aᵀb
""")

def lucas_kanade_point(Ix, Iy, It, point, window_size=21):
    """
    단일 점에서 Lucas-Kanade 광류 계산
    
    Args:
        Ix, Iy, It: 그래디언트 이미지
        point: (x, y) 추적할 점
        window_size: 윈도우 크기
    
    Returns:
        (u, v): 광류 벡터
    """
    x, y = int(point[0]), int(point[1])
    half_w = window_size // 2
    
    # 윈도우 영역 추출
    y_min = max(0, y - half_w)
    y_max = min(Ix.shape[0], y + half_w + 1)
    x_min = max(0, x - half_w)
    x_max = min(Ix.shape[1], x + half_w + 1)
    
    Ix_win = Ix[y_min:y_max, x_min:x_max].flatten()
    Iy_win = Iy[y_min:y_max, x_min:x_max].flatten()
    It_win = It[y_min:y_max, x_min:x_max].flatten()
    
    # A 행렬
    A = np.column_stack([Ix_win, Iy_win])
    b = -It_win
    
    # AᵀA
    AtA = A.T @ A
    Atb = A.T @ b
    
    # 해 구하기
    try:
        # 고유값 확인 (추적 가능성)
        eigvals = np.linalg.eigvalsh(AtA)
        if np.min(eigvals) < 1e-6:
            return (0, 0), False  # 추적 불가
        
        flow = np.linalg.solve(AtA, Atb)
        return (flow[0], flow[1]), True
    except:
        return (0, 0), False

# 중심점에서 광류 계산
center_point = (50, 50)
(u, v), success = lucas_kanade_point(Ix, Iy, It, center_point, window_size=31)

print(f"\n중심점 ({center_point}) 광류:")
print(f"  추정: u={u:.2f}, v={v:.2f}")
print(f"  실제: u=10, v=3")
print(f"  추적 성공: {success}")

# ============================================================
# Part 4: 여러 점 추적
# ============================================================
print("\n" + "=" * 70)
print("Part 4: 여러 점 추적")
print("=" * 70)

def track_points(img1, img2, points, window_size=21):
    """여러 점 추적"""
    Ix, Iy, It = compute_gradients(img1, img2)
    
    tracked_points = []
    for pt in points:
        (u, v), success = lucas_kanade_point(Ix, Iy, It, pt, window_size)
        if success:
            new_pt = (pt[0] + u, pt[1] + v)
            tracked_points.append((pt, new_pt, (u, v)))
    
    return tracked_points

# 여러 테스트 점
test_points = [
    (45, 50),  # 원 중심 근처
    (40, 50),
    (50, 45),
    (45, 55),
]

results = track_points(frame1, frame2, test_points, window_size=31)

print(f"추적 결과 ({len(results)}/{len(test_points)} 성공):")
for orig, new, flow in results:
    print(f"  {orig} → ({new[0]:.1f}, {new[1]:.1f}), flow=({flow[0]:.2f}, {flow[1]:.2f})")

# ============================================================
# Part 5: Structure Tensor와 추적 품질
# ============================================================
print("\n" + "=" * 70)
print("Part 5: Structure Tensor와 추적 품질")
print("=" * 70)

print("""
🎯 Structure Tensor (AᵀA):

    ⎡ ΣIₓ²    ΣIₓIᵧ ⎤
M = ⎢               ⎥
    ⎣ ΣIₓIᵧ  ΣIᵧ²  ⎦

이것은 Harris 코너의 M과 같음!

고유값 분석:
- λ₁, λ₂ 둘 다 큼 → 코너 → 추적 좋음
- 하나만 큼 → 에지 → 조리개 문제
- 둘 다 작음 → 플랫 → 추적 불가
""")

def compute_trackability(Ix, Iy, point, window_size=21):
    """추적 품질 분석"""
    x, y = int(point[0]), int(point[1])
    half_w = window_size // 2
    
    y_min = max(0, y - half_w)
    y_max = min(Ix.shape[0], y + half_w + 1)
    x_min = max(0, x - half_w)
    x_max = min(Ix.shape[1], x + half_w + 1)
    
    Ix_win = Ix[y_min:y_max, x_min:x_max].flatten()
    Iy_win = Iy[y_min:y_max, x_min:x_max].flatten()
    
    # Structure Tensor
    M = np.array([
        [np.sum(Ix_win**2), np.sum(Ix_win * Iy_win)],
        [np.sum(Ix_win * Iy_win), np.sum(Iy_win**2)]
    ])
    
    eigvals = np.linalg.eigvalsh(M)
    min_eig = np.min(eigvals)
    
    if min_eig > 1000:
        quality = "Good (Corner)"
    elif min_eig > 100:
        quality = "Fair (Edge)"
    else:
        quality = "Poor (Flat)"
    
    return min_eig, quality

# 다양한 위치에서 추적 품질
test_locations = [
    (50, 50, "Center of dot"),
    (45, 50, "Edge of dot"),
    (20, 20, "Background"),
]

print("\n추적 품질 분석:")
print("-" * 50)
for x, y, desc in test_locations:
    min_eig, quality = compute_trackability(Ix, Iy, (x, y))
    print(f"  ({x}, {y}) {desc}: λ_min={min_eig:.1f}, {quality}")

# ============================================================
# Part 6: 시각화
# ============================================================
print("\n" + "=" * 70)
print("Part 6: 시각화")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

# 프레임 1
ax1 = axes[0, 0]
ax1.imshow(frame1, cmap='gray')
ax1.set_title('Frame 1', fontsize=11)
ax1.axis('off')

# 프레임 2
ax2 = axes[0, 1]
ax2.imshow(frame2, cmap='gray')
ax2.set_title('Frame 2', fontsize=11)
ax2.axis('off')

# 차이
ax3 = axes[0, 2]
ax3.imshow(np.abs(frame2.astype(float) - frame1), cmap='hot')
ax3.set_title('|Frame2 - Frame1|', fontsize=11)
ax3.axis('off')

# 그래디언트 Ix
ax4 = axes[1, 0]
ax4.imshow(Ix, cmap='RdBu')
ax4.set_title('Ix (x-gradient)', fontsize=11)
ax4.axis('off')

# 그래디언트 Iy
ax5 = axes[1, 1]
ax5.imshow(Iy, cmap='RdBu')
ax5.set_title('Iy (y-gradient)', fontsize=11)
ax5.axis('off')

# 광류 시각화
ax6 = axes[1, 2]
ax6.imshow(frame1, cmap='gray', alpha=0.5)
ax6.imshow(frame2, cmap='gray', alpha=0.5)

# 추적 점 표시
for orig, new, flow in results:
    ax6.arrow(orig[0], orig[1], flow[0], flow[1], 
              head_width=2, head_length=1, fc='red', ec='red')
    ax6.scatter([orig[0]], [orig[1]], c='blue', s=50)

ax6.set_title('Optical Flow', fontsize=11)
ax6.axis('off')

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week8/optical_flow_basics.png', dpi=150)
print("\nOptical flow basics saved: optical_flow_basics.png")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 70)
print("📚 Week 8 Basics 정리")
print("=" * 70)

print("""
✅ Part 1: 밝기 항상성
   - I(x,y,t) = I(x+Δx, y+Δy, t+Δt)
   - 광류 방정식: Iₓu + Iᵧv + Iₜ = 0

✅ Part 2: 그래디언트
   - Sobel로 Iₓ, Iᵧ
   - 프레임 차이로 Iₜ

✅ Part 3: Lucas-Kanade
   - 윈도우 내 광류 일정 가정
   - (AᵀA)⁻¹Aᵀb로 해 구함

✅ Part 4: 다중 점 추적
   - 각 점에 LK 적용
   - 성공/실패 판단

✅ Part 5: Structure Tensor
   - AᵀA = Harris M
   - 고유값으로 추적 품질 판단

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 핵심 메시지:
   Lucas-Kanade = 윈도우 내 일정 광류 가정
   코너에서 잘 작동, 에지/플랫에서 불안정!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 다음: optical_flow_quiz.py
""")

print("\n" + "=" * 70)
print("optical_flow_basics.py 실행 완료! 🎉")
print("=" * 70)
