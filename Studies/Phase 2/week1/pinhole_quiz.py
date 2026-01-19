"""
Phase 2 - Week 1: 핀홀 카메라 모델 실습 문제
==========================================
역투영, 카메라 이동, 다중 뷰 시뮬레이션

학습 목표:
1. 역투영 (2D → 3D ray) 이해
2. 카메라 포즈 변화에 따른 투영 변화
3. 이미지 경계 체크
4. 실전 시나리오 시뮬레이션

실행 시간: 약 2분
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=4, suppress=True)

# === 기본 함수 (pinhole_basics.py에서 가져옴) ===

def project_point(P_world, R, t, K):
    """3D 월드 점을 2D 픽셀로 투영"""
    P_world = np.array(P_world).flatten()
    P_camera = R @ P_world + t.flatten()
    
    Zc = P_camera[2]
    if Zc <= 0:
        return None, P_camera
    
    x_normalized = P_camera[0] / Zc
    y_normalized = P_camera[1] / Zc
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = fx * x_normalized + cx
    v = fy * y_normalized + cy
    
    return np.array([u, v]), P_camera

def project_points(points_world, R, t, K):
    """여러 3D 점을 투영"""
    pixels = []
    points_camera = []
    
    for P in points_world:
        pixel, P_cam = project_point(P, R, t, K)
        if pixel is not None:
            pixels.append(pixel)
            points_camera.append(P_cam)
    
    return np.array(pixels) if pixels else np.array([]), np.array(points_camera)

# 카메라 설정
image_width = 640
image_height = 480
K = np.array([
    [500,  0, 320],
    [0,  500, 240],
    [0,    0,   1]
])

print("=" * 70)
print("       Phase 2 - Week 1: 핀홀 카메라 실습 문제")
print("=" * 70)
print("\n이 실습에서는 핀홀 카메라 모델을 더 깊이 탐구합니다.\n")

# ============================================================
# 문제 1: 역투영 (Back-projection)
# ============================================================
print("\n" + "=" * 70)
print("문제 1: 역투영 (Back-projection)")
print("=" * 70)

print("""
🎯 목표: 2D 픽셀에서 3D 광선(ray) 방향 복원

역투영이란?
- 투영의 반대: 2D → 3D
- 하지만 완전한 3D 점은 복원 불가! (깊이 정보 없음)
- 복원 가능한 것: 광선의 "방향" (ray direction)

공식:
    x' = (u - cx) / fx
    y' = (v - cy) / fy
    ray_direction = [x', y', 1]  (정규화 필요)
""")

def back_project(pixel, K):
    """
    2D 픽셀에서 3D 광선 방향 복원
    
    Args:
        pixel: [u, v] 픽셀 좌표
        K: 내부 파라미터 행렬
    
    Returns:
        ray_direction: 정규화된 3D 방향 벡터
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u, v = pixel
    
    # 역투영
    x_normalized = (u - cx) / fx
    y_normalized = (v - cy) / fy
    
    # 광선 방향 (카메라 좌표계)
    ray = np.array([x_normalized, y_normalized, 1.0])
    
    # 정규화 (단위 벡터)
    ray_normalized = ray / np.linalg.norm(ray)
    
    return ray_normalized

# 테스트: 이미지 중심
pixel_center = np.array([320, 240])  # 이미지 중심 = 주점
ray_center = back_project(pixel_center, K)

print(f"\n[테스트 1] 이미지 중심")
print(f"  픽셀: {pixel_center}")
print(f"  광선 방향: {ray_center}")
print(f"  → Z축 방향 (0, 0, 1)과 같음!  ✅")

# 테스트: 이미지 모서리
pixel_corner = np.array([0, 0])  # 좌상단
ray_corner = back_project(pixel_corner, K)

print(f"\n[테스트 2] 이미지 좌상단")
print(f"  픽셀: {pixel_corner}")
print(f"  광선 방향: {ray_corner}")
print(f"  → 왼쪽 위를 향함 (음의 X, 음의 Y)")

# 검증: 투영 → 역투영 → 투영
P_test = np.array([2, 1, 5])
R_identity = np.eye(3)
t_zero = np.zeros((3, 1))

pixel_proj, _ = project_point(P_test, R_identity, t_zero, K)
ray_back = back_project(pixel_proj, K)

# 원본 점의 광선 방향
ray_original = P_test / np.linalg.norm(P_test)

print(f"\n[검증] 투영 → 역투영")
print(f"  원본 3D 점: {P_test}")
print(f"  투영된 픽셀: {pixel_proj}")
print(f"  역투영 광선: {ray_back}")
print(f"  원본 점의 방향: {ray_original}")
print(f"  방향 일치? {np.allclose(ray_back, ray_original, atol=1e-6)}  ✅")

# ============================================================
# 문제 2: 카메라 포즈 변화
# ============================================================
print("\n" + "=" * 70)
print("문제 2: 카메라 포즈 변화에 따른 투영")
print("=" * 70)

print("""
🎯 목표: 카메라가 움직이면 투영이 어떻게 변하는지 관찰

시나리오:
- 고정된 3D 점
- 카메라가 왼쪽으로 이동
- 점이 이미지에서 어디로 움직일까?
""")

def rotation_y(angle_deg):
    """Y축 회전 행렬"""
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

# 고정된 3D 점 (정면 5m)
P_fixed = np.array([0, 0, 5])

# 카메라 이동 시뮬레이션
print(f"\n고정된 3D 점: {P_fixed}")
print("\n카메라 X축 이동에 따른 투영 변화:")
print(f"{'t_x':>8} | {'u (픽셀)':>12} | 설명")
print("-" * 45)

for tx in [-2, -1, 0, 1, 2]:
    R = np.eye(3)
    t = np.array([[tx], [0], [0]])
    
    pixel, _ = project_point(P_fixed, R, t, K)
    
    if tx < 0:
        desc = "카메라 왼쪽 → 점이 오른쪽으로"
    elif tx > 0:
        desc = "카메라 오른쪽 → 점이 왼쪽으로"
    else:
        desc = "카메라 중앙 → 점도 중앙"
    
    print(f"{tx:>8} | {pixel[0]:>12.1f} | {desc}")

# 카메라 회전 시뮬레이션
print("\n카메라 Y축 회전에 따른 투영 변화:")
print(f"{'각도':>8} | {'u (픽셀)':>12} | 설명")
print("-" * 45)

for angle in [-20, -10, 0, 10, 20]:
    R = rotation_y(angle)
    t = np.zeros((3, 1))
    
    pixel, _ = project_point(P_fixed, R, t, K)
    
    if angle < 0:
        desc = "카메라 왼쪽 회전 → 점이 오른쪽으로"
    elif angle > 0:
        desc = "카메라 오른쪽 회전 → 점이 왼쪽으로"
    else:
        desc = "회전 없음 → 중앙"
    
    if pixel is None:
        print(f"{angle:>8}° | {'카메라 뒤':>12} | 투영 불가")
    else:
        print(f"{angle:>8}° | {pixel[0]:>12.1f} | {desc}")

# ============================================================
# 문제 3: 이미지 경계 체크
# ============================================================
print("\n" + "=" * 70)
print("문제 3: 이미지 경계 체크 (Visibility)")
print("=" * 70)

print("""
🎯 목표: 3D 점이 이미지에 보이는지 판단

조건:
1. 점이 카메라 앞에 있어야 함 (Zc > 0)
2. 픽셀이 이미지 경계 안에 있어야 함 (0 ≤ u < width, 0 ≤ v < height)
""")

def is_visible(P_world, R, t, K, image_size):
    """
    3D 점이 이미지에 보이는지 판단
    
    Returns:
        visible: bool
        reason: 설명 문자열
    """
    pixel, P_camera = project_point(P_world, R, t, K)
    width, height = image_size
    
    # 조건 1: 카메라 앞에 있어야
    if P_camera[2] <= 0:
        return False, "카메라 뒤에 있음"
    
    # 조건 2: 이미지 경계 내
    if pixel is None:
        return False, "투영 실패"
    
    u, v = pixel
    if u < 0 or u >= width:
        return False, f"X축 범위 밖 (u={u:.1f})"
    if v < 0 or v >= height:
        return False, f"Y축 범위 밖 (v={v:.1f})"
    
    return True, f"보임 ({u:.1f}, {v:.1f})"

# 테스트
R_test = np.eye(3)
t_test = np.zeros((3, 1))
image_size = (image_width, image_height)

test_points = [
    np.array([0, 0, 5]),      # 중앙 - 보임
    np.array([3, 0, 5]),      # 오른쪽 - 보임
    np.array([10, 0, 5]),     # 멀리 오른쪽 - 범위 밖
    np.array([0, 0, -5]),     # 카메라 뒤 - 안 보임
    np.array([0, 5, 5]),      # 위 - 범위 밖
]

print("\n가시성 테스트:")
print("-" * 60)
for P in test_points:
    visible, reason = is_visible(P, R_test, t_test, K, image_size)
    status = "✅" if visible else "❌"
    print(f"{str(P):>20} | {status} {reason}")

# ============================================================
# 문제 4: 다중 카메라 시뮬레이션
# ============================================================
print("\n" + "=" * 70)
print("문제 4: 다중 카메라 시뮬레이션")
print("=" * 70)

print("""
🎯 목표: 여러 카메라에서 같은 3D 점을 투영

시나리오:
- 3개의 카메라가 다른 위치에서 같은 점을 바라봄
- 각 카메라에서의 픽셀 좌표 계산
- 이것이 Visual SLAM의 기초!
""")

# 3D 점들 (정육면체 중심 5m 전방)
points_3d = np.array([
    [0, 0, 5],
    [1, 0, 5],
    [0, 1, 5],
    [-1, 0, 5],
    [0, -1, 5],
])

# 3개 카메라 설정
cameras = [
    {"name": "Cam 0 (Center)", "R": np.eye(3), "t": np.array([[0], [0], [0]])},
    {"name": "Cam 1 (Left)", "R": np.eye(3), "t": np.array([[-1], [0], [0]])},
    {"name": "Cam 2 (Right)", "R": np.eye(3), "t": np.array([[1], [0], [0]])},
]

print("\n각 카메라에서 3D 점 투영:")
for cam in cameras:
    print(f"\n{cam['name']}:")
    print(f"  t = {cam['t'].flatten()}")
    
    for i, P in enumerate(points_3d):
        pixel, _ = project_point(P, cam['R'], cam['t'], K)
        print(f"  점 {i} {P} → ({pixel[0]:.1f}, {pixel[1]:.1f})")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, cam in enumerate(cameras):
    ax = axes[idx]
    ax.set_title(cam['name'])
    ax.set_xlim([0, image_width])
    ax.set_ylim([image_height, 0])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    pixels, _ = project_points(points_3d, cam['R'], cam['t'], K)
    
    if len(pixels) > 0:
        ax.scatter(pixels[:, 0], pixels[:, 1], c='blue', s=100)
        
        for i, (p, P) in enumerate(zip(pixels, points_3d)):
            ax.annotate(f'P{i}', (p[0]+10, p[1]), fontsize=10)
    
    # 주점 표시
    ax.scatter([K[0, 2]], [K[1, 2]], c='red', marker='+', s=200, linewidths=2)

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 2/week1/multi_camera_projection.png', dpi=150)
print("\n시각화 저장: multi_camera_projection.png")
print("→ 카메라 위치에 따라 같은 3D 점이 다르게 투영됨")

# ============================================================
# 문제 5: FOV와 가시 영역
# ============================================================
print("\n" + "=" * 70)
print("문제 5: 시야각(FOV)과 가시 영역")
print("=" * 70)

print("""
🎯 목표: 특정 거리에서 볼 수 있는 영역 계산

공식:
    visible_width = 2 × Z × tan(FOV_x / 2)
""")

def visible_area_at_distance(K, image_size, distance):
    """특정 거리에서 볼 수 있는 영역 계산"""
    fx, fy = K[0, 0], K[1, 1]
    width, height = image_size
    
    fov_x = 2 * np.arctan(width / (2 * fx))
    fov_y = 2 * np.arctan(height / (2 * fy))
    
    visible_width = 2 * distance * np.tan(fov_x / 2)
    visible_height = 2 * distance * np.tan(fov_y / 2)
    
    return visible_width, visible_height

print("\n거리별 가시 영역:")
print(f"{'거리 (m)':>12} | {'너비 (m)':>12} | {'높이 (m)':>12}")
print("-" * 45)

for distance in [1, 5, 10, 20, 50]:
    vw, vh = visible_area_at_distance(K, (image_width, image_height), distance)
    print(f"{distance:>12} | {vw:>12.2f} | {vh:>12.2f}")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 70)
print("📚 Week 1 Quiz 정리")
print("=" * 70)

print("""
✅ 문제 1: 역투영
   - 2D 픽셀 → 3D 광선 방향
   - 깊이 정보 없이는 정확한 3D 점 복원 불가
   
✅ 문제 2: 카메라 포즈 변화
   - 카메라 이동 → 점이 반대 방향으로 움직임
   - 카메라 회전 → 점이 반대 방향으로 이동
   
✅ 문제 3: 가시성 판단
   - Zc > 0 AND 픽셀이 이미지 경계 내
   
✅ 문제 4: 다중 카메라
   - 같은 3D 점 → 다른 카메라에서 다른 픽셀
   - Visual SLAM의 기초!
   
✅ 문제 5: 가시 영역
   - FOV와 거리로 볼 수 있는 범위 계산

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 SLAM에서의 활용:

1. 역투영 → 삼각측량의 기초 (Week 7)
2. 다중 카메라 투영 → Stereo / Multi-view (Week 5-7)
3. 가시성 체크 → 특징점 추적 시 사용 (Week 8)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 다음: Week 2 - 렌즈 왜곡과 캘리브레이션
""")

print("\n" + "=" * 70)
print("pinhole_quiz.py 실행 완료! 🎉")
print("=" * 70)
print("\n생성된 파일:")
print("  1. projection_visualization.png - 3D→2D 투영 시각화")
print("  2. multi_camera_projection.png - 다중 카메라 시뮬레이션")
