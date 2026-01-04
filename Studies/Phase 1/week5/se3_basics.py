"""
Phase 1 - Week 5: 강체 변환 (SE(3))
===================================
SE(3) 변환 행렬과 동차 좌표 실습

학습 목표:
1. SE(3) 변환 행렬 구성
2. 동차 좌표 이해
3. 변환 합성과 역변환
4. 좌표계 변환
"""

import numpy as np
np.set_printoptions(precision=4, suppress=True)

print("=" * 60)
print("Phase 1 - Week 5: 강체 변환 (SE(3))")
print("=" * 60)

# ============================================================
# Part 1: SE(3) 변환 행렬 생성
# ============================================================
print("\n" + "=" * 60)
print("Part 1: SE(3) 변환 행렬")
print("=" * 60)

def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0], [s,c,0], [0,0,1]])

def make_se3(R, t):
    """SE(3) 변환 행렬 생성"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

# 예제: Z축 45도 회전 + (1, 2, 3) 이동
R = rotation_z(np.radians(45))
t = np.array([1, 2, 3])
T = make_se3(R, t)

print("\n회전 행렬 R (Z축 45도):")
print(R)
print(f"\n평행이동 t: {t}")
print("\nSE(3) 변환 행렬 T:")
print(T)

print("\n💡 T의 구조:")
print("   T[:3,:3] = R (회전)")
print("   T[:3,3]  = t (평행이동)")
print("   T[3,:]   = [0, 0, 0, 1]")

# ============================================================
# Part 2: 동차 좌표
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 동차 좌표 (Homogeneous Coordinates)")
print("=" * 60)

def to_homogeneous(p):
    """3D 점 → 동차 좌표"""
    return np.append(p, 1)

def from_homogeneous(p_h):
    """동차 좌표 → 3D 점"""
    return p_h[:3] / p_h[3]

# 점 변환 비교
p = np.array([1, 0, 0])

# 방법 1: 일반 좌표 (R @ p + t)
p_transformed_normal = R @ p + t

# 방법 2: 동차 좌표 (T @ p_h)
p_h = to_homogeneous(p)
p_h_transformed = T @ p_h
p_transformed_homo = from_homogeneous(p_h_transformed)

print(f"\n원점 p = {p}")
print(f"\n방법 1 (R@p + t): {p_transformed_normal}")
print(f"방법 2 (T @ p_h): {p_transformed_homo}")
print(f"\n결과 동일: {np.allclose(p_transformed_normal, p_transformed_homo)}")

print("\n💡 동차 좌표의 장점:")
print("   1. 회전+평행이동을 행렬 곱 하나로!")
print("   2. 연속 변환이 행렬 곱 체인")
print("   3. 투영도 통일된 형태")

# ============================================================
# Part 3: 변환 합성
# ============================================================
print("\n" + "=" * 60)
print("Part 3: 변환 합성 (Composition)")
print("=" * 60)

# 변환 1: Z축 90도 회전
T1 = make_se3(rotation_z(np.radians(90)), np.array([0, 0, 0]))

# 변환 2: X방향 2만큼 이동
T2 = make_se3(np.eye(3), np.array([2, 0, 0]))

# 합성: 먼저 T1, 다음 T2
T_combined = T2 @ T1

print("T1: Z축 90도 회전")
print(T1)
print("\nT2: X방향 2 이동")  
print(T2)
print("\nT2 @ T1 (먼저 T1, 다음 T2):")
print(T_combined)

# 점 변환으로 확인
p = np.array([1, 0, 0, 1])
print(f"\n점 (1,0,0)에 적용:")
print(f"  T1 후: {(T1 @ p)[:3]}")
print(f"  T2@T1 후: {(T_combined @ p)[:3]}")

# 순서 중요!
T_reverse = T1 @ T2
print(f"\n주의: T1 @ T2 ≠ T2 @ T1")
print(f"  T2@T1 결과: {(T_combined @ p)[:3]}")
print(f"  T1@T2 결과: {(T_reverse @ p)[:3]}")

# ============================================================
# Part 4: 역변환
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 역변환 (Inverse)")
print("=" * 60)

def inverse_se3(T):
    """SE(3) 역변환
    
    T^(-1) = [R^T | -R^T @ t]
             [0   |    1    ]
    """
    R = T[:3, :3]
    t = T[:3, 3]
    
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

# 역변환 계산
T_inv = inverse_se3(T)

print("원본 T:")
print(T)
print("\nT의 역변환 T^(-1):")
print(T_inv)

# 검증: T @ T^(-1) = I
identity = T @ T_inv
print("\nT @ T^(-1) (단위행렬이어야 함):")
print(identity)
print(f"단위행렬 맞음: {np.allclose(identity, np.eye(4))}")

# NumPy 역행렬과 비교
T_inv_numpy = np.linalg.inv(T)
print(f"\nNumPy inv와 결과 동일: {np.allclose(T_inv, T_inv_numpy)}")

print("\n💡 SE(3) 역변환 공식:")
print("   R^(-1) = R^T (회전 역변환)")
print("   t^(-1) = -R^T @ t")

# ============================================================
# Part 5: 좌표계 변환 예제
# ============================================================
print("\n" + "=" * 60)
print("Part 5: 좌표계 변환 (SLAM 활용)")
print("=" * 60)

# 월드 좌표계에서 카메라 포즈
T_wc = make_se3(
    rotation_z(np.radians(30)),  # 30도 회전
    np.array([5, 3, 1])          # 위치
)

print("카메라 포즈 T_wc (월드 → 카메라 좌표계):")
print("  회전: Z축 30도")
print("  위치: (5, 3, 1)")

# 카메라→월드 변환 (역변환)
T_cw = inverse_se3(T_wc)

# 월드 좌표의 3D 점
P_world = np.array([6, 4, 1, 1])
print(f"\n월드 좌표 점: {P_world[:3]}")

# 카메라 좌표로 변환
P_camera = T_cw @ P_world
print(f"카메라 좌표 점: {P_camera[:3]}")

# ============================================================
# Part 6: 상대 포즈
# ============================================================
print("\n" + "=" * 60)
print("Part 6: 상대 포즈 (Relative Pose)")
print("=" * 60)

# 키프레임 1, 2의 포즈 (월드 기준)
T_w1 = make_se3(rotation_z(np.radians(0)), np.array([0, 0, 0]))
T_w2 = make_se3(rotation_z(np.radians(45)), np.array([2, 1, 0]))

print("키프레임 1 포즈 (월드): 원점, 회전 없음")
print("키프레임 2 포즈 (월드): (2,1,0), Z축 45도")

# 상대 포즈: 1 기준으로 2가 어디?
T_12 = inverse_se3(T_w1) @ T_w2
print("\n상대 포즈 T_12 (1→2):")
print(T_12)
print(f"  상대 위치: {T_12[:3, 3]}")

# 반대 방향
T_21 = inverse_se3(T_w2) @ T_w1
print("\n상대 포즈 T_21 (2→1):")
print(f"  상대 위치: {T_21[:3, 3]}")

print("\n💡 상대 포즈 공식:")
print("   T_ij = T_wi^(-1) @ T_wj")
print("   = i 기준으로 j가 어디 있는가")

# ============================================================
# Part 7: ROS TF2 연결
# ============================================================
print("\n" + "=" * 60)
print("Part 7: ROS TF2 연결")
print("=" * 60)

def rotation_to_quaternion(R):
    """회전 행렬 → 쿼터니언 [w,x,y,z]"""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5/np.sqrt(tr+1)
        w = 0.25/s
        x = (R[2,1]-R[1,2])*s
        y = (R[0,2]-R[2,0])*s
        z = (R[1,0]-R[0,1])*s
    else:
        # 간단화된 버전
        w, x, y, z = 1, 0, 0, 0
    return np.array([w,x,y,z]) / np.linalg.norm([w,x,y,z])

def se3_to_ros_transform(T):
    """SE(3) → ROS geometry_msgs/Transform 형식"""
    t = T[:3, 3]
    q = rotation_to_quaternion(T[:3, :3])
    
    return {
        'translation': {'x': t[0], 'y': t[1], 'z': t[2]},
        'rotation': {'x': q[1], 'y': q[2], 'z': q[3], 'w': q[0]}
    }

ros_tf = se3_to_ros_transform(T_w2)
print("\nT_w2를 ROS Transform 형식으로:")
print(f"  translation: x={ros_tf['translation']['x']:.2f}, "
      f"y={ros_tf['translation']['y']:.2f}, z={ros_tf['translation']['z']:.2f}")
print(f"  rotation: x={ros_tf['rotation']['x']:.4f}, "
      f"y={ros_tf['rotation']['y']:.4f}, z={ros_tf['rotation']['z']:.4f}, "
      f"w={ros_tf['rotation']['w']:.4f}")

print("\n💡 ROS TF2 구조:")
print("   - translation: Vector3 (x, y, z)")
print("   - rotation: Quaternion (x, y, z, w)")
print("   - 주의: ROS는 [x,y,z,w] 순서!")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 60)
print("📝 Week 5 정리")
print("=" * 60)
print("""
✅ SE(3) 변환 행렬
   - 4x4 행렬: [R|t; 0|1]
   - 6 자유도 (회전 3 + 이동 3)

✅ 동차 좌표
   - 회전+이동을 행렬 곱 하나로
   - 3D 점 (x,y,z) → (x,y,z,1)

✅ 변환 연산
   - 합성: T2 @ T1 (순서 중요!)
   - 역변환: [R^T | -R^T@t]

✅ 좌표계 변환
   - T_wc: 카메라 포즈 (월드 기준)
   - T_cw = T_wc^(-1): 월드 → 카메라

✅ 상대 포즈
   - T_ij = T_i^(-1) @ T_j

✅ ROS TF2
   - translation(Vector3) + rotation(Quaternion)

🎯 다음: se3_quiz.py → Week 6: Lie 군/대수
""")
