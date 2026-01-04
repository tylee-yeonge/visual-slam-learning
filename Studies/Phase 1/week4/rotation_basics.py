"""
Phase 1 - Week 4: 회전 표현 기초
================================
회전 행렬, 오일러 각, 쿼터니언 실습

학습 목표:
1. 2D/3D 회전 행렬 구성
2. 오일러 각과 짐벌락 이해
3. 쿼터니언 기본 연산
4. 회전 표현 간 변환
"""

import numpy as np
np.set_printoptions(precision=4, suppress=True)

print("=" * 60)
print("Phase 1 - Week 4: 회전 표현 기초")
print("=" * 60)

# ============================================================
# Part 1: 2D/3D 회전 행렬
# ============================================================
print("\n" + "=" * 60)
print("Part 1: 회전 행렬")
print("=" * 60)

def rotation_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1,0,0], [0,c,-s], [0,s,c]])

def rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,0,s], [0,1,0], [-s,0,c]])

def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0], [s,c,0], [0,0,1]])

# 45도 회전
angle = np.radians(45)
Rx, Ry, Rz = rotation_x(angle), rotation_y(angle), rotation_z(angle)

print("\nZ축 45도 회전 행렬:")
print(Rz)
print(f"\ndet(Rz) = {np.linalg.det(Rz):.4f}")
print(f"Rz^T @ Rz = I? {np.allclose(Rz.T @ Rz, np.eye(3))}")

# 회전 순서 비가환성
print("\n--- 회전 순서의 중요성 ---")
R_xyz = Rz @ Ry @ Rx
R_zyx = Rx @ Ry @ Rz
print(f"Rz@Ry@Rx == Rx@Ry@Rz? {np.allclose(R_xyz, R_zyx)}")

# ============================================================
# Part 2: 오일러 각과 짐벌락
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 오일러 각과 짐벌락")
print("=" * 60)

def euler_to_rotation(roll, pitch, yaw):
    """ZYX 순서"""
    return rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)

# 정상적인 경우
R_normal = euler_to_rotation(np.radians(10), np.radians(20), np.radians(30))
print("\nRoll=10°, Pitch=20°, Yaw=30° 회전 행렬:")
print(R_normal)

# 짐벌락: pitch = 90도
print("\n--- 짐벌락 (Pitch=90°) ---")
for r, y in [(0, 30), (30, 0), (15, 15)]:
    R = euler_to_rotation(np.radians(r), np.radians(90), np.radians(y))
    print(f"Roll={r:2d}°, Yaw={y:2d}° → R[0,1:3]={R[0,1:3]}")

print("\n💡 Pitch=90°에서 Roll과 Yaw가 같은 효과!")

# ============================================================
# Part 3: 쿼터니언
# ============================================================
print("\n" + "=" * 60)
print("Part 3: 쿼터니언")
print("=" * 60)

def axis_angle_to_quat(axis, angle):
    axis = axis / np.linalg.norm(axis)
    return np.array([np.cos(angle/2), *(np.sin(angle/2) * axis)])

def quat_multiply(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2-x1*x2-y1*y2-z1*z2,
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2])

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def rotate_by_quat(v, q):
    v_q = np.array([0, *v])
    return quat_multiply(quat_multiply(q, v_q), quat_conjugate(q))[1:]

# 쿼터니언 생성
q = axis_angle_to_quat([0,0,1], np.radians(45))
print(f"\nZ축 45도 회전 쿼터니언: {q}")
print(f"노름: {np.linalg.norm(q):.6f}")

# 벡터 회전
v = np.array([1, 0, 0])
v_rot = rotate_by_quat(v, q)
print(f"\n[1,0,0] 회전 결과: {v_rot}")

# ============================================================
# Part 4: 쿼터니언 ↔ 회전 행렬
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 상호 변환")
print("=" * 60)

def quat_to_rotmat(q):
    q = q / np.linalg.norm(q)
    w,x,y,z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])

def rotmat_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = 0.5/np.sqrt(tr+1)
        w = 0.25/s
        x,y,z = (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2])
            w,x = (R[2,1]-R[1,2])/s, 0.25*s
            y,z = (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
        elif R[1,1] > R[2,2]:
            s = 2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2])
            w,y = (R[0,2]-R[2,0])/s, 0.25*s
            x,z = (R[0,1]+R[1,0])/s, (R[1,2]+R[2,1])/s
        else:
            s = 2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1])
            w,z = (R[1,0]-R[0,1])/s, 0.25*s
            x,y = (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s
    q = np.array([w,x,y,z])
    return q / np.linalg.norm(q)

R_orig = rotation_z(np.radians(60))
q_conv = rotmat_to_quat(R_orig)
R_back = quat_to_rotmat(q_conv)
print(f"변환 정확도: {np.allclose(R_orig, R_back)}")

# ============================================================
# Part 5: SLERP
# ============================================================
print("\n" + "=" * 60)
print("Part 5: SLERP (구면 선형 보간)")
print("=" * 60)

def slerp(q1, q2, t):
    dot = np.dot(q1, q2)
    if dot < 0:
        q2, dot = -q2, -dot
    if dot > 0.9995:
        return (q1 + t*(q2-q1)) / np.linalg.norm(q1 + t*(q2-q1))
    theta = np.arccos(dot)
    return (np.sin((1-t)*theta)*q1 + np.sin(t*theta)*q2) / np.sin(theta)

q0 = axis_angle_to_quat([0,0,1], 0)
q1 = axis_angle_to_quat([0,0,1], np.radians(90))

print("\n0° → 90° 보간:")
for t in [0, 0.25, 0.5, 0.75, 1.0]:
    q_t = slerp(q0, q1, t)
    ang = np.degrees(2*np.arccos(np.clip(q_t[0], -1, 1)))
    print(f"  t={t:.2f}: {ang:.1f}°")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 60)
print("📝 Week 4 정리")
print("=" * 60)
print("""
✅ 회전 행렬: 직교, det=1, R⁻¹=Rᵀ
✅ 오일러 각: 직관적이나 짐벌락 문제
✅ 쿼터니언: 짐벌락 없음, SLERP 가능
✅ SLAM 활용: IMU 적분, 최적화에 쿼터니언 사용

🎯 다음: rotation_quiz.py → Week 5: SE(3)
""")
