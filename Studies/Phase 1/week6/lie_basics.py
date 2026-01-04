"""
Phase 1 - Week 6: Lie 군/대수 기초
==================================
SO(3), SE(3)와 exp/log 매핑 실습

학습 목표:
1. Over-parameterized 문제 이해
2. exp/log 매핑 구현
3. 최적화에서의 활용 이해
"""

import numpy as np
np.set_printoptions(precision=4, suppress=True)

print("=" * 60)
print("Phase 1 - Week 6: Lie 군/대수 기초")
print("=" * 60)

# ============================================================
# Part 1: Over-parameterized 문제
# ============================================================
print("\n" + "=" * 60)
print("Part 1: Over-parameterized 문제")
print("=" * 60)

print("""
회전 표현의 파라미터 수 vs 자유도:

| 표현 | 파라미터 | 자유도 | 초과 |
|------|---------|--------|------|
| 회전행렬 | 9 | 3 | +6 |
| 쿼터니언 | 4 | 3 | +1 |
| Lie대수 | 3 | 3 | ±0 ✓ |
""")

# 최적화에서 문제 시연
R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
dR = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])

R_bad = R + dR  # 단순 덧셈
print("잘못된 업데이트 (R + dR):")
print(R_bad)
print(f"RᵀR = I? {np.allclose(R_bad.T @ R_bad, np.eye(3))}")  # False!

# ============================================================
# Part 2: Skew-symmetric 행렬
# ============================================================
print("\n" + "=" * 60)
print("Part 2: Skew-symmetric 행렬")
print("=" * 60)

def skew(w):
    """벡터 → 반대칭 행렬 (skew-symmetric)"""
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

def vee(W):
    """반대칭 행렬 → 벡터"""
    return np.array([W[2,1], W[0,2], W[1,0]])

w = np.array([0.1, 0.2, 0.3])
W = skew(w)
print(f"벡터 ω = {w}")
print(f"\nSkew(ω) = ω^:")
print(W)
print(f"\nVee(ω^) = {vee(W)}")
print(f"원래 벡터와 같음: {np.allclose(w, vee(W))}")

# 반대칭 성질
print(f"\nW + W.T = 0? {np.allclose(W + W.T, 0)}")

# ============================================================
# Part 3: Rodrigues 공식 (exp 매핑)
# ============================================================
print("\n" + "=" * 60)
print("Part 3: Rodrigues 공식 (so(3) → SO(3))")
print("=" * 60)

def exp_so3(omega):
    """so(3) → SO(3) via Rodrigues formula
    
    R = I + sin(θ)K + (1-cos(θ))K²
    
    Args:
        omega: 3D 벡터 (축 × 각도)
    Returns:
        3x3 회전 행렬
    """
    theta = np.linalg.norm(omega)
    
    if theta < 1e-10:
        return np.eye(3)
    
    axis = omega / theta
    K = skew(axis)
    
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K @ K)
    return R

# Z축 90도 회전
omega = np.array([0, 0, np.pi/2])  # Z축, 90도
R = exp_so3(omega)

print(f"ω = [0, 0, π/2] (Z축 90도)")
print(f"\nexp(ω) =")
print(R)

# 검증: 직교성
print(f"\nRᵀR = I? {np.allclose(R.T @ R, np.eye(3))}")
print(f"det(R) = {np.linalg.det(R):.4f}")

# ============================================================
# Part 4: Log 매핑 (SO(3) → so(3))
# ============================================================
print("\n" + "=" * 60)
print("Part 4: Log 매핑 (SO(3) → so(3))")
print("=" * 60)

def log_so3(R):
    """SO(3) → so(3)
    
    Returns:
        3D 벡터 (축 × 각도)
    """
    # trace(R) = 1 + 2cos(theta)
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)  # 수치 안정성
    theta = np.arccos(cos_theta)
    
    if theta < 1e-10:
        return np.zeros(3)
    
    # (R - R^T) / 2 = sin(theta) * K
    omega_hat = (R - R.T) / (2 * np.sin(theta)) * theta
    return vee(omega_hat)

# 왕복 테스트
omega_original = np.array([0.3, 0.2, 0.5])
R_temp = exp_so3(omega_original)
omega_recovered = log_so3(R_temp)

print(f"원본 ω: {omega_original}")
print(f"exp 후 log: {omega_recovered}")
print(f"일치: {np.allclose(omega_original, omega_recovered)}")

# ============================================================
# Part 5: 최적화에서의 활용
# ============================================================
print("\n" + "=" * 60)
print("Part 5: 최적화에서의 활용")
print("=" * 60)

def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0], [s,c,0], [0,0,1]])

# 현재 회전
R_current = rotation_z(np.radians(30))

# 업데이트 (Lie 대수에서 3차원 벡터로)
delta_xi = np.array([0.01, 0.02, 0.05])  # 작은 변화량

# 올바른 업데이트: exp(Δξ) @ R
R_updated = exp_so3(delta_xi) @ R_current

print("현재 회전 R:")
print(R_current)
print(f"\n업데이트 Δξ = {delta_xi}")
print("\n업데이트된 R (exp(Δξ) @ R):")
print(R_updated)
print(f"\n여전히 회전 행렬? {np.allclose(R_updated.T @ R_updated, np.eye(3))}")

print("\n💡 핵심:")
print("   - Lie 대수에서 3개 파라미터로 업데이트")
print("   - exp 매핑으로 항상 유효한 회전 행렬 보장")
print("   - 제약 조건 처리 불필요!")

# ============================================================
# Part 6: SE(3)의 Lie 대수
# ============================================================
print("\n" + "=" * 60)
print("Part 6: SE(3)의 Lie 대수 (개념)")
print("=" * 60)

print("""
se(3): SE(3)의 접선 공간

ξ = [ρ, φ]ᵀ  (6차원 벡터)
  - ρ: 평행이동 관련 (3차원)
  - φ: 회전 관련 (3차원)

4x4 행렬 표현:
ξ^ = [φ^  | ρ ]
     [----+---]
     [0   | 0 ]

exp(ξ^) → SE(3) 변환 행렬
""")

def exp_se3_simple(xi):
    """간단한 SE(3) exp 매핑 (작은 각도 근사)"""
    rho = xi[:3]  # 평행이동
    phi = xi[3:]  # 회전
    
    R = exp_so3(phi)
    t = rho  # 작은 각도에서 근사
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

xi = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.1])  # [ρx,ρy,ρz, φx,φy,φz]
T = exp_se3_simple(xi)
print(f"ξ = {xi}")
print(f"\nexp(ξ) =")
print(T)

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 60)
print("📝 Week 6 정리")
print("=" * 60)
print("""
✅ Over-parameterized 문제
   - 회전행렬 9개, 쿼터니언 4개 vs 자유도 3
   - 단순 덧셈 업데이트 불가

✅ Skew-symmetric 행렬
   - 3D 벡터 ↔ 3x3 반대칭 행렬
   - skew(), vee() 변환

✅ exp/log 매핑
   - exp: so(3) → SO(3) (Rodrigues)
   - log: SO(3) → so(3)
   - 왕복 변환 가능

✅ 최적화 활용
   - Lie 대수에서 업데이트 (3 파라미터)
   - R_new = exp(Δξ) @ R
   - 항상 유효한 회전 보장

💡 Phase 5 (VINS-Fusion)에서 더 자세히!

🎯 다음: lie_quiz.py → Week 7: 최소자승법
""")
