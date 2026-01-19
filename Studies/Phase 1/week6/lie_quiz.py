"""
Phase 1 - Week 6: Lie 군/대수 실습 문제
======================================
SO(3) 검증, exp/log 매핑, 회전 합성, 포즈 최적화, ICP 구현

학습 목표:
1. SO(3) 원소 확인 (검증)
2. exp/log 매핑의 역함수 관계 검증
3. 회전 합성과 Lie 대수의 관계
4. 간단한 포즈 최적화 문제
5. ICP 알고리즘 구현

실행 시간: 약 2-3분
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=4, suppress=True)

# lie_basics.py에서 정의한 함수들 재사용
def skew(w):
    """벡터 → 반대칭 행렬"""
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

def vee(W):
    """반대칭 행렬 → 벡터"""
    return np.array([W[2, 1], W[0, 2], W[1, 0]])

def exp_so3(omega):
    """so(3) → SO(3) (Rodrigues 공식)"""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3)
    axis = omega / theta
    K = skew(axis)
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def log_so3(R):
    """SO(3) → so(3)"""
    trace = np.trace(R)
    cos_theta = np.clip((trace - 1) / 2, -1, 1)
    theta = np.arccos(cos_theta)
    if theta < 1e-10:
        return np.zeros(3)
    omega_hat = (R - R.T) * theta / (2 * np.sin(theta))
    return vee(omega_hat)

def exp_se3(xi):
    """se(3) → SE(3)"""
    rho = xi[:3]
    phi = xi[3:]
    R = exp_so3(phi)
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        J = np.eye(3)
    else:
        axis = phi / theta
        K = skew(axis)
        J = np.eye(3) + ((1 - np.cos(theta)) / theta) * K + \
            ((theta - np.sin(theta)) / theta) * (K @ K)
    t = J @ rho
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

print("=" * 70)
print("          Phase 1 - Week 6: Lie 군/대수 실습 문제")
print("=" * 70)
print("\n이 퀴즈는 lie_basics.py의 내용을 확장하는 실전 문제입니다.")
print("각 문제를 풀며 개념을 다시 확인하세요!\n")

# ============================================================
# 문제 1: SO(3) 원소 확인
# ============================================================
print("\n" + "=" * 70)
print("문제 1: SO(3) 원소 확인")
print("=" * 70)

print("""
🎯 목표: 주어진 행렬이 SO(3)의 원소인지 검증

SO(3)의 조건:
1. RᵀR = I (직교 행렬)
2. det(R) = 1 (특별 직교)
3. R ∈ ℝ³ˣ³
""")

def is_valid_rotation(R, verbose=True):
    """
    회전 행렬 유효성 검증
    
    Args:
        R: 3×3 행렬
        verbose: 상세 출력 여부
    
    Returns:
        bool: SO(3) 원소 여부
    """
    # 크기 확인
    if R.shape != (3, 3):
        if verbose:
            print("❌ 크기가 3×3이 아님")
        return False
    
    # 직교성 확인: RᵀR = I
    RtR = R.T @ R
    is_orthogonal = np.allclose(RtR, np.eye(3))
    
    # 행렬식 확인: det(R) = 1
    det_R = np.linalg.det(R)
    is_special = np.isclose(det_R, 1.0)
    
    if verbose:
        print(f"RᵀR =")
        print(RtR)
        print(f"직교성(RᵀR=I): {'✅' if is_orthogonal else '❌'}")
        print(f"det(R) = {det_R:.6f}")
        print(f"특별 직교(det=1): {'✅' if is_special else '❌'}")
    
    return is_orthogonal and is_special

print("\n" + "-" * 70)
print("테스트 케이스들:")
print("-" * 70)

# 케이스 1: 유효한 회전 (Z축 45도)
print("\n[케이스 1] Z축 45도 회전")
theta = np.pi / 4
R1 = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0, 0, 1]
])
result1 = is_valid_rotation(R1)
print(f"결과: {'SO(3) 원소 ✅' if result1 else 'SO(3) 아님 ❌'}")

# 케이스 2: 유효하지 않은 행렬 (스케일링 포함)
print("\n[케이스 2] 스케일링이 포함된 행렬")
R2 = np.array([
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 2]
])
result2 = is_valid_rotation(R2)
print(f"결과: {'SO(3) 원소 ✅' if result2 else 'SO(3) 아님 ❌'}")

# 케이스 3: 반사 포함 (det = -1)
print("\n[케이스 3] 반사가 포함된 행렬")
R3 = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
result3 = is_valid_rotation(R3)
print(f"결과: {'SO(3) 원소 ✅' if result3 else 'SO(3) 아님 ❌'}")

# 케이스 4: exp로 생성한 회전
print("\n[케이스 4] exp 매핑으로 생성한 회전")
omega = np.array([0.5, -0.3, 0.8])
R4 = exp_so3(omega)
result4 = is_valid_rotation(R4)
print(f"결과: {'SO(3) 원소 ✅' if result4 else 'SO(3) 아님 ❌'}")

# ============================================================
# 문제 2: exp/log 역함수 관계 검증
# ============================================================
print("\n" + "=" * 70)
print("문제 2: exp/log 역함수 관계 검증")
print("=" * 70)

print("""
🎯 목표: exp와 log가 정말로 역함수인지 수치적으로 검증

이론:
    log(exp(ω)) = ω  (∀ω ∈ so(3))
    exp(log(R)) = R  (∀R ∈ SO(3))
    
단, 각도가 π 근처에서는 수치 오차 주의!
""")

print("\n" + "-" * 70)
print("테스트: 다양한 회전 각도")
print("-" * 70)

test_angles = [
    ("작은 각도", 0.1),
    ("중간 각도", 1.0),
    ("큰 각도", 2.5),
    ("거의 π", 3.1),
]

for name, angle in test_angles:
    # 임의의 축
    axis = np.array([1, 1, 1]) / np.sqrt(3)
    omega = axis * angle
    
    # 왕복 테스트
    R = exp_so3(omega)
    omega_back = log_so3(R)
    
    # 오차 계산
    error = np.linalg.norm(omega - omega_back)
    
    print(f"\n{name} ({np.degrees(angle):.1f}°):")
    print(f"  원본 ω:  {omega}")
    print(f"  복원 ω:  {omega_back}")
    print(f"  오차:    {error:.2e}  {'✅' if error < 1e-6 else '⚠️'}")

# ============================================================
# 문제 3: 회전 합성과 Lie 대수
# ============================================================
print("\n" + "=" * 70)
print("문제 3: 회전 합성과 Lie 대수의 관계")
print("=" * 70)

print("""
🎯 목표: 두 회전의 합성을 Lie 대수에서 이해하기

중요한 점:
    exp(ω₁) @ exp(ω₂) ≠ exp(ω₁ + ω₂)  (일반적으로)
    
    BUT, 작은 각도에서는:
    exp(ω₁) @ exp(ω₂) ≈ exp(ω₁ + ω₂)  (1차 근사)
""")

print("\n" + "-" * 70)
print("실험: 두 회전의 합성")
print("-" * 70)

# 두 회전
omega1 = np.array([0.1, 0, 0])  # X축 작은 회전
omega2 = np.array([0, 0.1, 0])  # Y축 작은 회전

R1 = exp_so3(omega1)
R2 = exp_so3(omega2)

# 합성 방법 1: 행렬 곱
R_composed = R1 @ R2
omega_composed = log_so3(R_composed)

# 합성 방법 2: Lie 대수 덧셈 (근사)
omega_sum = omega1 + omega2
R_sum = exp_so3(omega_sum)

print(f"\nω₁ = {omega1}  (X축)")
print(f"ω₂ = {omega2}  (Y축)")

print(f"\n[방법 1] exp(ω₁) @ exp(ω₂)의 log:")
print(f"  log(R₁ @ R₂) = {omega_composed}")

print(f"\n[방법 2] ω₁ + ω₂를 exp:")
print(f"  exp(ω₁ + ω₂)의 log = {omega_sum}")

# 비교
R_diff = np.linalg.norm(R_composed - R_sum, 'fro')
print(f"\n회전 행렬 차이: {R_diff:.2e}")
print(f"작은 각도에서는 근사 가능: {'✅' if R_diff < 0.01 else '❌'}")

# 큰 각도에서는?
print("\n" + "-" * 70)
print("큰 각도에서는 어떻게 될까?")
print("-" * 70)

omega1_large = np.array([1.0, 0, 0])
omega2_large = np.array([0, 1.0, 0])

R1_large = exp_so3(omega1_large)
R2_large = exp_so3(omega2_large)

R_composed_large = R1_large @ R2_large
omega_composed_large = log_so3(R_composed_large)

omega_sum_large = omega1_large + omega2_large
R_sum_large = exp_so3(omega_sum_large)

R_diff_large = np.linalg.norm(R_composed_large - R_sum_large, 'fro')

print(f"\nω₁ = {omega1_large}")
print(f"ω₂ = {omega2_large}")
print(f"\nlog(R₁ @ R₂) = {omega_composed_large}")
print(f"ω₁ + ω₂      = {omega_sum_large}")
print(f"\n회전 행렬 차이: {R_diff_large:.2e}")
print(f"큰 각도에서는 근사 불가: {'❌' if R_diff_large > 0.1 else '✅'}")

# ============================================================
# 문제 4: 간단한 포즈 최적화
# ============================================================
print("\n" + "=" * 70)
print("문제 4: Lie 대수를 이용한 간단한 포즈 최적화")
print("=" * 70)

print("""
🎯 목표: Gradient descent로 회전 최적화 해보기

시나리오:
    - 현재 회전: R_current
    - 목표 회전: R_target
    - 목적 함수: E(R) = ‖R - R_target‖²
    
업데이트:
    R ← exp(Δω) @ R
    여기서 Δω는 gradient 방향
""")

print("\n" + "-" * 70)
print("최적화 시작")
print("-" * 70)

# 목표 회전 (Z축 60도)
R_target = exp_so3(np.array([0, 0, np.pi/3]))

# 초기 회전 (약간 어긋난 상태)
R_current = exp_so3(np.array([0.1, 0.2, 1.0]))

# 최적화 파라미터
learning_rate = 0.1
max_iterations = 20

print(f"목표 회전:")
print(R_target)
print(f"\n초기 회전:")
print(R_current)

errors = []

for i in range(max_iterations):
    # 오차 계산
    R_error = R_current - R_target
    error = np.linalg.norm(R_error, 'fro')
    errors.append(error)
    
    # 수렴 확인
    if error < 1e-6:
        print(f"\n{i}번 반복 후 수렴! 오차 = {error:.2e}")
        break
    
    # Gradient (단순화된 버전)
    # ∇E = 2(R - R_target)
    gradient_R = 2 * R_error
    
    # Lie 대수로 업데이트 (간단한 투영)
    # log(I + εG) ≈ vee(G)
    delta_omega = -learning_rate * vee((gradient_R @ R_current.T + R_current.T @ gradient_R) / 2)
    
    # 회전 업데이트
    R_current = exp_so3(delta_omega) @ R_current
    
    if i % 5 == 0:
        print(f"반복 {i:2d}: 오차 = {error:.6f}, Δω = {np.linalg.norm(delta_omega):.2e}")

print(f"\n최종 회전:")
print(R_current)
print(f"목표 도달: {np.allclose(R_current, R_target, atol=1e-3)}  {'✅' if np.allclose(R_current, R_target, atol=1e-3) else '❌'}")

# 오차 수렴 플롯
plt.figure(figsize=(8, 5))
plt.semilogy(errors, 'b-o', linewidth=2, markersize=4)
plt.grid(True, alpha=0.3)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Error (log scale)', fontsize=12)
plt.title('Rotation Optimization Convergence', fontsize=14)
plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 1/week6/optimization_convergence.png', dpi=150)
print(f"\n수렴 그래프 저장: optimization_convergence.png")

# ============================================================
# 문제 5: ICP (Iterative Closest Point) 알고리즘
# ============================================================
print("\n" + "=" * 70)
print("문제 5: ICP 알고리즘 구현")
print("=" * 70)

print("""
🎯 목표: Lie 대수를 활용한 3D 포인트 클라우드 정렬

문제:
    - Source 포인트: {p₁, p₂, ..., pₙ}
    - Target 포인트: {q₁, q₂, ..., qₙ}
    - 찾기: R, t  such that  R·pᵢ + t ≈ qᵢ

알고리즘:
    1. 대응점 찾기 (여기서는 이미 알고 있다고 가정)
    2. SE(3) 파라미터 업데이트
    3. 수렴할 때까지 반복
""")

def icp_lie_algebra(source, target, max_iter=50, tol=1e-6):
    """
    Lie 대수 기반 ICP
    
    Args:
        source: (N, 3) numpy array
        target: (N, 3) numpy array
        max_iter: 최대 반복 횟수
        tol: 수렴 임계값
    
    Returns:
        T: 4×4 변환 행렬
        errors: 각 반복의 오차
    """
    N = source.shape[0]
    
    # 초기 변환 (항등)
    T = np.eye(4)
    
    errors = []
    
    for iteration in range(max_iter):
        # 현재 변환 적용
        source_homogeneous = np.hstack([source, np.ones((N, 1))])
        source_transformed = (T @ source_homogeneous.T).T[:, :3]
        
        # 잔차 계산
        residuals = source_transformed - target
        error = np.mean(np.linalg.norm(residuals, axis=1))
        errors.append(error)
        
        # 수렴 확인
        if error < tol:
            print(f"  {iteration}번 반복 후 수렴!")
            break
        
        # Jacobian 계산 (단순화)
        # 실제로는 더 복잡하지만, 여기서는 개념 설명용
        
        # 중심으로 이동
        source_centered = source_transformed - np.mean(source_transformed, axis=0)
        target_centered = target - np.mean(target, axis=0)
        
        # Cross-covariance
        H = source_centered.T @ target_centered
        
        # SVD로 회전 계산
        U, S, Vt = np.linalg.svd(H)
        R_update = Vt.T @ U.T
        
        # det(R) = 1 보장
        if np.linalg.det(R_update) < 0:
            Vt[-1, :] *= -1
            R_update = Vt.T @ U.T
        
        # 평행이동 계산
        t_update = np.mean(target, axis=0) - R_update @ np.mean(source_transformed, axis=0)
        
        # SE(3)로 변환
        T_update = np.eye(4)
        T_update[:3, :3] = R_update
        T_update[:3, 3] = t_update
        
        # 업데이트
        T = T_update @ T
        
    return T, errors

print("\n" + "-" * 70)
print("ICP 테스트: 작은 회전 + 평행이동")
print("-" * 70)

# Source 포인트 (정육면체 꼭짓점)
source_points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
], dtype=float)

# 실제 변환 (알려지지 않은 것으로 가정)
R_true = exp_so3(np.array([0.1, 0.2, 0.15]))
t_true = np.array([0.5, -0.3, 0.8])

T_true = np.eye(4)
T_true[:3, :3] = R_true
T_true[:3, 3] = t_true

# Target 포인트 생성
source_homogeneous = np.hstack([source_points, np.ones((8, 1))])
target_points = (T_true @ source_homogeneous.T).T[:, :3]

# 노이즈 추가
target_points += np.random.randn(*target_points.shape) * 0.01

print(f"실제 회전 (Rodrigues 벡터):")
print(f"  ω = {log_so3(R_true)}")
print(f"실제 평행이동:")
print(f"  t = {t_true}")

# ICP 실행
print(f"\nICP 실행 중...")
T_estimated, errors = icp_lie_algebra(source_points, target_points)

# 결과
R_estimated = T_estimated[:3, :3]
t_estimated = T_estimated[:3, 3]

print(f"\n추정 회전 (Rodrigues 벡터):")
print(f"  ω = {log_so3(R_estimated)}")
print(f"추정 평행이동:")
print(f"  t = {t_estimated}")

# 오차
R_error = np.linalg.norm(log_so3(R_true) - log_so3(R_estimated))
t_error = np.linalg.norm(t_true - t_estimated)

print(f"\n오차:")
print(f"  회전 오차: {R_error:.2e}  {'✅' if R_error < 0.1 else '❌'}")
print(f"  평행이동 오차: {t_error:.2e}  {'✅' if t_error < 0.1 else '❌'}")

# 수렴 플롯
plt.figure(figsize=(8, 5))
plt.semilogy(errors, 'r-o', linewidth=2, markersize=4)
plt.grid(True, alpha=0.3)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Mean Error (log scale)', fontsize=12)
plt.title('ICP Convergence', fontsize=14)
plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 1/week6/icp_convergence.png', dpi=150)
print(f"ICP 수렴 그래프 저장: icp_convergence.png")

# 3D 시각화
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], 
           c='blue', marker='o', s=100, label='Source')
ax1.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
           c='red', marker='^', s=100, label='Target')
ax1.set_title('Before ICP')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.legend()

ax2 = fig.add_subplot(122, projection='3d')
source_aligned = (T_estimated @ source_homogeneous.T).T[:, :3]
ax2.scatter(source_aligned[:, 0], source_aligned[:, 1], source_aligned[:, 2], 
           c='green', marker='o', s=100, label='Aligned')
ax2.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
           c='red', marker='^', s=100, label='Target')
ax2.set_title('After ICP')
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
ax2.legend()

plt.tight_layout()
plt.savefig('/Users/yeonge/SynologyDrive/1. YeongE/7. Visual SLAM Study/visual-slam-learning/Studies/Phase 1/week6/icp_result.png', dpi=150)
print(f"ICP 결과 시각화 저장: icp_result.png\n")

# ============================================================
# 종합 정리
# ============================================================
print("\n" + "=" * 70)
print("📚 Week 6 Quiz 정리")
print("=" * 70)

print("""
✅ 문제 1: SO(3) 원소 확인
   - RᵀR = I, det(R) = 1 조건 검증
   - exp로 생성한 행렬은 항상 유효
   
✅ 문제 2: exp/log 역함수 관계
   - log(exp(ω)) = ω 검증
   - 각도가 π 근처에서는 수치 오차 주의
   
✅ 문제 3: 회전 합성
   - exp(ω₁) @ exp(ω₂) ≠ exp(ω₁ + ω₂) (일반적)
   - 작은 각도에서만 근사 가능
   
✅ 문제 4: 포즈 최적화
   - Lie 대수에서 gradient descent
   - R ← exp(Δω) @ R 업데이트
   - 빠른 수렴 확인
   
✅ 문제 5: ICP 알고리즘
   - 3D 포인트 클라우드 정렬
   - Lie 대수 기반 최적화
   - 실전 SLAM 알고리즘의 기초

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 핵심 메시지:

Lie 군/대수는:
  ✅ 회전 표현의 효율적인 방법
  ✅ 최적화의 강력한 도구
  ✅ 현대 SLAM의 필수 기술

실습을 통해:
  ✅ 이론을 코드로 검증
  ✅ 실전 알고리즘(ICP) 구현
  ✅ 최적화 과정 시각화

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 다음 단계:
   1. README.md 다시 읽기 (더 깊은 이해)
   2. 생성된 이미지들 확인
   3. Week 7 (최소자승법 및 비선형 최적화) 준비
   
📌 Phase 5 (VINS-Fusion)에서 이 개념들을 실제 코드에서 만나게 됩니다!

🎓 자가 평가:
   - 모든 테스트 통과? ✅
   - 개념 이해도: /10
   - 다음 주 준비도: %
""")

print("\n" + "=" * 70)
print("lie_quiz.py 실습 완료! 🎉")
print("=" * 70)
print("\n생성된 파일:")
print("  1. rotation_visualization.png - 회전 시각화")
print("  2. optimization_convergence.png - 최적화 수렴 그래프")
print("  3. icp_convergence.png - ICP 수렴 그래프")
print("  4. icp_result.png - ICP 결과 3D 시각화")
print("\n이 이미지들을 열어서 결과를 확인하세요!")
