"""
Phase 1 - Week 2: 선형대수 실습
=================================
NumPy를 활용한 선형대수 기본 연산 및 SLAM 적용 예제

학습 목표:
1. 행렬 곱셈 직접 계산 vs 라이브러리 비교
2. 역행렬 계산
3. 고유값 분해 실습
4. 행렬식 계산
5. 회전 행렬의 성질 이해
6. 공분산 행렬과 불확실성 표현
"""

import numpy as np
np.set_printoptions(precision=4, suppress=True)

print("=" * 60)
print("Phase 1 - Week 2: 선형대수 실습")
print("=" * 60)

# ============================================================
# Part 1: 행렬 곱셈 - 직접 계산 vs 라이브러리
# ============================================================
print("\n" + "=" * 60)
print("Part 1: 행렬 곱셈 - 직접 계산 vs 라이브러리")
print("=" * 60)

# 2x3 행렬 A와 3x2 행렬 B 정의
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

B = np.array([
    [7, 8],
    [9, 10],
    [11, 12]
])

print("\n행렬 A (2x3):")
print(A)
print("\n행렬 B (3x2):")
print(B)

# 방법 1: 직접 계산 (수동으로 이해하기)
def matrix_multiply_manual(A, B):
    """행렬 곱셈을 직접 구현하여 원리 이해"""
    # A: (m x n), B: (n x p) -> 결과: (m x p)
    m, n = A.shape
    n2, p = B.shape
    
    if n != n2:
        raise ValueError("행렬 곱셈 불가: A의 열 수와 B의 행 수가 다릅니다")
    
    result = np.zeros((m, p))
    
    for i in range(m):        # A의 각 행에 대해
        for j in range(p):    # B의 각 열에 대해
            for k in range(n): # 내적 계산
                result[i, j] += A[i, k] * B[k, j]
    
    return result

# 방법 2: NumPy 라이브러리 사용
result_manual = matrix_multiply_manual(A, B)
result_numpy = A @ B  # 또는 np.dot(A, B)

print("\n직접 계산 결과:")
print(result_manual)
print("\nNumPy 계산 결과:")
print(result_numpy)
print("\n두 결과가 같은가?:", np.allclose(result_manual, result_numpy))

# 행렬 곱셈의 기하학적 의미 설명
print("\n💡 행렬 곱셈의 의미:")
print("   AB에서 B의 각 열 벡터가 A에 의해 '변환'됩니다.")
print("   결과의 (i,j) 원소 = A의 i번째 행과 B의 j번째 열의 내적")

# ============================================================
# Part 2: 역행렬 계산
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 역행렬 계산")
print("=" * 60)

# 역행렬이 존재하는 2x2 정방행렬
M = np.array([
    [4, 7],
    [2, 6]
])

print("\n행렬 M:")
print(M)

# 역행렬 계산
M_inv = np.linalg.inv(M)
print("\nM의 역행렬 M^(-1):")
print(M_inv)

# 검증: M * M^(-1) = I
identity_check = M @ M_inv
print("\nM * M^(-1) (단위행렬이어야 함):")
print(identity_check)

# 2x2 역행렬 공식 직접 계산
def inverse_2x2_manual(M):
    """2x2 행렬의 역행렬 공식: 1/det(M) * [[d, -b], [-c, a]]"""
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    
    det = a * d - b * c
    
    if abs(det) < 1e-10:
        raise ValueError("행렬식이 0이므로 역행렬이 존재하지 않습니다")
    
    return np.array([
        [d, -b],
        [-c, a]
    ]) / det

M_inv_manual = inverse_2x2_manual(M)
print("\n2x2 공식으로 직접 계산한 역행렬:")
print(M_inv_manual)
print("NumPy 결과와 같은가?:", np.allclose(M_inv, M_inv_manual))

print("\n💡 역행렬의 SLAM 활용:")
print("   - 좌표계 변환의 역변환 (카메라 → 월드, 월드 → 카메라)")
print("   - 칼만 필터에서 공분산 행렬의 역행렬 계산")

# ============================================================
# Part 3: 고유값 분해 (Eigenvalue Decomposition)
# ============================================================
print("\n" + "=" * 60)
print("Part 3: 고유값 분해 (Eigenvalue Decomposition)")
print("=" * 60)

# 대칭 행렬 예제 (공분산 행렬처럼)
P = np.array([
    [4, 2],
    [2, 3]
])

print("\n대칭 행렬 P (공분산 행렬 형태):")
print(P)

# 고유값 분해
eigenvalues, eigenvectors = np.linalg.eig(P)

print("\n고유값 (eigenvalues):")
print(eigenvalues)
print("\n고유벡터 (eigenvectors) - 각 열이 하나의 고유벡터:")
print(eigenvectors)

# 검증: P * v = λ * v
print("\n검증: P * v₁ = λ₁ * v₁")
v1 = eigenvectors[:, 0]
lambda1 = eigenvalues[0]
Pv1 = P @ v1
lambda1_v1 = lambda1 * v1
print(f"P * v₁ = {Pv1}")
print(f"λ₁ * v₁ = {lambda1_v1}")
print(f"같은가? {np.allclose(Pv1, lambda1_v1)}")

print("\n💡 고유값 분해의 기하학적 의미:")
print("   - 고유벡터: 행렬 변환 후에도 방향이 변하지 않는 특별한 방향")
print("   - 고유값: 그 방향으로 얼마나 늘어나거나 줄어드는지 (스케일)")

print("\n💡 SLAM에서의 활용:")
print("   - 공분산 행렬의 고유벡터 → 불확실성의 주축 방향")
print("   - 고유값 → 각 방향의 불확실성 크기")

# ============================================================
# Part 4: 행렬식 (Determinant) 계산
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 행렬식 (Determinant) 계산")
print("=" * 60)

D = np.array([
    [3, 1, 2],
    [0, 4, 1],
    [5, 2, 3]
])

print("\n행렬 D:")
print(D)

det_D = np.linalg.det(D)
print(f"\ndet(D) = {det_D:.4f}")

# 2x2 행렬식 직접 계산
M2 = np.array([
    [4, 7],
    [2, 6]
])
det_manual = M2[0, 0] * M2[1, 1] - M2[0, 1] * M2[1, 0]
det_numpy = np.linalg.det(M2)
print(f"\n2x2 행렬식 직접 계산: {det_manual}")
print(f"NumPy 계산: {det_numpy:.4f}")

print("\n💡 행렬식의 기하학적 의미:")
print("   - 2D: 두 벡터가 이루는 평행사변형의 넓이")
print("   - 3D: 세 벡터가 이루는 평행육면체의 부피")
print("   - 부호: 양수면 방향 유지, 음수면 뒤집힘")

# ============================================================
# Part 5: SLAM 활용 - 회전 행렬이 직교 행렬인 이유
# ============================================================
print("\n" + "=" * 60)
print("Part 5: SLAM 활용 - 회전 행렬의 성질")
print("=" * 60)

# 30도 회전 행렬 생성
theta = np.radians(30)  # 30도를 라디안으로
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

print(f"\n30도 (2D) 회전 행렬 R:")
print(R)

# 성질 1: 직교 행렬 확인 (R^T * R = I)
RtR = R.T @ R
print("\n[성질 1] R^T * R (단위행렬이어야 함):")
print(RtR)
print(f"단위행렬과 같은가? {np.allclose(RtR, np.eye(2))}")

# 성질 2: 행렬식 = 1 확인
det_R = np.linalg.det(R)
print(f"\n[성질 2] det(R) = {det_R:.4f}")
print(f"det(R) = 1인가? {np.isclose(det_R, 1.0)}")

# 성질 3: 역행렬 = 전치행렬
R_inv = np.linalg.inv(R)
print("\n[성질 3] R의 역행렬:")
print(R_inv)
print("R의 전치행렬:")
print(R.T)
print(f"R^(-1) = R^T인가? {np.allclose(R_inv, R.T)}")

# 3D 회전 행렬 예제
print("\n--- 3D 회전 행렬 ---")

def rotation_matrix_x(angle):
    """X축 회전"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rotation_matrix_y(angle):
    """Y축 회전"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def rotation_matrix_z(angle):
    """Z축 회전"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

Rx = rotation_matrix_x(np.radians(45))
print(f"\nX축 45도 회전 행렬 Rx:")
print(Rx)
print(f"det(Rx) = {np.linalg.det(Rx):.4f}")
print(f"Rx^T * Rx = I? {np.allclose(Rx.T @ Rx, np.eye(3))}")

print("\n💡 회전 행렬이 직교 행렬인 이유:")
print("   - 회전은 길이를 보존하는 변환")
print("   - 직교 행렬은 내적을 보존 → 길이와 각도 보존")
print("   - 따라서 R^T * R = I (정규직교 열벡터)")

print("\n💡 왜 det(R) = 1인가?")
print("   - det > 0: 방향(왼손/오른손 좌표계)을 유지")
print("   - |det| = 1: 부피(크기)를 유지")
print("   - det(R) = 1: 순수한 회전 (반사 없음)")

# ============================================================
# Part 6: 공분산 행렬과 불확실성 표현
# ============================================================
print("\n" + "=" * 60)
print("Part 6: 공분산 행렬과 불확실성 표현 (칼만 필터 예고)")
print("=" * 60)

# 로봇 위치 추정의 불확실성 예제
# 실제 측정 데이터 시뮬레이션
np.random.seed(42)
true_position = np.array([5.0, 3.0])
n_samples = 1000

# 불확실성: x방향 분산 0.5, y방향 분산 2.0, 약간의 상관관계
measurements = np.random.multivariate_normal(
    mean=true_position,
    cov=[[0.5, 0.3], [0.3, 2.0]],
    size=n_samples
)

print(f"\n실제 로봇 위치: {true_position}")
print(f"측정 횟수: {n_samples}")

# 공분산 행렬 계산
cov_matrix = np.cov(measurements.T)
print("\n추정된 공분산 행렬 P:")
print(cov_matrix)

# 고유값 분해로 불확실성 분석
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\n공분산 행렬의 고유값 분해:")
print(f"고유값 (불확실성 크기): {eigenvalues}")
print(f"고유벡터 (불확실성 방향):\n{eigenvectors}")

# 불확실성 타원 매개변수
# 5.991은 카이제곱 분포(Chi-squared distribution)에서 유래한 값입니다.
# - 자유도(DOF) = 2 (2차원 평면 x, y)
# - 신뢰수준(Confidence) = 95% (데이터의 95%를 포함하는 범위)
# - 즉, 2차원 정규분포에서 95% 확률 범위를 그리기 위한 스케일 계수입니다.
scale = np.sqrt(5.991)
print(f"\n95% 신뢰 타원 반지름:")
print(f"  주축 방향: {scale * np.sqrt(eigenvalues[0]):.3f}")
print(f"  부축 방향: {scale * np.sqrt(eigenvalues[1]):.3f}")

print("\n💡 공분산 행렬의 의미:")
print("   - 대각 원소: 각 축의 분산 (cov[0,0]=x분산, cov[1,1]=y분산)")
print("   - 비대각 원소: 변수 간 상관관계")
print("   - 대칭 행렬: cov[i,j] = cov[j,i]")

print("\n💡 칼만 필터에서의 활용:")
print("   - P (상태 공분산): 현재 추정치의 불확실성")
print("   - R (측정 노이즈 공분산): 센서 측정의 불확실성")
print("   - Q (프로세스 노이즈 공분산): 모델 예측의 불확실성")

print("\n💡 시각화 팁:")
print("   - 공분산 행렬 → 불확실성 타원으로 시각화 가능")
print("   - 고유값 작을수록 → 그 방향의 불확실성 작음")

# ============================================================
# Part 7: 전치 행렬의 의미
# ============================================================
print("\n" + "=" * 60)
print("Part 7: 전치 행렬의 의미 - 단순한 행/열 교환 그 이상")
print("=" * 60)

# 7-1: 기본 전치 연산
print("\n--- 7-1: 전치 연산 기본 ---")
A_trans = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("원래 행렬 A (2x3):")
print(A_trans)
print("\n전치 행렬 A^T (3x2):")
print(A_trans.T)
print("\n→ 행과 열이 바뀜: (i,j) 원소가 (j,i)로 이동")

# 7-2: 전치의 핵심 성질 - 내적 관점
print("\n--- 7-2: 전치의 핵심 성질 (내적 관점) ---")
A_inner = np.array([[2, 1], [1, 3]])
x_inner = np.array([1, 2])
y_inner = np.array([3, 1])

# <Ax, y> = <x, A^T y> 검증
left_inner = np.dot(A_inner @ x_inner, y_inner)
right_inner = np.dot(x_inner, A_inner.T @ y_inner)

print(f"A = \n{A_inner}")
print(f"x = {x_inner}, y = {y_inner}")
print(f"\n<Ax, y> = {left_inner}")
print(f"<x, A^T y> = {right_inner}")
print(f"같은가? {left_inner == right_inner}")
print("\n💡 핵심: A^T는 A의 '듀얼(dual)' 변환 - 같은 변환을 반대쪽에서 바라본 것")

# 7-3: 일반 행렬에서는 전치 ≠ 역행렬
print("\n--- 7-3: 일반 행렬에서: 전치 ≠ 역행렬 ---")
M_general = np.array([[1, 2], [3, 4]])

print("행렬 M =")
print(M_general)
print("\n전치 행렬 M^T =")
print(M_general.T)
print("\n역행렬 M^(-1) =")
print(np.linalg.inv(M_general))
print("\n→ 전치와 역행렬이 완전히 다름!")

# 7-4: 직교 행렬에서는 전치 = 역행렬 (특별한 경우!)
print("\n--- 7-4: 직교/회전 행렬에서: 전치 = 역행렬 ---")
theta_trans = np.radians(30)
R_trans = np.array([
    [np.cos(theta_trans), -np.sin(theta_trans)],
    [np.sin(theta_trans),  np.cos(theta_trans)]
])

print(f"30도 회전 행렬 R =")
print(R_trans)
print("\nR^T =")
print(R_trans.T)
print("\nR^(-1) =")
print(np.linalg.inv(R_trans))
print(f"\nR^T = R^(-1)? {np.allclose(R_trans.T, np.linalg.inv(R_trans))}")

# 7-5: 회전 행렬에서 전치의 기하학적 의미
print("\n--- 7-5: 회전 행렬에서 전치 = 역회전 ---")
theta_30 = np.radians(30)
theta_neg30 = np.radians(-30)

R_30 = np.array([
    [np.cos(theta_30), -np.sin(theta_30)],
    [np.sin(theta_30),  np.cos(theta_30)]
])

R_neg30 = np.array([
    [np.cos(theta_neg30), -np.sin(theta_neg30)],
    [np.sin(theta_neg30),  np.cos(theta_neg30)]
])

print("R(30°) =")
print(R_30)
print("\nR(30°)^T =")
print(R_30.T)
print("\nR(-30°) =")
print(R_neg30)
print(f"\nR(30°)^T = R(-30°)? {np.allclose(R_30.T, R_neg30)}")

print("\n💡 핵심: 회전 행렬의 전치 = 반대 방향 회전 = 역회전!")

# 7-6: SLAM에서의 실용적 의미
print("\n--- 7-6: SLAM에서의 활용 ---")
print("""
좌표계 변환에서:

  월드 좌표계 ──R──► 카메라 좌표계
               ◄──R^T──

  R: 월드 → 카메라 변환
  R^T: 카메라 → 월드 변환 (역행렬 계산 없이 빠르게!)
  
💡 계산 효율성:
   - 역행렬 계산: O(n³) 복잡도
   - 전치 연산: O(1) 또는 O(n²) 복사
   - 회전 행렬에서 R^T = R^(-1) 이므로 역변환이 매우 빠름!
""")

# 7-7: 왜 직교 행렬에서만 R^T = R^(-1)인가?
print("--- 7-7: 왜 직교 행렬에서만 R^T = R^(-1)인가? ---")
print("""
직교 행렬의 열벡터들은:
1. 길이가 1 (단위벡터)
2. 서로 수직 (내적 = 0)

이런 '정규직교' 구조 덕분에:
R^T @ R = I (각 열벡터끼리 내적하면 자기자신과는 1, 다른 것과는 0)

따라서 R^T가 곧 R^(-1)이 됨!
""")

print("검증: 열벡터들이 정규직교인지 확인")
col1 = R_30[:, 0]
col2 = R_30[:, 1]
print(f"첫째 열벡터: {col1}, 크기: {np.linalg.norm(col1):.4f}")
print(f"둘째 열벡터: {col2}, 크기: {np.linalg.norm(col2):.4f}")
print(f"두 열벡터의 내적(직교하면 0): {np.dot(col1, col2):.10f}")

# ============================================================
# 보너스: 간단한 2D 시각화 코드 (matplotlib 있으면 실행)
# ============================================================
print("\n" + "=" * 60)
print("보너스: 시각화 예제 코드")
print("=" * 60)

visualization_code = """
# 아래 코드를 Jupyter Notebook이나 별도 스크립트에서 실행하세요

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

# Data (using measurements generated above)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Geometric meaning of matrix multiplication
ax1 = axes[0]
# Original vectors
vectors = np.array([[1, 0], [0, 1], [1, 1]]).T
# Transformation matrix
A = np.array([[2, 1], [1, 2]])
# Transformed vectors
transformed = A @ vectors

colors = ['red', 'blue', 'green']
labels = ['e1', 'e2', 'e1+e2']
for i in range(3):
    ax1.arrow(0, 0, vectors[0, i], vectors[1, i], head_width=0.1, 
              color=colors[i], linestyle='--', alpha=0.5, label=f'Original {labels[i]}')
    ax1.arrow(0, 0, transformed[0, i], transformed[1, i], head_width=0.1,
              color=colors[i], label=f'Transformed {labels[i]}')

ax1.set_xlim(-1, 4)
ax1.set_ylim(-1, 4)
ax1.set_aspect('equal')
ax1.grid(True)
ax1.legend()
ax1.set_title('Geometric Meaning of Matrix Transformation')

# 2. Covariance matrix and uncertainty ellipse
ax2 = axes[1]
# Plot measurement points
ax2.scatter(measurements[:, 0], measurements[:, 1], alpha=0.3, s=5, label='Measurements')
ax2.scatter(*true_position, color='red', s=100, marker='x', label='True Position')

# Draw uncertainty ellipse
angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
for n_std in [1, 2, 3]:  # 1, 2, 3 standard deviations
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])
    ellipse = Ellipse(true_position, width, height, angle=angle,
                     fill=False, color=f'C{n_std}', linewidth=2,
                     label=f'{n_std}sigma Ellipse')
    ax2.add_patch(ellipse)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_aspect('equal')
ax2.legend()
ax2.set_title('Uncertainty Ellipse from Covariance Matrix')

plt.tight_layout()
plt.savefig('linear_algebra_visualization.png', dpi=150)
plt.show()
"""

print(visualization_code)

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 60)
print("📝 Part 1-7 정리")
print("=" * 60)

print("""
✅ Part 1 - 행렬 곱셈
   - 행렬 곱셈 = 선형 변환의 합성
   - (AB)의 (i,j) = A의 i행과 B의 j열의 내적

✅ Part 2 - 역행렬
   - A * A^(-1) = I
   - det(A) ≠ 0 일 때만 존재
   - SLAM: 좌표계 역변환에 활용

✅ Part 3 - 고유값 분해
   - A * v = λ * v
   - 고유벡터: 변환 후에도 방향 유지되는 특별한 벡터
   - 고유값: 그 방향의 스케일 변화량

✅ Part 4 - 행렬식
   - 기하학적 의미: 부피/넓이의 변화율
   - det = 0이면 역행렬 없음 (특이 행렬)

✅ Part 5 - 회전 행렬
   - 직교 행렬: R^T * R = I
   - det(R) = 1: 순수 회전 (반사 없음)
   - R^(-1) = R^T: 역행렬 = 전치행렬

✅ Part 6 - 공분산 행렬
   - 불확실성 표현의 핵심
   - 고유값 분해 → 불확실성 주축과 크기
   - 칼만 필터, Bundle Adjustment에서 필수

✅ Part 7 - 전치 행렬
   - 전치: 행과 열을 바꾸는 연산 (A^T)_ij = A_ji
   - 내적 성질: <Ax, y> = <x, A^T y>
   - 일반 행렬: 전치 ≠ 역행렬
   - 직교/회전 행렬: 전치 = 역행렬 (R^T = R^(-1))
   - 회전 행렬의 전치 = 역회전 (반대 방향 회전)
   - SLAM: 좌표계 역변환을 빠르게 계산
""")

print("\n🎯 다음 단계:")
print("   - Phase 1.md의 Week 2 체크리스트 항목들을 [x]로 체크")
print("   - Week 3: SVD 집중 학습으로 이동")
