"""
Phase 1 - Week 7: 최소자승법 (Least Squares)
=============================================
정규방정식과 직선 피팅 실습

학습 목표:
1. 최소자승 문제 정의
2. 정규방정식 유도
3. 직선 피팅 실습
4. SLAM 활용 이해
"""

import numpy as np
np.set_printoptions(precision=4, suppress=True)

print("=" * 60)
print("Phase 1 - Week 7: 최소자승법 (Least Squares)")
print("=" * 60)

# ============================================================
# Part 1: 최소자승 문제
# ============================================================
print("\n" + "=" * 60)
print("Part 1: 최소자승 문제란?")
print("=" * 60)

print("""
과결정 시스템: Ax = b (방정식 > 미지수)

정확한 해가 없을 때:
→ min ||Ax - b||² 를 만족하는 x 찾기
""")

# 예제: 3개 방정식, 2개 미지수
A = np.array([
    [1, 1],
    [2, 1],
    [3, 1]
])
b = np.array([2.1, 3.9, 6.2])

print("A (설계 행렬):")
print(A)
print(f"\nb (관측값): {b}")
print("\n문제: Ax = b의 최소자승 해 x 찾기")

# ============================================================
# Part 2: 정규방정식
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 정규방정식 (Normal Equation)")
print("=" * 60)

print("""
비용 함수: J(x) = ||Ax - b||²

최소화 조건: ∂J/∂x = 0
→ 정규방정식: AᵀAx = Aᵀb
→ 해: x = (AᵀA)⁻¹Aᵀb
""")

def least_squares_normal(A, b):
    """정규방정식으로 최소자승 해 계산"""
    AtA = A.T @ A
    Atb = A.T @ b
    x = np.linalg.solve(AtA, Atb)  # inv 대신 solve 사용 (수치 안정)
    return x

x_normal = least_squares_normal(A, b)
print(f"정규방정식 해: x = {x_normal}")

# NumPy 비교
x_numpy, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
print(f"NumPy lstsq 해: x = {x_numpy}")
print(f"일치: {np.allclose(x_normal, x_numpy)}")

# ============================================================
# Part 3: 잔차 분석
# ============================================================
print("\n" + "=" * 60)
print("Part 3: 잔차 (Residual) 분석")
print("=" * 60)

b_pred = A @ x_numpy
residual = b - b_pred

print(f"예측값: {b_pred}")
print(f"관측값: {b}")
print(f"잔차 (관측-예측): {residual}")
print(f"\n잔차 제곱합: {np.sum(residual**2):.6f}")

# ============================================================
# Part 4: 직선 피팅 예제
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 직선 피팅 (y = ax + b)")
print("=" * 60)

# 데이터 생성: y = 2x + 1 + 노이즈
np.random.seed(42)
x_data = np.array([1, 2, 3, 4, 5])
y_data = 2 * x_data + 1 + np.random.randn(5) * 0.3

print(f"x 데이터: {x_data}")
print(f"y 데이터: {y_data}")

# 설계 행렬: [1, x] (절편, 기울기)
A_line = np.column_stack([np.ones(len(x_data)), x_data])
print(f"\n설계 행렬 A:")
print(A_line)

# 최소자승 해
params = np.linalg.lstsq(A_line, y_data, rcond=None)[0]
intercept, slope = params

print(f"\n피팅 결과: y = {slope:.4f}x + {intercept:.4f}")
print(f"실제 모델: y = 2x + 1")

# ============================================================
# Part 5: SVD로 최소자승 해
# ============================================================
print("\n" + "=" * 60)
print("Part 5: SVD로 최소자승 해")
print("=" * 60)

print("""
SVD 방법: x = V Σ⁺ Uᵀ b
(Σ⁺: 유사역행렬, 각 σᵢ → 1/σᵢ)
""")

U, S, Vt = np.linalg.svd(A_line, full_matrices=False)

# 유사역행렬 계산
S_inv = np.diag(1 / S)
A_pinv = Vt.T @ S_inv @ U.T

x_svd = A_pinv @ y_data
print(f"SVD 해: {x_svd}")
print(f"lstsq 해: {params}")
print(f"일치: {np.allclose(x_svd, params)}")

# ============================================================
# Part 6: SLAM 활용 예시
# ============================================================
print("\n" + "=" * 60)
print("Part 6: SLAM에서의 최소자승")
print("=" * 60)

print("""
SLAM 문제들의 최소자승 형태:

1. PnP 문제
   - 3D-2D 대응점에서 카메라 포즈 추정
   - min Σ||π(T·P) - p||²

2. Triangulation
   - Ax = 0 형태 (동차 시스템)
   - SVD로 null space 찾기

3. Bundle Adjustment
   - 카메라 포즈 + 3D 점 동시 최적화
   - 매우 큰 최소자승 문제 (sparse)

4. 재투영 오차
   - 비선형 → Gauss-Newton으로 선형화
""")

# 간단한 예: 2D 점 위치 추정
print("\n--- 예제: 여러 방향에서 거리 측정으로 위치 추정 ---")

# 진짜 위치
true_pos = np.array([3, 4])

# 측정 방향과 거리 (노이즈 포함)
directions = np.array([
    [1, 0],
    [0, 1],
    [1, 1] / np.sqrt(2),
    [1, -1] / np.sqrt(2)
])
measured_dist = directions @ true_pos + np.random.randn(4) * 0.1

# 최소자승으로 위치 추정
estimated_pos = np.linalg.lstsq(directions, measured_dist, rcond=None)[0]

print(f"실제 위치: {true_pos}")
print(f"추정 위치: {estimated_pos}")
print(f"오차: {np.linalg.norm(estimated_pos - true_pos):.4f}")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 60)
print("📝 Week 7 정리")
print("=" * 60)
print("""
✅ 최소자승 문제
   - min ||Ax - b||²
   - 과결정 시스템 (방정식 > 미지수)

✅ 정규방정식
   - AᵀAx = Aᵀb
   - x = (AᵀA)⁻¹Aᵀb

✅ 잔차
   - r = b - Ax (측정 - 예측)
   - 비용 = ||r||²

✅ 풀이 방법
   - 정규방정식 (직접)
   - SVD (수치 안정)
   - np.linalg.lstsq (실용)

✅ SLAM 활용
   - PnP, Triangulation, BA 모두 최소자승

🎯 다음: least_squares_quiz.py → Week 8: 비선형 최적화
""")
