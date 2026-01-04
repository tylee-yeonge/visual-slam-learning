"""
Phase 1 - Week 8: 비선형 최적화
================================
Gauss-Newton, Levenberg-Marquardt 실습

학습 목표:
1. 비선형 문제의 선형화
2. Jacobian 계산
3. Gauss-Newton 구현
4. LM과 비교
"""

import numpy as np
np.set_printoptions(precision=4, suppress=True)

print("=" * 60)
print("Phase 1 - Week 8: 비선형 최적화")
print("=" * 60)

# ============================================================
# Part 1: 비선형 문제
# ============================================================
print("\n" + "=" * 60)
print("Part 1: 비선형 문제")
print("=" * 60)

print("""
비선형 최소자승: min Σ||f(x) - z||²

예제: y = a * exp(b * x) 피팅
- 파라미터: [a, b]
- 비선형: exp 함수
""")

# 실제 데이터 생성
np.random.seed(42)
x_data = np.linspace(0, 2, 10)
a_true, b_true = 2.5, 0.8
y_data = a_true * np.exp(b_true * x_data) + np.random.randn(10) * 0.2

print(f"실제 파라미터: a={a_true}, b={b_true}")
print(f"데이터 포인트: {len(x_data)}개")

# ============================================================
# Part 2: Jacobian 계산
# ============================================================
print("\n" + "=" * 60)
print("Part 2: Jacobian 행렬")
print("=" * 60)

def f(params, x):
    """모델 함수: y = a * exp(b * x)"""
    a, b = params
    return a * np.exp(b * x)

def jacobian(params, x):
    """Jacobian: ∂f/∂[a,b]
    
    ∂f/∂a = exp(b*x)
    ∂f/∂b = a * x * exp(b*x)
    """
    a, b = params
    J = np.zeros((len(x), 2))
    J[:, 0] = np.exp(b * x)           # ∂f/∂a
    J[:, 1] = a * x * np.exp(b * x)   # ∂f/∂b
    return J

# 초기값에서 Jacobian
params_init = np.array([1.0, 0.5])
J = jacobian(params_init, x_data)
print(f"초기 파라미터: {params_init}")
print(f"Jacobian 크기: {J.shape} (데이터수 x 파라미터수)")
print(f"Jacobian[:3]:\n{J[:3]}")

# ============================================================
# Part 3: Gauss-Newton 알고리즘
# ============================================================
print("\n" + "=" * 60)
print("Part 3: Gauss-Newton 알고리즘")
print("=" * 60)

def gauss_newton(x_data, y_data, params_init, max_iter=20, tol=1e-8):
    """Gauss-Newton 최적화
    
    반복:
    1. J = Jacobian 계산
    2. r = 잔차 (y - f(x))
    3. (JᵀJ)Δp = Jᵀr
    4. p ← p + Δp
    """
    params = params_init.copy()
    
    print("iter | cost     | a      | b      | |Δp|")
    print("-" * 50)
    
    for i in range(max_iter):
        # 예측 및 잔차
        y_pred = f(params, x_data)
        r = y_data - y_pred
        cost = np.sum(r**2)
        
        # Jacobian
        J = jacobian(params, x_data)
        
        # 정규방정식: JᵀJ·Δp = Jᵀr
        JtJ = J.T @ J
        Jtr = J.T @ r
        dp = np.linalg.solve(JtJ, Jtr)
        
        # 업데이트
        params = params + dp
        
        print(f"{i:4d} | {cost:8.4f} | {params[0]:.4f} | {params[1]:.4f} | {np.linalg.norm(dp):.2e}")
        
        if np.linalg.norm(dp) < tol:
            print("수렴!")
            break
    
    return params

params_gn = gauss_newton(x_data, y_data, params_init)
print(f"\nGauss-Newton 결과: a={params_gn[0]:.4f}, b={params_gn[1]:.4f}")
print(f"실제값: a={a_true}, b={b_true}")

# ============================================================
# Part 4: Levenberg-Marquardt
# ============================================================
print("\n" + "=" * 60)
print("Part 4: Levenberg-Marquardt")
print("=" * 60)

print("""
LM 알고리즘:
(JᵀJ + λI)Δp = Jᵀr

λ 크면 → Gradient Descent (안정적, 느림)
λ 작으면 → Gauss-Newton (빠름, 불안정 가능)
""")

def levenberg_marquardt(x_data, y_data, params_init, max_iter=20, lam=0.01):
    """Levenberg-Marquardt 최적화"""
    params = params_init.copy()
    
    for i in range(max_iter):
        y_pred = f(params, x_data)
        r = y_data - y_pred
        J = jacobian(params, x_data)
        
        # LM 정규방정식: (JᵀJ + λI)Δp = Jᵀr
        JtJ = J.T @ J
        H = JtJ + lam * np.eye(2)  # Damping term 추가
        dp = np.linalg.solve(H, J.T @ r)
        
        # 새 파라미터로 비용 계산
        params_new = params + dp
        cost_old = np.sum(r**2)
        cost_new = np.sum((y_data - f(params_new, x_data))**2)
        
        if cost_new < cost_old:
            params = params_new
            lam /= 2  # 성공: λ 감소 (GN에 가깝게)
        else:
            lam *= 2  # 실패: λ 증가 (GD에 가깝게)
    
    return params

params_lm = levenberg_marquardt(x_data, y_data, np.array([0.5, 0.1]))
print(f"LM 결과: a={params_lm[0]:.4f}, b={params_lm[1]:.4f}")

# ============================================================
# Part 5: scipy.optimize 비교
# ============================================================
print("\n" + "=" * 60)
print("Part 5: scipy 활용 (실무)")
print("=" * 60)

from scipy.optimize import least_squares

def residual_scipy(params, x, y):
    return y - params[0] * np.exp(params[1] * x)

result = least_squares(residual_scipy, [1, 0.5], args=(x_data, y_data))
print(f"scipy 결과: a={result.x[0]:.4f}, b={result.x[1]:.4f}")
print(f"수렴: {result.success}")

# ============================================================
# Part 6: SLAM 활용
# ============================================================
print("\n" + "=" * 60)
print("Part 6: SLAM에서의 활용")
print("=" * 60)

print("""
SLAM 최적화 구조:

1. 비용 함수 정의
   cost = Σ ||재투영_오차||² + Σ ||IMU_오차||²

2. Jacobian 계산
   - 자동 미분 (Ceres AutoDiff)
   - 또는 분석적 미분

3. 희소 행렬 활용
   - BA: JᵀJ가 sparse
   - Schur complement로 효율적 풀이

4. VINS-Fusion에서
   - Ceres Solver 사용
   - Pose는 SE(3)로 표현 (Lie 대수)
""")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 60)
print("📝 Week 8 정리 & Phase 1 완료!")
print("=" * 60)
print("""
✅ 비선형 최소자승
   - f(x + Δx) ≈ f(x) + J·Δx (선형화)
   - Jacobian: J = ∂f/∂x

✅ Gauss-Newton
   - (JᵀJ)Δx = Jᵀr
   - 빠르지만 초기값 민감

✅ Levenberg-Marquardt
   - (JᵀJ + λI)Δx = Jᵀr
   - λ로 안정성-속도 균형

✅ Ceres Solver (C++)
   - SLAM 표준 최적화 라이브러리
   - AutoDiff, Problem, CostFunction

🎉 Phase 1 (수학 핵심) 완료!
   → Phase 2: 컴퓨터 비전 기초
""")
