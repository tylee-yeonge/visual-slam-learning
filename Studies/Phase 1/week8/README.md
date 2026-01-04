# Week 8: 비선형 최적화 (Nonlinear Optimization)

## 📌 개요

실제 SLAM 문제는 대부분 **비선형**입니다. 카메라 투영, 회전 연산 등이 모두 비선형이기 때문입니다. 이번 주에는 **Gauss-Newton**, **Levenberg-Marquardt** 알고리즘을 학습하고, SLAM에서 실제로 사용되는 **Ceres Solver**를 실습합니다.

## 🎯 학습 목표

1. 비선형 문제의 선형화 이해
2. Jacobian 행렬의 의미
3. Gauss-Newton 알고리즘
4. Levenberg-Marquardt 알고리즘
5. Ceres Solver 기본 사용법

## ⏱️ 예상 학습 시간: **5-7시간**

---

## 📖 핵심 개념

### 1. 비선형 최소자승

```
min Σ ||f(x) - z||²
 x

f(x): 비선형 함수
z: 측정값
```

선형화 (1차 Taylor 전개):
```
f(x + Δx) ≈ f(x) + J·Δx

J = ∂f/∂x (Jacobian 행렬)
```

### 2. Gauss-Newton

```
반복:
  1. Jacobian J = ∂f/∂x 계산
  2. 잔차 r = z - f(x) 계산
  3. 정규방정식: JᵀJ·Δx = Jᵀr
  4. 업데이트: x ← x + Δx
```

```python
def gauss_newton(f, J_func, x0, z, max_iter=10):
    x = x0.copy()
    for i in range(max_iter):
        J = J_func(x)
        r = z - f(x)
        dx = np.linalg.solve(J.T @ J, J.T @ r)
        x = x + dx
        if np.linalg.norm(dx) < 1e-8:
            break
    return x
```

### 3. Levenberg-Marquardt

Gauss-Newton의 문제: 초기값이 멀면 발산

LM 해결책:
```
(JᵀJ + λI)·Δx = Jᵀr

λ 크면 → Gradient Descent (안정적)
λ 작으면 → Gauss-Newton (빠름)
```

---

## 🔧 Ceres Solver

### C++ 예제 (곡선 피팅)

```cpp
struct CostFunctor {
    CostFunctor(double x, double y) : x_(x), y_(y) {}
    
    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        // y = a*exp(b*x)
        residual[0] = y_ - params[0] * ceres::exp(params[1] * x_);
        return true;
    }
    
private:
    const double x_, y_;
};

// Problem 설정
ceres::Problem problem;
for (int i = 0; i < N; ++i) {
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<CostFunctor, 1, 2>(
            new CostFunctor(x[i], y[i])),
        nullptr, params);
}

// 최적화 실행
ceres::Solve(options, &problem, &summary);
```

### Python (scipy)

```python
from scipy.optimize import least_squares

def residual(params, x, y):
    a, b = params
    return y - a * np.exp(b * x)

result = least_squares(residual, x0=[1, 0], args=(x_data, y_data))
```

---

## 🤖 SLAM에서의 활용

| 문제 | Ceres 역할 |
|------|-----------|
| **Bundle Adjustment** | 포즈 + 3D점 동시 최적화 |
| **VIO** | IMU 사전적분 최적화 |
| **PnP Refinement** | 초기 PnP 해 정제 |

### VINS-Fusion에서

```cpp
ceres::Problem problem;

// 카메라 포즈 최적화
for (auto& frame : keyframes) {
    problem.AddParameterBlock(frame.pose, 7, new PoseLocalParameterization());
}

// 재투영 오차
for (auto& obs : observations) {
    problem.AddResidualBlock(
        new ReprojectionError(obs),
        loss_function,
        frame_pose, point_3d);
}
```

---

## 💻 실습 파일

| 파일 | 내용 |
|------|------|
| `nonlinear_basics.py` | Gauss-Newton, 곡선 피팅 |
| `nonlinear_quiz.py` | 개념 퀴즈 |

---

## ✅ 체크리스트

- [ ] 비선형 문제의 선형화 이해
- [ ] Jacobian의 의미 이해
- [ ] Gauss-Newton 알고리즘 이해
- [ ] LM이 GN보다 나은 점 설명 가능
- [ ] Ceres의 CostFunction, Problem 역할 이해

---

## 🔗 Phase 1 완료!

Week 8 완료 후 → **Phase 2: 컴퓨터 비전 기초**로 이동
