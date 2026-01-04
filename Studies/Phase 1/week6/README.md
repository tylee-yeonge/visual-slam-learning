# Week 6: Lie 군/대수 기초

## 📌 개요

> ⚠️ **참고**: 이번 주는 가볍게 훑고, VINS-Fusion 코드 (Phase 5)에서 필요할 때 돌아와서 심화 학습

**Lie 군/대수**는 회전과 변환을 **최적화 가능한 형태**로 표현하는 수학적 프레임워크입니다. 회전 행렬은 9개 파라미터이지만 실제 자유도는 3이고, 쿼터니언도 4개 파라미터에 단위 제약이 있습니다. 최적화할 때 이런 **제약 조건**을 어떻게 처리할 것인가가 핵심 문제입니다.

## 🎯 학습 목표

1. 왜 회전/변환의 "over-parameterized" 문제가 있는지 이해
2. SO(3), SE(3)의 기본 개념 이해
3. Lie 대수(접선 공간)의 직관적 이해
4. exp/log 매핑의 의미 파악
5. Sophus 라이브러리 기본 사용법

## 📚 사전 지식

- Week 4: 회전 표현
- Week 5: SE(3) 변환

## ⏱️ 예상 학습 시간

| 항목 | 시간 |
|------|------|
| 이론 (가볍게) | 2시간 |
| 실습 예제 | 1-2시간 |
| **총 소요시간** | **3-4시간** |

---

## 📖 핵심 개념

### 1. 왜 Lie 군/대수가 필요한가?

#### Over-parameterized 문제

| 표현 | 파라미터 수 | 실제 자유도 | 제약 |
|------|------------|------------|------|
| 회전 행렬 R | 9 | 3 | RᵀR=I, det=1 (6개 제약) |
| 쿼터니언 q | 4 | 3 | ‖q‖=1 (1개 제약) |
| SE(3) T | 16 | 6 | 여러 제약 |

#### 최적화에서의 문제

```python
# 일반적인 최적화: x ← x + Δx
R_new = R + ΔR  # ❌ R+ΔR은 회전 행렬이 아닐 수 있음!
q_new = q + Δq  # ❌ 단위 쿼터니언이 아닐 수 있음!
```

**해결책**: Lie 대수에서 업데이트 후 manifold로 투영

```python
# Lie 대수 방식
R_new = exp(Δξ) @ R  # ✅ 항상 회전 행렬
```

---

### 2. 군(Group)의 기본 개념

**군(Group)**: 연산이 정의된 집합으로, 다음 성질 만족:
- **닫힘**: 두 원소의 연산 결과도 그 집합의 원소
- **결합법칙**: (a·b)·c = a·(b·c)
- **항등원**: 연산해도 바뀌지 않는 원소 존재
- **역원**: 역연산이 가능한 원소 존재

#### SO(3): Special Orthogonal Group

```
SO(3) = {R ∈ ℝ³ˣ³ | RᵀR = I, det(R) = 1}
```
- **의미**: 모든 3D 회전 행렬의 집합
- **연산**: 행렬 곱셈
- **항등원**: 단위 행렬 I
- **역원**: Rᵀ (전치)

#### SE(3): Special Euclidean Group

```
SE(3) = {T ∈ ℝ⁴ˣ⁴ | T = [R|t], R ∈ SO(3)}
```
- **의미**: 모든 강체 변환의 집합
- **자유도**: 6 (회전 3 + 이동 3)

---

### 3. Lie 대수: 접선 공간

**Lie 대수**: 군의 **접선 공간**에서의 표현

```
             Lie 군 (SO(3), SE(3))
                    ↑
        exp 매핑   │   │   log 매핑
                    ↓
             Lie 대수 (so(3), se(3))
```

#### so(3): SO(3)의 Lie 대수

```python
# 3D 벡터 ω = [ω1, ω2, ω3] (회전축 × 각도)
# 반대칭 행렬 ω^ (skew-symmetric)
def skew(w):
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

# ω^ ∈ so(3) (3x3 반대칭 행렬)
```

#### se(3): SE(3)의 Lie 대수

```
ξ = [ρ, φ]^T  (6차원 벡터)
- ρ: 평행이동 관련 (3차원)
- φ: 회전 관련 (3차원)
```

---

### 4. Exponential / Log 매핑

#### exp: Lie 대수 → Lie 군

```python
def exp_so3(omega):
    """so(3) → SO(3) (Rodrigues 공식)"""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3)
    
    axis = omega / theta
    K = skew(axis)
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*K@K
    return R
```

#### log: Lie 군 → Lie 대수

```python
def log_so3(R):
    """SO(3) → so(3)"""
    theta = np.arccos((np.trace(R) - 1) / 2)
    if theta < 1e-10:
        return np.zeros(3)
    
    omega_hat = (R - R.T) / (2 * np.sin(theta)) * theta
    return np.array([omega_hat[2,1], omega_hat[0,2], omega_hat[1,0]])
```

---

### 5. 최적화에서의 활용

#### 기존 방식 (제약 있는 최적화)

```python
# 쿼터니언 방식: 단위 제약 유지 필요
q_new = q + Δq
q_new = q_new / np.linalg.norm(q_new)  # 정규화
```

#### Lie 대수 방식 (제약 없는 최적화)

```python
# 1. Lie 대수에서 업데이트 계산 (3개 파라미터)
Δξ = compute_update()  # 3차원 벡터

# 2. exp 매핑으로 회전 행렬 업데이트
R_new = exp_so3(Δξ) @ R  # 항상 유효한 회전 행렬!
```

**장점**:
- 최적화 변수가 실제 자유도와 같음 (3개)
- 제약 조건 처리 없이 항상 유효한 회전

---

## 🔧 Sophus 라이브러리

### C++에서 사용

```cpp
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>

// SO(3) 생성
Sophus::SO3d R = Sophus::SO3d::exp(omega);  // Lie 대수에서
Sophus::SO3d R = Sophus::SO3d::rotZ(0.5);   // Z축 회전

// SE(3) 생성  
Sophus::SE3d T(R, t);  // 회전 + 평행이동

// Log 매핑
Eigen::Vector3d omega = R.log();
Eigen::Vector6d xi = T.log();
```

### Python에서 사용 (scipy)

```python
from scipy.spatial.transform import Rotation

# 회전 벡터 (axis-angle) → 회전 행렬
R = Rotation.from_rotvec([0, 0, np.pi/4]).as_matrix()

# 회전 행렬 → 회전 벡터
rotvec = Rotation.from_matrix(R).as_rotvec()
```

---

## 🤖 SLAM에서의 활용

### VINS-Fusion에서

```cpp
// IMU 사전적분에서 회전 업데이트
Eigen::Quaterniond q = q * Utility::deltaQ(gyr * dt);

// deltaQ 내부: Lie 대수 사용
static Quaterniond deltaQ(const Vector3d &theta) {
    // exp 매핑과 동등
    ...
}
```

### Bundle Adjustment에서

- 카메라 포즈 최적화 시 6 DOF 대신 se(3)의 6차원 벡터 사용
- Jacobian 계산이 간단해짐

---

## 💻 실습 파일

| 파일 | 내용 |
|------|------|
| `lie_basics.py` | Lie 군/대수 기본 연산 |
| `lie_quiz.py` | 개념 확인 퀴즈 |

---

## ✅ 학습 완료 체크리스트

- [ ] 회전 행렬이 왜 "over-parameterized"인지 설명 가능
- [ ] SO(3), SE(3)가 무엇인지 알고 있다
- [ ] Lie 대수가 "접선 공간"이라는 것을 이해했다
- [ ] exp/log 매핑의 역할을 알고 있다
- [ ] 왜 최적화에서 Lie 대수를 쓰면 편한지 설명 가능

---

## 🔗 다음 단계

Week 6 완료 후 → **Week 7: 최소자승법**으로 이동
- 비용 함수와 잔차
- 정규방정식
- SLAM의 기본 최적화 형태
