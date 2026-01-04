# Week 2: NumPy/Eigen 기본 연산 실습

## 📌 개요

Week 1에서 3Blue1Brown 영상을 통해 선형대수의 **직관**을 얻었다면, 이번 주에는 **직접 코드로 구현**하면서 개념을 확실히 다집니다. 행렬 곱셈, 역행렬, 고유값 분해, 행렬식 등 핵심 연산을 NumPy로 실습하고, 이것이 SLAM에서 어떻게 활용되는지 연결합니다.

특히 **회전 행렬의 성질**과 **공분산 행렬**은 Visual SLAM의 거의 모든 알고리즘에서 등장하므로 이번 주에 확실히 이해해야 합니다.

## 🎯 학습 목표

1. NumPy로 기본 선형대수 연산 수행하기
2. 행렬 곱셈의 기하학적 의미 이해
3. 고유값/고유벡터를 코드로 계산하고 의미 파악
4. 회전 행렬이 직교 행렬인 이유 이해
5. 공분산 행렬로 불확실성 표현하기 (칼만 필터 예고)

## 📚 사전 지식

- Week 1 완료 (3Blue1Brown Essence of Linear Algebra)
- Python 기본 문법
- NumPy 기초 (`import numpy as np`, 배열 생성)

## ⏱️ 예상 학습 시간

| 항목 | 시간 |
|------|------|
| 이론 복습 | 1시간 |
| 실습 예제 | 2-3시간 |
| SLAM 연결 이해 | 1-2시간 |
| 퀴즈 풀이 | 1시간 |
| **총 소요시간** | **5-7시간** |

---

## 📖 핵심 개념

### 1. 행렬 곱셈 (Matrix Multiplication)

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 행렬 곱셈
C = A @ B  # 또는 np.dot(A, B)
```

**기하학적 의미**: 행렬 곱셈 `AB`는 두 선형 변환의 **합성(composition)**입니다.
- 먼저 B로 변환, 그 다음 A로 변환

### 2. 역행렬 (Inverse Matrix)

```python
A = np.array([[4, 7], [2, 6]])
A_inv = np.linalg.inv(A)

# 검증: A @ A^(-1) = I
print(A @ A_inv)  # [[1, 0], [0, 1]]
```

**SLAM 활용**: 좌표계 변환의 역변환
- 카메라 → 월드, 월드 → 카메라 변환

### 3. 고유값 분해 (Eigenvalue Decomposition)

```python
A = np.array([[4, 2], [2, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
```

**핵심 직관**:
- **고유벡터**: 행렬 변환 후에도 **방향이 변하지 않는** 특별한 벡터
- **고유값**: 그 방향으로 **얼마나 스케일이 변하는지**

**SLAM 활용**: 공분산 행렬의 고유값 분해 → 불확실성의 주축 방향과 크기

### 4. 행렬식 (Determinant)

```python
A = np.array([[3, 1], [2, 4]])
det_A = np.linalg.det(A)
```

**기하학적 의미**:
- 2D: 두 벡터가 이루는 **평행사변형의 면적** 변화율
- 3D: 세 벡터가 이루는 **평행육면체의 부피** 변화율

---

## 🔄 회전 행렬 (Rotation Matrix) - 기초부터 이해하기

### 회전이란 무엇인가?

**회전**은 물체의 **위치는 그대로** 두고 **방향만 바꾸는** 변환입니다.

예를 들어:
- 시계 바늘이 원점(중심)을 기준으로 도는 것
- 로봇이 제자리에서 방향을 바꾸는 것
- 카메라가 고정된 채로 다른 방향을 바라보는 것

### 2D에서 점을 회전시키기

점 (1, 0)을 원점 기준으로 **θ도** 회전시키면 어디로 갈까요?

```
       y
       ↑
       |    ● (cos θ, sin θ) ← 회전 후
       |   /
       |  / θ
       | /
-------●------→ x
     (1,0) 원래 점
```

삼각함수의 정의에 따라:
- 회전 후 x좌표 = cos(θ)
- 회전 후 y좌표 = sin(θ)

### 회전 행렬의 유도

이제 **임의의 점 (x, y)** 를 θ만큼 회전시키면 어디로 갈까요?

**핵심 아이디어**: 점 (x, y)는 두 기저 벡터 (1,0)과 (0,1)의 조합으로 표현됩니다.

```
(x, y) = x·(1, 0) + y·(0, 1)
```

각 기저 벡터를 θ만큼 회전시키면:
- (1, 0) → (cos θ, sin θ)
- (0, 1) → (-sin θ, cos θ)

따라서 (x, y)를 회전하면:
```
x·(cos θ, sin θ) + y·(-sin θ, cos θ)
= (x·cos θ - y·sin θ, x·sin θ + y·cos θ)
```

이것을 행렬로 표현하면:

```
[x']   [cos θ  -sin θ] [x]
[y'] = [sin θ   cos θ] [y]
```

> [!NOTE]
> **회전 행렬 R(θ)**의 각 **열**은 회전된 기저 벡터입니다!
> - 1열: (1,0)이 회전 후 가는 곳
> - 2열: (0,1)이 회전 후 가는 곳

### 왜 "행렬"로 표현하는가?

1. **계산 편의성**: 행렬 곱셈 한 번으로 회전 완료
2. **회전 합성**: 연속 회전 = 행렬 곱셈
3. **역회전**: 역행렬 = 전치행렬 (계산 매우 간단)

```python
import numpy as np

# 30도 회전
theta = np.radians(30)  # 각도를 라디안으로

R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

# 점 (1, 0)을 회전
point = np.array([1, 0])
rotated = R @ point
print(f"(1, 0) → {rotated}")  # [0.866, 0.5]
```

### 3D 회전 행렬

3D에서는 **어떤 축 기준**으로 회전하는지가 중요합니다.

#### X축 회전 (Roll)

```
       y
       ↑
       |
       +---→ x
      /
     ↓ z

X축 기준으로 회전 = y-z 평면에서 회전
```

```python
def rotation_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1,  0,  0],   # x축은 변화 없음
        [0,  c, -s],   # y, z가 회전
        [0,  s,  c]
    ])
```

#### Y축 회전 (Pitch)

```python
def rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c,  0,  s],   # x, z가 회전
        [ 0,  1,  0],   # y축은 변화 없음
        [-s,  0,  c]
    ])
```

#### Z축 회전 (Yaw)

```python
def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s,  0],   # x, y가 회전
        [s,  c,  0],
        [0,  0,  1]    # z축은 변화 없음
    ])
```

### 회전 행렬의 핵심 성질

| 성질 | 수식 | 의미 |
|------|------|------|
| **직교 행렬** | R^T R = I | 열벡터들이 서로 직교하고 크기가 1 |
| **det(R) = 1** | det(R) = +1 | 부피를 늘리거나 줄이지 않음, 반사 없음 |
| **역행렬 = 전치** | R^(-1) = R^T | 역회전 계산이 매우 빠름 |

```python
theta = np.radians(30)
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# 성질 검증
print("R^T @ R =")
print(R.T @ R)  # 단위행렬

print(f"det(R) = {np.linalg.det(R):.4f}")  # 1.0

print("R^(-1) = R^T?:", np.allclose(np.linalg.inv(R), R.T))  # True
```

### 왜 직교 행렬인가? (길이 보존)

회전은 **길이를 바꾸지 않는** 변환입니다.

```
원래 벡터 v의 길이: ||v|| = √(v^T v)

회전 후 길이: ||Rv|| = √((Rv)^T (Rv))
            = √(v^T R^T R v)
            = √(v^T I v)     ← R^T R = I 이므로
            = √(v^T v)
            = ||v||           ← 원래 길이와 같음!
```

### 왜 det(R) = 1인가?

- **det > 0**: 좌표계 방향(오른손/왼손)을 유지
- **|det| = 1**: 부피를 변화시키지 않음
- 순수한 회전 = 방향 유지 + 크기 유지 → **det = 1**

> [!TIP]
> det = -1이면? → 회전 + 반사 (거울에 비친 것처럼)

### SLAM에서 회전 행렬이 중요한 이유

1. **카메라 자세 표현**: 카메라가 어느 방향을 보고 있는가
2. **좌표계 변환**: 월드 좌표계 ↔ 카메라 좌표계
3. **로봇 방향**: 로봇이 어느 방향을 향하고 있는가
4. **IMU 데이터 처리**: 센서 측정값을 월드 좌표로 변환

---

## 📊 공분산 행렬 (Covariance Matrix)

로봇 위치 추정의 **불확실성**을 표현:

```python
# 공분산 행렬
P = np.array([
    [0.04, 0.01],  # x분산, xy공분산
    [0.01, 0.09]   # xy공분산, y분산
])

# 고유값 분해 → 불확실성 타원
eigenvalues, eigenvectors = np.linalg.eig(P)
# eigenvalues: 각 주축 방향의 분산
# eigenvectors: 불확실성 타원의 주축 방향
```

**칼만 필터에서의 역할**:
- **P**: 상태 추정의 불확실성 (예측 후 증가, 관측 후 감소)
- **R**: 센서 측정 노이즈의 공분산
- **Q**: 프로세스 (모델) 노이즈의 공분산

---

## 💻 실습 파일

이 폴더에 포함된 실습 파일:

| 파일 | 내용 |
|------|------|
| `linear_algebra_practice_basics.py` | 6파트 종합 실습 (행렬곱, 역행렬, 고유값, 행렬식, 회전행렬, 공분산) |
| `linear_algebra_practice_quiz.py` | 10문제 주관식 퀴즈 (문제/답안 분리) |

### 실행 방법

```bash
cd "Studies/Phase 1/week2"
python3 linear_algebra_practice_basics.py
python3 linear_algebra_practice_quiz.py
```

---

## 🎬 추천 자료

### 영상

| 영상 | 설명 |
|------|------|
| [3Blue1Brown - Matrix multiplication](https://youtu.be/XkY2DOUCWMU) | 행렬 곱셈 시각화 (복습) |
| [3Blue1Brown - Inverse matrices](https://youtu.be/uQhTuRlWMxw) | 역행렬의 의미 |
| [3Blue1Brown - Eigenvectors](https://youtu.be/PFDu9oVAE-g) | 고유벡터 직관 |

### 실습 도구

| 도구 | 용도 |
|------|------|
| [Desmos Matrix Calculator](https://www.desmos.com/matrix) | 온라인 행렬 계산기 |
| [GeoGebra](https://www.geogebra.org/classic) | 선형 변환 시각화 |

---

## ✅ 학습 완료 체크리스트

- [ ] NumPy로 행렬 곱셈, 역행렬, 고유값을 계산할 수 있다
- [ ] 행렬 곱셈이 "선형 변환의 합성"임을 설명할 수 있다
- [ ] 고유벡터가 "방향이 변하지 않는 벡터"임을 이해했다
- [ ] 2D 회전 행렬을 직접 유도할 수 있다
- [ ] 회전 행렬이 직교 행렬인 이유를 기하학적으로 설명할 수 있다
- [ ] det(R) = 1인 이유를 "크기 보존 + 방향 유지"로 설명할 수 있다
- [ ] 공분산 행렬의 대각 원소가 각 축의 분산임을 안다
- [ ] 공분산 행렬의 고유값 분해가 불확실성 타원과 연결됨을 이해했다

---

## 🔗 다음 단계

Week 2 완료 후 → **Week 3: SVD 집중**으로 이동
- SVD = 회전-스케일-회전 분해
- SLAM의 핵심 도구 (Essential Matrix, Homography 분해 등)
