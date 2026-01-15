# Week 5: 강체 변환 (Rigid Body Transformation)

## 📌 개요

**강체 변환(Rigid Body Transformation)** 은 물체의 형태를 변형시키지 않고 위치와 방향만 바꾸는 변환입니다. 3D 공간에서 **회전(Rotation) + 평행이동(Translation)** 을 결합한 것으로, SLAM에서 카메라나 로봇의 **포즈(Pose)** 를 표현하는 핵심 개념입니다.

이번 주에는 **SE(3) 변환 행렬**, **동차 좌표**, 그리고 **ROS TF2**와의 연결을 학습합니다.

## 🎯 학습 목표

1. SE(3) 변환 행렬의 구조 이해
2. 동차 좌표(Homogeneous Coordinates)의 장점 이해
3. 변환 행렬 합성과 역변환 계산
4. ROS TF2와 변환 행렬의 관계 이해
5. SLAM에서 키프레임 간 상대 포즈 표현

## 📚 사전 지식

- Week 4: 회전 표현 (회전 행렬, 쿼터니언)
- 행렬 곱셈

## ⏱️ 예상 학습 시간

| 항목 | 시간 |
|------|------|
| 이론 학습 | 2시간 |
| 실습 예제 | 2시간 |
| ROS TF 연결 | 1-2시간 |
| **총 소요시간** | **5-6시간** |

---

## 📖 핵심 개념

### 1. SE(3) 변환 행렬

#### 구조

4×4 동차 변환 행렬:

```
T = [ R  | t ]  ∈ SE(3)
    [----+---]
    [ 0  | 1 ]

여기서:
- R: 3×3 회전 행렬 (SO(3))
- t: 3×1 평행이동 벡터
- SE(3): Special Euclidean Group (3D)
```

```python
def make_transform(R, t):
    """회전 행렬과 평행이동으로 SE(3) 생성"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
```

#### SE(3)의 자유도

| 요소 | 자유도 | 설명 |
|------|--------|------|
| 회전 R | 3 | roll, pitch, yaw |
| 평행이동 t | 3 | x, y, z |
| **총합** | **6** | 6 DOF (Degrees of Freedom) |

---

### 2. 동차 좌표 (Homogeneous Coordinates)

#### 동차 좌표란?

**동차 좌표**는 n차원 공간의 점을 **(n+1)개의 수**로 표현하는 좌표 시스템입니다.

```
n차원 유클리드 공간 → (n+1)차원 벡터로 표현
```

- **2D 점** `(x, y)` → 동차 좌표 `(x, y, w)`
- **3D 점** `(x, y, z)` → 동차 좌표 `(x, y, z, w)`

#### 핵심 특징: 스케일 불변성

동차 좌표의 가장 중요한 특징은 **0이 아닌 같은 배수로 곱한 좌표들이 모두 같은 점**을 나타낸다는 것입니다.

```
(x, y, w) ≡ (kx, ky, kw)  (k ≠ 0인 임의의 실수)
```

**예시**: 다음은 모두 2D 점 `(2, 3)`을 나타냅니다
```python
(2, 3, 1)        # 표준 형태 (w=1)
(4, 6, 2)        # 2배
(6, 9, 3)        # 3배
(10, 15, 5)      # 5배
(-2, -3, -1)     # -1배

# 모두 비율이 2:3:1로 같음!
```

#### 좌표 변환 방법

**동차 좌표 → 일반 좌표** (정규화)
```python
# 마지막 좌표 w로 나누기
(x_h, y_h, w) → (x_h/w, y_h/w)
(x_h, y_h, z_h, w) → (x_h/w, y_h/w, z_h/w)

# 예시
(6, 9, 3) → (6/3, 9/3) = (2, 3)
(2, 4, 6, 2) → (2/2, 4/2, 6/2) = (1, 2, 3)
```

**일반 좌표 → 동차 좌표**
```python
# 마지막에 1을 추가 (표준 형태)
(x, y) → (x, y, 1)
(x, y, z) → (x, y, z, 1)

# 일반 좌표 → 동차 좌표
p = np.array([x, y, z])
p_h = np.array([x, y, z, 1])  # w=1 추가

# 동차 좌표 → 일반 좌표
p = p_h[:3] / p_h[3]  # w로 나누기
```

#### 기하학적 직관: 원점에서의 광선

동차 좌표 `(x, y, w)`는 **원점에서 출발하는 광선**으로 생각할 수 있습니다.

```
       (4,6,2)
         ↗
(2,3,1) ↗
      ↗
 원점 •―――――→ 이 광선 위의 모든 점
           (6,9,3)
           (8,12,4)
           ...
```

같은 광선 위의 모든 점들이 같은 동차 좌표를 나타내며, 이 광선이 `w=1` 평면과 만나는 점이 **실제 일반 좌표**입니다.

#### 왜 동차 좌표를 사용하는가?

**문제점 (일반 좌표)**:
```python
# 회전과 이동을 동시에 하려면 두 가지 연산 필요
p' = R @ p + t  # 행렬 곱셈 + 벡터 덧셈
```

**해결책 (동차 좌표)**:
```python
# 모든 변환을 하나의 행렬 곱셈으로!
P' = T @ P  # 행렬 곱셈만!
```

**작동 원리**:
```
      [r11 r12 r13 tx]   [x]   [r11·x + r12·y + r13·z + tx·1]
P' =  [r21 r22 r23 ty] @ [y] = [r21·x + r22·y + r23·z + ty·1]
      [r31 r32 r33 tz]   [z]   [r31·x + r32·y + r33·z + tz·1]
      [ 0   0   0  1 ]   [1]   [          0  +  0  +  1         ]
      
      [R @ p + t]
    = [    ...    ]  ← 회전 + 이동이 한 번에!
      [    ...    ]
      [     1     ]
```

마지막 원소가 `1`이기 때문에 `tx·1 = tx`가 되어 이동 항이 자동으로 더해집니다!

**장점**:
1. **연산 통일**: 회전과 평행이동을 행렬 곱셈 하나로 처리
2. **변환 합성**: 여러 변환을 쉽게 합성
   ```python
   T_total = T3 @ T2 @ T1  # 행렬끼리만 곱하면 됨!
   P_final = T_total @ P
   ```
3. **투영 변환**: 카메라 투영도 행렬 곱셈으로 표현 가능
4. **무한원점 표현**: `w=0`으로 무한히 먼 점(방향 벡터) 표현 가능

---

### 3. 변환 연산

#### 변환 합성

```python
# 연속 변환: 먼저 T1, 그 다음 T2 적용
T_combined = T2 @ T1
```

> [!WARNING]
> 변환 순서 주의! T2 @ T1 ≠ T1 @ T2

#### 역변환

```
T⁻¹ = [ R^T | -R^T @ t ]
      [-----+---------]
      [  0  |    1     ]
```

```python
def inverse_transform(T):
    """SE(3) 역변환"""
    R = T[:3, :3]
    t = T[:3, 3]
    
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv
```

---

### 4. 좌표계 변환

#### 월드 ↔ 카메라 변환

```
         월드 좌표계
              │
              │ T_cw (world → camera)
              ▼
         카메라 좌표계
```

```python
# 월드 좌표의 점 → 카메라 좌표
p_camera = T_cw @ p_world

# 카메라 좌표의 점 → 월드 좌표
p_world = T_wc @ p_camera  # T_wc = T_cw^(-1)
```

#### 상대 포즈

키프레임 i와 j 사이의 상대 변환:

```
T_ij = T_j @ T_i⁻¹

해석: i 기준으로 j가 어디 있는가
```

---

## 🤖 SLAM에서의 활용

### 카메라 포즈 표현

```
T_wc = [ R_wc | t_wc ]
       [-----+-----]
       [  0  |  1  ]

- R_wc: 카메라 방향 (월드 기준)
- t_wc: 카메라 위치 (월드 기준)
```

### 3D 점 투영

```python
def project_point(P_world, T_cw, K):
    """3D 월드 점 → 2D 이미지 좌표"""
    # 월드 → 카메라 변환
    P_cam = T_cw @ np.append(P_world, 1)
    P_cam = P_cam[:3]
    
    # 정규화 좌표
    x_norm = P_cam[0] / P_cam[2]
    y_norm = P_cam[1] / P_cam[2]
    
    # 픽셀 좌표
    p = K @ np.array([x_norm, y_norm, 1])
    return p[:2]
```

### ROS TF2 연결

```python
# geometry_msgs/Transform 구조
# - translation: Vector3 (x, y, z)
# - rotation: Quaternion (x, y, z, w)

# SE(3) → TF2 메시지
def se3_to_transform(T):
    from geometry_msgs.msg import Transform, Vector3, Quaternion
    
    t = Transform()
    t.translation = Vector3(x=T[0,3], y=T[1,3], z=T[2,3])
    
    q = rotation_matrix_to_quaternion(T[:3,:3])
    t.rotation = Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])
    
    return t
```

---

## 📊 요약 표

| 개념 | 크기 | 자유도 | 용도 |
|------|------|--------|------|
| 회전 행렬 R | 3×3 | 3 | 방향 표현 |
| 쿼터니언 q | 4×1 | 3 | 방향 표현 |
| 평행이동 t | 3×1 | 3 | 위치 표현 |
| SE(3) 변환 T | 4×4 | 6 | 포즈 표현 |

---

## 💻 실습 파일

| 파일 | 내용 |
|------|------|
| `se3_basics.py` | SE(3) 변환 행렬 생성 및 연산 |
| `se3_quiz.py` | 주관식 퀴즈 |

### 실행 방법

```bash
cd "Studies/Phase 1/week5"
python3 se3_basics.py
python3 se3_quiz.py
```

---

## 🎬 추천 영상

| 영상 | 설명 |
|------|------|
| [Cyrill Stachniss - SE(3)](https://www.youtube.com/watch?v=khGGoAAl1c4) | SE(3) 이론 |
| [ROS TF Tutorial](http://wiki.ros.org/tf2/Tutorials) | TF2 사용법 |

---

## ✅ 학습 완료 체크리스트

- [ ] 4×4 변환 행렬의 구조를 그릴 수 있다
- [ ] 동차 좌표의 장점을 설명할 수 있다
- [ ] 두 변환의 합성을 계산할 수 있다
- [ ] 역변환 공식을 이해했다
- [ ] ROS TF2 메시지와 SE(3)의 관계를 안다
- [ ] 상대 포즈 T_ij = T_j @ T_i⁻¹ 의미를 이해했다

---

## 🔗 다음 단계

Week 5 완료 후 → **Week 6: Lie 군/대수 기초**로 이동
- 왜 over-parameterized인지
- SE(3)의 접선 공간
- 최적화에서의 활용
