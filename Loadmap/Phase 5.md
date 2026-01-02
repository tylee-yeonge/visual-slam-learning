# Phase 5: VINS-Fusion 코드 분석

> ⏰ **기간**: 2개월  
> 🎯 **목표**: VINS-Fusion 코드를 읽고 수정할 수 있는 수준  
> ⏱️ **주간 시간**: 약 7시간  
> 📍 **전제**: Phase 4의 개념을 이해한 상태

---

## 📖 코드 읽기 전 준비

### VINS-Fusion 기본 정보
- **원본**: ROS1 기반 (ROS Melodic/Noetic)
- **언어**: C++ (일부 Python 스크립트)
- **최적화**: Ceres Solver 사용
- **저장소**: `HKUST-Aerial-Robotics/VINS-Fusion`

### 추천 코드 읽기 순서
```
1. 전체 폴더 구조 파악
2. Config 파일 분석 (파라미터 이해)
3. Feature Tracker (가장 직관적)
4. Estimator 데이터 흐름
5. Pre-integration (Phase 4 복습)
6. Optimization (BA + IMU factor)
7. Marginalization (가장 어려움)
```

### 준비물
- [ ] VINS-Fusion 저장소 클론
- [ ] IDE 설정 (VSCode + C++ extensions 또는 CLion)
- [ ] EuRoC 데이터셋 (테스트용)
- [ ] VINS-Mono 논문 (코드와 대조용)

---

## 📋 Section 5.1: 전체 구조 파악 (2주)

### Week 1: 코드 구조

#### 저장소 구조 파악
```
VINS-Fusion/
├── camera_models/        # 카메라 모델, 왜곡 보정
├── config/              # 설정 파일들 (EuRoC, Realsense 등)
├── support_files/       # 단어장 (loop closure용)
├── vins_estimator/      # ⭐ 핵심 패키지
│   ├── src/
│   │   ├── estimator/   # 상태 추정기
│   │   ├── factor/      # Ceres factor들
│   │   ├── initial/     # 초기화
│   │   └── utility/     # 유틸리티
│   └── CMakeLists.txt
├── feature_tracker/     # ⭐ 특징점 추적
├── pose_graph/          # Loop closure (선택)
└── global_fusion/       # GPS 융합 (선택)
```

- [ ] 각 폴더 역할 파악
- [ ] CMakeLists.txt로 의존성 확인
- [ ] 빌드 순서 이해

#### 주요 패키지 역할

| 패키지 | 역할 | Phase 연결 |
|--------|------|-----------|
| `camera_models` | 카메라 모델, 왜곡 보정 | Phase 2.1 |
| `feature_tracker` | 특징점 검출/추적 | Phase 2.2, 2.4 |
| `vins_estimator` | VIO 핵심 (최적화) | Phase 3, 4 전체 |
| `pose_graph` | Loop closure | (범위 외) |

### Week 2: 데이터 흐름

#### 노드 간 통신 (ROS 토픽)
```
[Camera] → /cam0/image_raw
                ↓
        [feature_tracker_node]
                ↓
           /feature_tracker/feature
                ↓
        [estimator_node] ← /imu0 [IMU]
                ↓
           /vins_estimator/odometry
           /vins_estimator/path
```

- [ ] `rqt_graph`로 실제 토픽 확인
- [ ] 각 토픽의 메시지 타입 확인
- [ ] 데이터 흐름도 직접 그려보기

#### 메인 진입점

| 노드 | 파일 | 역할 |
|------|------|------|
| feature_tracker_node | `feature_tracker_node.cpp` | 이미지 → 특징점 |
| estimator_node | `estimator_node.cpp` | 특징점 + IMU → 포즈 |

- [ ] 각 노드의 `main()` 함수 찾기
- [ ] 콜백 함수 목록 파악
- [ ] 스레드 구조 파악 (measurement thread)

### 🔍 Section 5.1 자체 점검
1. Feature tracker와 estimator는 어떤 토픽으로 통신하는가?
2. IMU 데이터는 어느 노드에서 처리되는가?
3. 설정 파일(config)은 어디서 로드되는가?

---

## 📋 Section 5.2: Feature Tracker 분석 (2주)

> 💡 가장 직관적인 모듈. 여기서 코드 읽기 연습!

### Week 3: 특징점 추적

#### feature_tracker.cpp 분석

**클래스 구조**
```cpp
class FeatureTracker {
    cv::Mat prev_img, cur_img;      // 이전/현재 이미지
    vector<cv::Point2f> prev_pts, cur_pts;  // 특징점
    vector<int> ids;                 // 특징점 ID
    // ...
};
```
- [ ] 멤버 변수 파악
- [ ] Phase 2.4 (KLT) 개념과 연결

**핵심 함수: `readImage()`**
```cpp
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
```
- [ ] 이미지 전처리 (히스토그램 평활화)
- [ ] `cv::calcOpticalFlowPyrLK()` — KLT 추적
- [ ] `cv::goodFeaturesToTrack()` — 새 특징점 검출
- [ ] 마스킹으로 균등 분포

#### 실습: 로깅 추가
```cpp
// readImage() 함수 내에 추가
ROS_INFO("Tracked: %d, New: %d, Total: %d", 
         tracked_cnt, new_cnt, cur_pts.size());
```
- [ ] 추적 성공률 로깅
- [ ] 특징점 수 변화 관찰

### Week 4: 특징점 관리

#### 특징점 ID 관리
```cpp
// 새 특징점에 ID 할당
for (auto &p : new_pts) {
    cur_pts.push_back(p);
    ids.push_back(n_id++);  // 전역 카운터
}
```
- [ ] `n_id`: 전역 특징점 카운터
- [ ] 한번 할당된 ID는 추적되는 동안 유지
- [ ] 추적 실패 시 해당 ID 제거

#### 발행 데이터 구조
```cpp
sensor_msgs::PointCloud feature_points;
// - 정규화 좌표 (x, y)
// - 특징점 ID
// - 추적 횟수
// - 속도 (optical flow)
```
- [ ] 정규화 좌표 계산 (왜곡 보정 + K^-1)
- [ ] 속도 계산 (프레임 간 픽셀 이동)

#### Phase 2 개념 연결
| 코드 | Phase 2 개념 |
|------|-------------|
| `goodFeaturesToTrack` | FAST 코너 (Section 2.2) |
| `calcOpticalFlowPyrLK` | KLT Tracker (Section 2.4) |
| `undistortedPoints` | 카메라 캘리브레이션 (Section 2.1) |

### 🔍 Section 5.2 자체 점검
1. `readImage()`에서 KLT 추적이 실패한 점은 어떻게 처리되는가?
2. 특징점의 정규화 좌표는 어떻게 계산되는가?
3. `MAX_CNT` 파라미터는 어디서 어떻게 사용되는가?

---

## 📋 Section 5.3: Estimator 분석 - 데이터 처리 (2주)

### Week 5: 데이터 수신

#### estimator_node.cpp 구조
```cpp
int main() {
    // 1. 파라미터 로드
    readParameters(config_file);
    
    // 2. Estimator 생성
    Estimator estimator;
    
    // 3. 콜백 등록
    ros::Subscriber sub_imu = nh.subscribe(IMU_TOPIC, ...);
    ros::Subscriber sub_feature = nh.subscribe(FEATURE_TOPIC, ...);
    
    // 4. 처리 스레드 시작
    std::thread measurement_process{process};
    
    ros::spin();
}
```

#### IMU 콜백
```cpp
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg) {
    // IMU 데이터를 버퍼에 저장
    imu_buf.push(imu_msg);
}
```
- [ ] 버퍼 구조 파악
- [ ] 타임스탬프 관리

#### 핵심 함수: `getMeasurements()`
```cpp
// IMU와 이미지를 시간 정렬
std::vector<std::pair<
    std::vector<ImuConstPtr>,  // IMU 데이터들
    ImgConstPtr                 // 하나의 이미지
>> getMeasurements()
```
- [ ] 이미지 하나당 그 사이의 IMU 데이터 묶음
- [ ] 시간 동기화 로직 분석

### Week 6: Estimator 클래스 구조

#### 상태 변수 (estimator.h)
```cpp
class Estimator {
    // Sliding window 상태 (WINDOW_SIZE + 1개)
    Vector3d Ps[(WINDOW_SIZE + 1)];   // 위치
    Vector3d Vs[(WINDOW_SIZE + 1)];   // 속도
    Matrix3d Rs[(WINDOW_SIZE + 1)];   // 회전
    Vector3d Bas[(WINDOW_SIZE + 1)];  // 가속도 바이어스
    Vector3d Bgs[(WINDOW_SIZE + 1)];  // 자이로 바이어스
    
    // Pre-integration
    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    
    // 특징점 관리
    FeatureManager f_manager;
};
```
- [ ] Phase 4의 상태 벡터와 매칭
- [ ] Sliding window 인덱스 이해

#### 초기화 과정
```
processImage()
  ├─ (not initialized) → initialStructure()
  │                        ├─ relativePose()     // Essential Matrix
  │                        ├─ sfm()              // Structure from Motion
  │                        └─ visualInitialAlign() // VIO 정렬
  └─ (initialized) → optimization()
```
- [ ] `initial/initial_sfm.cpp` 분석
- [ ] `initial/initial_alignment.cpp` 분석
- [ ] Phase 4.4 (VIO 초기화) 개념과 연결

### 🔍 Section 5.3 자체 점검
1. `getMeasurements()`는 왜 필요한가?
2. Sliding window의 크기는 어디서 정의되는가?
3. 초기화가 완료되었는지 어떻게 판단하는가?

---

## 📋 Section 5.4: Estimator 분석 - 최적화 (4주)

> ⭐ Phase 4에서 배운 개념들이 구현된 핵심 부분

### Week 7: Pre-integration 코드

#### integration_base.h 분석
```cpp
class IntegrationBase {
    // Pre-integrated measurements
    Eigen::Vector3d delta_p;    // Δp_ij
    Eigen::Quaterniond delta_q; // Δq_ij
    Eigen::Vector3d delta_v;    // Δv_ij
    
    // 공분산
    Eigen::Matrix<double, 15, 15> covariance;
    
    // 바이어스 보정용 자코비안
    Eigen::Matrix<double, 15, 15> jacobian;
    
    void propagate(double dt, const Vector3d &acc, const Vector3d &gyr);
    void repropagate(const Vector3d &_linearized_ba, const Vector3d &_linearized_bg);
};
```

**핵심 함수: `propagate()`**
- [ ] IMU 측정값으로 delta_p, delta_v, delta_q 업데이트
- [ ] 공분산 전파
- [ ] Phase 4.3 수식과 대조

**핵심 함수: `repropagate()`**
- [ ] 바이어스 변경 시 재계산
- [ ] 자코비안으로 1차 보정 (또는 full repropagate)

#### imu_factor.h 분석
```cpp
class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9> {
    // 15: 잔차 차원 (p, q, v, ba, bg)
    // 7: 포즈 (p + q)
    // 9: 속도 + 바이어스
    
    virtual bool Evaluate(double const *const *parameters,
                         double *residuals,
                         double **jacobians) const;
};
```
- [ ] `Evaluate()`: 잔차 계산 (Phase 4.3 Factor 오차)
- [ ] 자코비안 계산 (최적화용)
- [ ] Ceres cost function 인터페이스

#### Phase 4 개념 연결
| 코드 | Phase 4 개념 |
|------|-------------|
| `delta_p, delta_v, delta_q` | Pre-integrated measurement |
| `propagate()` | IMU 적분 |
| `IMUFactor::Evaluate()` | IMU Factor 오차 |

### Week 8: Visual Factor 코드

#### projection_factor.h 분석
```cpp
class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1> {
    // 2: 잔차 차원 (u, v 재투영 오차)
    // 7: 카메라 i 포즈
    // 7: 카메라 j 포즈
    // 7: IMU-Camera extrinsic
    // 1: 역깊이 (inverse depth)
};
```

**역깊이 파라미터화**
- [ ] 3D 점을 (x, y, 1/d)로 표현
- [ ] 첫 관측 카메라 기준
- [ ] 수치적 안정성 (먼 점도 OK)

**재투영 오차 계산**
```cpp
// pseudo code
pts_camera_j = R_j^T * (R_i * pts_camera_i / inv_depth + P_i - P_j)
residual = pts_2d_j - project(pts_camera_j)
```
- [ ] Phase 3.4 (BA) 개념과 연결

### Week 9: Marginalization

> ⚠️ 가장 어려운 부분. 개념적 이해에 집중!

#### Marginalization이란?
```
문제: Sliding window가 이동하면 오래된 프레임 제거
질문: 그 프레임의 정보를 그냥 버려도 되나?
답: Prior로 변환하여 보존!
```

#### marginalization_factor.cpp 핵심
- [ ] 오래된 프레임에 연결된 factor 수집
- [ ] Schur complement로 해당 변수 소거
- [ ] 남은 변수에 대한 prior (정보 행렬) 생성
- [ ] 다음 최적화에서 prior factor로 사용

#### Sliding Window 관리
```cpp
// 두 가지 marginalization 전략
if (marginalization_flag == MARGIN_OLD) {
    // 가장 오래된 프레임 제거
} else {
    // 두 번째 최신 프레임 제거 (키프레임 아닐 때)
}
```
- [ ] Phase 3.3 (키프레임) 개념과 연결
- [ ] 어떤 프레임을 제거할지 결정 로직

### Week 10: 최적화 실행

#### optimization() 함수 흐름
```cpp
void Estimator::optimization() {
    // 1. Ceres Problem 생성
    ceres::Problem problem;
    
    // 2. Parameter blocks 추가
    for (int i = 0; i <= WINDOW_SIZE; i++) {
        problem.AddParameterBlock(para_Pose[i], 7, ...);
        problem.AddParameterBlock(para_SpeedBias[i], 9);
    }
    
    // 3. Residual blocks 추가
    // - Marginalization prior
    // - IMU factors
    // - Visual factors
    
    // 4. Solve
    ceres::Solve(options, &problem, &summary);
    
    // 5. 결과 적용
    double2vector();
}
```

- [ ] `vector2double()`: 상태 → Ceres 파라미터
- [ ] `double2vector()`: Ceres 결과 → 상태
- [ ] Solver 옵션 분석 (max iterations, tolerance)

#### 실습: 최적화 과정 로깅
```cpp
// optimization() 내에 추가
ROS_INFO("Optimization: iter=%d, cost=%.4f → %.4f",
         summary.iterations.size(),
         summary.initial_cost,
         summary.final_cost);
```
- [ ] 수렴 과정 관찰
- [ ] 비용 함수 감소 확인

### 🔍 Section 5.4 자체 점검
1. `IntegrationBase`에서 `delta_p, delta_v, delta_q`는 무엇을 의미하는가?
2. `ProjectionFactor`에서 역깊이를 사용하는 이유는?
3. Marginalization이 필요한 이유는?

---

## 📋 Section 5.5: 파라미터 실험 (2주)

### Week 11: 주요 파라미터 이해

#### Config 파일 구조 (예: `euroc_stereo_config.yaml`)
```yaml
# 카메라
image_width: 752
image_height: 480
model_type: PINHOLE

# IMU 노이즈 (⭐ 중요)
acc_n: 0.1          # 가속도계 white noise
gyr_n: 0.01         # 자이로 white noise  
acc_w: 0.001        # 가속도계 random walk
gyr_w: 0.0001       # 자이로 random walk

# 시스템
window_size: 10     # Sliding window 크기
max_cnt: 150        # 최대 특징점 수
min_dist: 30        # 특징점 최소 간격
min_parallax: 10.0  # 키프레임 선택 기준
```

#### 파라미터 역할

| 파라미터 | 역할 | 영향 |
|----------|------|------|
| `acc_n`, `gyr_n` | IMU 노이즈 | 클수록 IMU 신뢰↓, Vision 신뢰↑ |
| `window_size` | Sliding window | 클수록 정확↑, 계산량↑ |
| `max_cnt` | 특징점 수 | 클수록 정확↑, 계산량↑ |
| `min_parallax` | 키프레임 기준 | 클수록 키프레임 적게 선택 |

### Week 12: 파라미터 튜닝 실험

#### 실험 설정
- [ ] 데이터셋: EuRoC MH_01_easy
- [ ] 기준: 기본 파라미터로 ATE 측정

#### 실험 1: IMU 노이즈 파라미터
```yaml
# 실험 A: IMU 신뢰 높임
acc_n: 0.05  # 기본의 절반
gyr_n: 0.005

# 실험 B: IMU 신뢰 낮춤  
acc_n: 0.2   # 기본의 2배
gyr_n: 0.02
```
- [ ] 각각 실행 후 ATE 비교
- [ ] 어떤 환경에서 어떤 설정이 좋은지 분석

#### 실험 2: Sliding Window 크기
```yaml
window_size: 5   # 작게
window_size: 15  # 크게
```
- [ ] ATE, 계산 시간 비교
- [ ] Trade-off 분석

#### 실험 3: 특징점 수
```yaml
max_cnt: 80   # 적게
max_cnt: 200  # 많이
```
- [ ] 정확도, 처리 속도 비교

#### 결과 정리
| 실험 | ATE (m) | 처리시간 (ms) | 비고 |
|------|---------|--------------|------|
| 기본 | | | |
| IMU 신뢰↑ | | | |
| IMU 신뢰↓ | | | |
| Window 작게 | | | |
| Window 크게 | | | |

### 🔍 Section 5.5 자체 점검
1. IMU 노이즈 파라미터를 실제보다 크게 설정하면 어떤 현상이 나타나는가?
2. `window_size`가 작으면 왜 정확도가 떨어지는가?
3. 특징점이 너무 적으면 어떤 문제가 생기는가?

---

## ✅ Phase 5 완료 체크리스트

### 전체 구조
- [ ] VINS-Fusion 폴더 구조 이해
- [ ] 노드 간 데이터 흐름 파악
- [ ] 데이터 흐름도 직접 그림

### Feature Tracker
- [ ] 특징점 검출/추적 과정 이해
- [ ] `readImage()` 함수 분석 완료
- [ ] 로깅 추가 실습

### Estimator
- [ ] Pre-integration 코드 분석 (`integration_base.h`)
- [ ] IMU Factor 구조 이해 (`imu_factor.h`)
- [ ] Visual Factor 구조 이해 (`projection_factor.h`)
- [ ] Marginalization 개념적 이해
- [ ] `optimization()` 흐름 파악

### 파라미터
- [ ] 주요 파라미터 역할 이해
- [ ] 파라미터 변경 실험 완료
- [ ] 결과 정리 및 분석

---

## 🎯 Phase 5 완료 기준

> "VINS-Fusion 코드에서 원하는 부분을 찾아가고, 파라미터를 바꾸거나 로깅을 추가하여 동작을 분석할 수 있다"

---

## 📚 참고 자료

### 코드
- VINS-Fusion: `github.com/HKUST-Aerial-Robotics/VINS-Fusion`
- VINS-Mono (원본): `github.com/HKUST-Aerial-Robotics/VINS-Mono`

### 코드 분석 자료
- VINS 코드 분석 블로그들 (검색)
- SLAM KR 커뮤니티 자료
- GitHub Issues/Discussions

### 논문 (코드와 함께)
- VINS-Mono 논문: 코드 각 부분과 대조하며 읽기

---

## 💡 팁

1. **IDE 적극 활용**: "Go to Definition", "Find References"
2. **디버거 사용**: Breakpoint + 변수 watch
3. **로그 추가**: 의심되는 곳에 `ROS_INFO` 추가
4. **작은 단위로**: 한 함수씩 완전히 이해하고 넘어가기
5. **논문과 대조**: 코드를 논문 수식/그림과 매칭
6. **그림 그리기**: 클래스 관계, 데이터 흐름 시각화
7. **실험하기**: 파라미터 바꿔보고 결과 관찰

---

## ❓ 다음 단계

Phase 5 완료 후:
- Phase 6 (AMR 실적용)로 진행
- 회사 AMR에 실제 적용 시작
- ROS2 통합, 휠 오도메트리 융합