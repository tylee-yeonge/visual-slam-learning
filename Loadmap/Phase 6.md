# Phase 6: 회사 AMR 실적용

> ⏰ **기간**: 2-3개월 (실제 적용은 예상보다 오래 걸릴 수 있음)  
> 🎯 **목표**: 실제 AMR에 VIO 적용, 휠 오도메트리 융합, 실패 대응까지  
> ⏱️ **주간 시간**: 약 7시간  
> 📍 **전제**: Phase 5까지 VINS-Fusion 이해 완료

---

## ⚠️ 실적용 전 주의사항

- 실제 환경은 데이터셋과 **매우 다름**
- 예상치 못한 문제가 많이 발생함 → 여유 시간 확보
- **안전 최우선**: 테스트 시 로봇 속도 제한, 비상 정지 준비
- 기존 시스템(휠 오도) 백업 항상 유지

---

## 📋 Section 6.1: 하드웨어 준비 (2주)

### Week 1: 카메라 선정 및 마운트

#### 카메라 선정

| 옵션 | 장점 | 단점 |
|------|------|------|
| **Realsense D435i** (추천) | IMU 내장, 스테레오 | 가격 |
| Realsense T265 | VIO 최적화 | 단종 |
| USB 카메라 + 외부 IMU | 저렴 | 동기화 어려움 |

- [ ] 회사에 사용 가능한 카메라 확인
- [ ] 해상도, 화각, 프레임레이트 확인
- [ ] IMU 내장 여부 확인 (D435i는 BMI055)

#### 마운트 설계
- [ ] AMR 프레임에 부착 위치 결정
  - 전방 시야 확보
  - 바닥이 아닌 **전방/측면** 향하도록
  - 높이: 지면에서 30cm 이상 권장
- [ ] **진동 최소화** 고려 (댐퍼, 고무 마운트)
- [ ] 마운트 제작 또는 3D 프린팅

#### 설치
- [ ] 카메라 견고하게 고정 (흔들리면 캘리브레이션 무효)
- [ ] 케이블 배선 (움직이는 부분 피하기)
- [ ] USB 연결 테스트

### Week 2: 캘리브레이션

#### Camera Intrinsic 캘리브레이션
- [ ] 체스보드 패턴 준비 (A3 이상 권장)
- [ ] 다양한 거리/각도에서 이미지 수집 (15-25장)
- [ ] Kalibr 또는 OpenCV로 캘리브레이션
- [ ] **검증**: 재투영 오차 < 0.5 pixel

#### Camera-IMU Extrinsic 캘리브레이션
- [ ] AprilGrid 타겟 준비
- [ ] **다양한 움직임** 필수 (모든 축 회전 + 평행이동)
- [ ] 60-120초 데이터 수집
- [ ] Kalibr `kalibr_calibrate_imu_camera` 실행
- [ ] **검증**: Reprojection error, IMU error 확인

#### Camera-Robot Base 변환 측정
```
base_link → camera_link 변환 필요
```
- [ ] 수동 측정 또는 CAD 기반
- [ ] 회전 (카메라 방향)
- [ ] 평행이동 (카메라 위치)
- [ ] TF static publisher 준비

### 🔍 Section 6.1 자체 점검
1. 카메라 마운트가 진동에 견고한가?
2. Camera-IMU 캘리브레이션 결과가 합리적인가?
3. TF tree가 올바르게 구성되었는가?

---

## 📋 Section 6.2: ROS2 통합 (3주)

### Week 3: ROS 환경 결정

#### Option A: ROS1 + ros1_bridge (권장 - 빠른 시작)
```
[ROS1 VINS-Fusion] ←bridge→ [ROS2 Nav2/AMR]
```
- [ ] ROS1 환경에서 VINS-Fusion 실행
- [ ] ros1_bridge로 토픽 변환
- [ ] 장점: 안정적, 빠른 적용
- [ ] 단점: 두 ROS 버전 관리

#### Option B: VINS-Fusion ROS2 포팅
- [ ] 커뮤니티 ROS2 포팅 버전 검색
- [ ] 직접 포팅 (시간 많이 소요)
- [ ] 장점: 깔끔한 구조
- [ ] 단점: 안정성 검증 필요

> 💡 Phase 6에서는 **Option A**로 빠르게 시작하고, 나중에 필요시 포팅 권장

#### 카메라/IMU 드라이버
- [ ] `realsense-ros` 설치 (ROS1 또는 ROS2)
- [ ] 카메라 노드 실행 확인
- [ ] 토픽 확인:
  - `/camera/color/image_raw`
  - `/camera/imu` (D435i)
- [ ] IMU 주파수 확인 (200Hz 이상 권장)

### Week 4: VINS Config 및 기본 테스트

#### Config 파일 작성
```yaml
# amr_config.yaml 예시

# 카메라 파라미터 (캘리브레이션 결과)
model_type: PINHOLE
camera_name: camera
image_width: 640
image_height: 480
distortion_parameters:
  k1: ...
  k2: ...
projection_parameters:
  fx: ...
  fy: ...
  cx: ...
  cy: ...

# IMU 파라미터 (데이터시트 또는 Allan Variance)
acc_n: 0.1          # 가속도계 noise density
gyr_n: 0.01         # 자이로 noise density
acc_w: 0.001        # 가속도계 random walk
gyr_w: 0.0001       # 자이로 random walk

# Extrinsic (Kalibr 결과)
body_T_cam0: !!opencv-matrix
  rows: 4
  cols: 4
  data: [...]

# 시스템 설정
max_cnt: 150
min_dist: 30
```

#### 정적 테스트
- [ ] 로봇 정지 상태에서 VINS 실행
- [ ] **기대**: 포즈가 거의 고정
- [ ] Drift 발생 시 → IMU 파라미터 확인

#### 동적 테스트 (수동 조작)
- [ ] 저속으로 로봇 이동 (0.1-0.2 m/s)
- [ ] RViz에서 궤적 시각화
- [ ] 원위치 복귀 테스트

### Week 5: TF Tree 및 Nav2 연동

#### TF Tree 설계
```
map
 └── odom (VIO 또는 융합 결과)
      └── base_link
           ├── camera_link
           ├── imu_link
           └── lidar_link (있다면)
```

- [ ] VINS 출력을 `odom → base_link` 변환으로 발행
- [ ] Static TF: `base_link → camera_link`
- [ ] `tf2_ros` 사용

#### VINS → Odometry 변환
```cpp
// VINS 출력 (camera frame) → base_link frame 변환 필요
nav_msgs::Odometry odom_msg;
odom_msg.header.frame_id = "odom";
odom_msg.child_frame_id = "base_link";
// ... 좌표 변환 적용
```

#### Nav2 연동 준비
- [ ] `robot_localization` 또는 직접 융합
- [ ] Odometry 토픽 설정
- [ ] Costmap에 반영 확인

### 🔍 Section 6.2 자체 점검
1. ros1_bridge가 올바르게 토픽을 변환하는가?
2. VINS 출력 좌표계와 AMR base_link 좌표계가 일치하는가?
3. TF tree에 순환이나 끊김이 없는가?

---

## 📋 Section 6.3: 휠 오도메트리 융합 (3주)

### Week 6: 융합 전략 수립

#### 센서 특성 비교

| 특성 | VIO | 휠 오도메트리 |
|------|-----|-------------|
| **장점** | 슬립 영향 없음, 절대 스케일 | 빠름, 안정적, 계산량 적음 |
| **단점** | 계산량, 텍스처/조명 의존 | 슬립, 누적 오차 |
| **실패 조건** | 어두움, 특징점 없음, 빠른 움직임 | 바퀴 슬립, 범프 |

#### 융합 방법 선택

| 방법 | 난이도 | 장점 | 단점 |
|------|--------|------|------|
| **A. robot_localization EKF** | ⭐⭐ | 표준 방법, 검증됨 | 설정 복잡 |
| B. VINS에 휠 factor 추가 | ⭐⭐⭐ | Tight coupling | 코드 수정 필요 |
| **C. 가중 평균** | ⭐ | 간단 | 최적 아님 |

> 💡 **Option A (robot_localization)** 권장 — 검증된 방법, ROS 생태계 지원

### Week 7: robot_localization EKF 구현

#### 설치 및 설정
```bash
# ROS2
sudo apt install ros-humble-robot-localization
```

#### EKF Config 파일
```yaml
# ekf.yaml
ekf_filter_node:
  ros__parameters:
    frequency: 50.0
    two_d_mode: true  # AMR은 2D
    
    # VIO 입력
    odom0: /vins_estimator/odometry
    odom0_config: [true,  true,  false,  # x, y, z
                   false, false, true,   # roll, pitch, yaw
                   true,  true,  false,  # vx, vy, vz
                   false, false, true,   # vroll, vpitch, vyaw
                   false, false, false]  # ax, ay, az
    
    # 휠 오도메트리 입력
    odom1: /wheel_odometry
    odom1_config: [true,  true,  false,
                   false, false, true,
                   true,  true,  false,
                   false, false, true,
                   false, false, false]
    
    # 공분산 설정 (튜닝 필요)
    process_noise_covariance: [...]
```

#### 테스트
- [ ] EKF 노드 실행
- [ ] `/odometry/filtered` 토픽 확인
- [ ] VIO와 휠 오도 불일치 시 동작 확인
- [ ] 공분산 튜닝

### Week 8: Fallback 전략

#### VIO 실패 감지 지표
```python
# 모니터링할 지표
class VIOHealthMonitor:
    def check_health(self):
        # 1. 트래킹 특징점 수
        if tracked_features < 30:
            return DEGRADED
        
        # 2. 재투영 오차
        if reprojection_error > 2.0:
            return DEGRADED
        
        # 3. 최적화 수렴
        if optimization_cost > threshold:
            return DEGRADED
        
        return HEALTHY
```

- [ ] 특징점 수 모니터링 (임계값: 30-50개)
- [ ] 재투영 오차 모니터링
- [ ] VINS 내부 상태 확인

#### Fallback 로직
```
상태 전이:
  NORMAL (VIO + 휠) 
    ↓ VIO 품질 저하
  DEGRADED (휠 가중치 증가)
    ↓ VIO 완전 실패
  FALLBACK (휠만 사용)
    ↓ VIO 복구
  RECOVERY (점진적 VIO 가중치 증가)
    ↓ 안정화
  NORMAL
```

- [ ] 상태 머신 구현
- [ ] 전환 시 **점프 방지** (부드러운 전환)
- [ ] 로그 기록

#### Fallback 테스트 시나리오
- [ ] 정상 주행 → 정상 동작 확인
- [ ] 카메라 손으로 가리기 → Fallback 전환
- [ ] 카메라 가림 해제 → 복구 확인
- [ ] 어두운 환경 진입/탈출

### 🔍 Section 6.3 자체 점검
1. robot_localization에서 두 센서의 가중치는 어떻게 결정되는가?
2. VIO가 실패했을 때 어떻게 감지하는가?
3. Fallback 전환 시 포즈 점프는 어떻게 방지하는가?

---

## 📋 Section 6.4: 실패 모드 분석 및 대응 (2주)

> ⚠️ VIO는 **실패할 때 어떻게 하느냐**가 실무에서 더 중요

### Week 9: 실패 모드 테스트

#### 실패 모드 1: 특징점 부족
**상황**: 빈 벽, 단조로운 바닥, 반복 패턴
```
테스트:
- 빈 복도에서 주행
- 창고의 동일한 선반 사이 주행
```
- [ ] 테스트 실행 및 동작 관찰
- [ ] 특징점 수 로깅
- [ ] 대응: Fallback to 휠 오도

#### 실패 모드 2: 빠른 움직임
**상황**: 급회전, 급가속, 충돌 등
```
테스트:
- 제자리 빠른 회전 (> 90°/s)
- 급정거
```
- [ ] IMU saturation 확인
- [ ] Motion blur 영향 확인
- [ ] 대응: 움직임 제한 또는 IMU 신뢰도 조정

#### 실패 모드 3: 조명 변화
**상황**: 창문 근처, 조명 on/off, 그림자
```
테스트:
- 밝은 창문 쪽으로 이동
- 조명 켜고 끄기
```
- [ ] 노출 변화에 따른 트래킹 품질
- [ ] 대응: Auto exposure 설정, Fallback

#### 실패 모드 4: 동적 물체
**상황**: 사람, 지게차 등이 시야에 많을 때
```
테스트:
- 사람들이 지나다니는 환경에서 주행
```
- [ ] Outlier rejection 동작 확인
- [ ] 대응: RANSAC 파라미터 조정

### Week 10: 대응 전략 구현

#### VIO Degradation 감지 로직
```cpp
enum VIOState { HEALTHY, DEGRADED, FAILED };

VIOState assessVIOHealth() {
    int num_features = feature_tracker.getNumFeatures();
    double reproj_error = estimator.getReprojectionError();
    
    if (num_features < 20 || reproj_error > 3.0) {
        return FAILED;
    } else if (num_features < 50 || reproj_error > 1.5) {
        return DEGRADED;
    }
    return HEALTHY;
}
```

#### 공분산 기반 가중치 조정
```yaml
# VIO 상태에 따라 공분산 동적 조정
HEALTHY:
  vio_position_covariance: 0.01
DEGRADED:
  vio_position_covariance: 0.1  # 신뢰도 낮춤
FAILED:
  vio_position_covariance: 999  # 사실상 무시
```

#### 복구 전략
- [ ] VIO 재초기화 조건 정의
- [ ] 점진적 신뢰도 회복
- [ ] 히스테리시스 적용 (떨림 방지)

### 🔍 Section 6.4 자체 점검
1. 가장 자주 발생하는 VIO 실패 모드는?
2. 실패를 감지하는 가장 신뢰할 수 있는 지표는?
3. Fallback에서 복구까지 얼마나 걸리는가?

---

## 📋 Section 6.5: 정량적 평가 및 문서화 (2주)

### Week 11: 정량적 평가

#### 평가 지표

| 지표 | 설명 | 목표 |
|------|------|------|
| **ATE (Absolute Trajectory Error)** | 전체 궤적 정확도 | < 1% of traveled distance |
| **RPE (Relative Pose Error)** | 짧은 구간 드리프트 | < 0.5% per meter |
| **CPU 사용량** | 연산 부하 | < 50% (single core) |
| **초기화 시간** | 시작 → VIO ready | < 5초 |
| **Fallback 빈도** | 안정성 지표 | 정상 환경에서 0 |

#### Ground Truth 확보 방법

| 방법 | 정확도 | 난이도 |
|------|--------|--------|
| 폐루프 테스트 | 중 | ⭐ |
| 마커 기반 (AprilTag) | 상 | ⭐⭐ |
| 기존 맵 대비 비교 | 중 | ⭐⭐ |
| Motion capture | 최상 | ⭐⭐⭐ |

> 💡 **폐루프 테스트** 권장: 시작점으로 돌아와서 오차 측정

#### evo 툴로 평가
```bash
# 궤적 저장
rosbag record /odometry/filtered -O trajectory.bag

# bag → TUM 형식 변환
evo_traj bag trajectory.bag /odometry/filtered --save_as_tum

# ATE 계산 (폐루프)
evo_ape tum ground_truth.tum estimate.tum -va --plot

# RPE 계산
evo_rpe tum ground_truth.tum estimate.tum -va --plot --delta 1 --delta_unit m
```

#### 성능 프로파일링
```bash
# CPU 사용량
top -p $(pgrep vins_estimator)

# 처리 시간 로깅 (코드에 추가)
auto start = std::chrono::high_resolution_clock::now();
// ... processing ...
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::high_resolution_clock::now() - start);
ROS_INFO("Processing time: %ld ms", duration.count());
```

### Week 12: 문서화 및 공유

#### 기술 문서 구성
```
1. 개요
   - 목적, 범위, 시스템 구성

2. 하드웨어 설정
   - 카메라 사양, 마운트 위치
   - 캘리브레이션 결과

3. 소프트웨어 구성
   - 노드 구조, 토픽, TF tree
   - Config 파일 설명

4. 사용 방법
   - 실행 절차
   - 파라미터 튜닝 가이드

5. 트러블슈팅
   - 자주 발생하는 문제와 해결책
   - 실패 모드별 대응

6. 성능 평가 결과
   - 정량적 지표
   - 기존 시스템 대비 비교
```

#### 결과 정리

| 항목 | 휠 오도만 | VIO + 휠 융합 | 개선 |
|------|----------|--------------|------|
| 폐루프 오차 | m | m | % |
| 드리프트/m | % | % | |
| 슬립 대응 | ❌ | ✅ | |

#### 공유
- [ ] 사내 기술 공유 발표
- [ ] GitHub 저장소 정리 (가능하면)
- [ ] 블로그 포스트 (선택)
- [ ] 데모 영상 촬영

### 🔍 Section 6.5 자체 점검
1. ATE와 RPE의 차이는?
2. 폐루프 테스트의 한계는?
3. 성능 개선의 가장 큰 요인은?

---

## ✅ Phase 6 완료 체크리스트

### 하드웨어
- [ ] 카메라 AMR에 안정적으로 장착
- [ ] Camera-IMU 캘리브레이션 완료
- [ ] Camera-Robot 변환 측정 완료

### ROS2 통합
- [ ] VINS-Fusion AMR에서 동작
- [ ] TF tree 올바르게 구성
- [ ] Nav2와 연동 확인

### 휠 오도 융합
- [ ] robot_localization EKF 구성
- [ ] 두 센서 융합 동작 확인
- [ ] 공분산 튜닝 완료

### 실패 모드 대응
- [ ] VIO 실패 감지 로직 구현
- [ ] Fallback 전략 구현 및 테스트
- [ ] 복구 로직 검증

### 평가 및 문서화
- [ ] 정량적 평가 완료 (ATE, RPE, CPU)
- [ ] 기존 시스템 대비 비교
- [ ] 기술 문서 작성
- [ ] 결과 공유

---

## 🎯 Phase 6 완료 기준

> "회사 AMR에서 VIO가 안정적으로 동작하고, 실패 상황에서도 fallback되며, 정량적 성능 지표와 기존 대비 개선점을 제시할 수 있다"

---

## 📚 참고 자료

### ROS2 패키지

| 패키지 | 용도 |
|--------|------|
| robot_localization | 센서 융합 EKF |
| nav2 | 네비게이션 스택 |
| tf2_ros | 좌표 변환 |
| ros1_bridge | ROS1-ROS2 브릿지 |

### 평가 도구

| 도구 | 용도 |
|------|------|
| evo | 궤적 평가 (ATE, RPE) |
| PlotJuggler | 실시간 데이터 시각화 |

### 캘리브레이션

| 도구 | 용도 |
|------|------|
| Kalibr | Camera-IMU 캘리브레이션 |
| camera_calibration (ROS) | 카메라 내부 파라미터 |

---

## 💡 팁

1. **점진적 접근**: 한 번에 모든 것 하려 하지 않기
2. **로그 많이 남기기**: 문제 발생 시 분석용
3. **영상 기록**: 테스트 장면 녹화 (재현용)
4. **백업**: 잘 되는 설정은 Git으로 관리
5. **안전 우선**: 테스트 시 로봇 속도 제한
6. **동료 협업**: 혼자 하지 말고 팀원과 공유

---

## 🎉 로드맵 완료 후

### 달성한 것
- AMR VIO 시스템 구축 경험
- VINS-Fusion 이해 및 수정 능력
- 센서 융합 실무 경험
- 실패 대응 및 시스템 안정화 경험

### 다음 단계 (선택)
- [ ] Loop Closure 추가 (장시간 운용)
- [ ] 다른 VIO 알고리즘 적용 (MSCKF, ORB-SLAM3)
- [ ] 딥러닝 기반 특징점 (SuperPoint 등)
- [ ] 기술 블로그 시리즈 작성
- [ ] 학회/컨퍼런스 발표

---

> "축하합니다! 이제 당신은 **실무 경험을 갖춘** AMR VIO 전문가입니다." 🎊