# AMR VIO 전문가 로드맵

> 🎯 **목표**: AMR 환경에 특화된 Visual-Inertial Odometry 전문가  
> ⏰ **기간**: 15-18개월 (주 7-10시간) — 12개월은 best case  
> 👶 **전제**: 4개월 딸과 함께하는 직장인 아빠

---

## 📊 전체 개요

| Phase | 기간 | 핵심 목표 | 주간 시간 |
|-------|------|----------|----------|
| 0 | 2주 | 환경 세팅 + VINS 먼저 돌려보기 | 5시간 |
| 1 | 2개월 | 수학 핵심 (선형대수, 3D 기하) | 7시간 |
| 2 | 2개월 | 컴퓨터 비전 기초 | 7시간 |
| 3 | 3개월 | Visual Odometry + Bundle Adjustment | 10시간 |
| 4 | 3-4개월 | **VIO 핵심** (IMU 융합) | 10시간 |
| 5 | 2개월 | VINS-Fusion 코드 분석 | 7시간 |
| 6 | 2-3개월 | 회사 AMR 실적용 + 실패 모드 대응 | 7시간 |

---

## 🚀 Phase 0: 역순 시작 (2주)

> **목표**: 먼저 돌려보고, 이해 안 되는 부분을 채워가는 방식

### 0.1 환경 구축

- Ubuntu 22.04 또는 Docker 환경 준비
- ROS2 Humble 설치
- VINS-Fusion 빌드 (ROS1 기반 → ros1_bridge 또는 ROS2 포팅 버전 확인)
- Realsense D435i 드라이버 설치 (있다면)

### 0.2 VINS-Fusion 첫 실행

- EuRoC MH_01_easy 데이터셋 다운로드
- VINS-Fusion 실행 및 RViz 시각화
- 궤적 결과 확인
- **모르는 개념 목록 작성** (이게 Phase 1-4의 학습 동기)

### 0.3 평가 도구 설치

- evo 툴 설치 (`pip install evo`)
- 기본 사용법 익히기 (ATE, RPE 계산)

### 🎯 Phase 0 체크포인트
> "VINS-Fusion이 돌아가는 것을 보았고, 무엇을 모르는지 안다"

---

## 📐 Phase 1: 수학 핵심 (2개월)

> **원칙**: SLAM에 쓰이는 것만, 깊이보다 이해 우선

### 1.1 선형대수 (3주)

**최소 필수:**
- 3Blue1Brown "Essence of Linear Algebra" 전체 시청
- 행렬 = 선형 변환 이해
- 고유값/고유벡터 직관
- SVD 기하학적 의미 (특이값 = 스케일)

**실습:**
- NumPy로 행렬 분해 실습
- Eigen 기본 사용법

> 💡 수학 증명보다 "왜 이게 SLAM에서 쓰이는가"에 집중

### 1.2 3D 기하학 (3주)

**최소 필수:**
- 회전 표현: 회전행렬, 쿼터니언
- SE(3): 회전 + 평행이동 결합
- 동차 좌표 (Homogeneous Coordinates)

**실습:**
- ROS TF2로 좌표 변환 복습
- Sophus 라이브러리 기본 사용

**선택적 심화 (나중에):**
- Lie 군/대수 기초 (VINS 이해 시 필요하면)

### 1.3 최적화 기초 (2주)

**최소 필수:**
- 최소자승법 원리
- Gauss-Newton 알고리즘 개념
- 비용 함수 (Cost Function) 이해

**실습:**
- Ceres Solver로 곡선 피팅 예제 실행

### 🎯 Phase 1 체크포인트
> "VINS-Fusion 코드에서 `Eigen::Matrix`, `Quaterniond`가 뭔지 안다"

---

## 👁️ Phase 2: 컴퓨터 비전 기초 (2개월)

### 2.1 카메라 모델 (2주)

- 핀홀 모델 이해
- 내부 파라미터 (fx, fy, cx, cy)
- 렌즈 왜곡 모델
- **카메라 캘리브레이션 직접 수행** (체스보드)

**실습:**
- OpenCV `calibrateCamera` 실습
- Kalibr 툴 사용법 익히기 (Camera-IMU 캘리브레이션)

### 2.2 특징점 검출/매칭 (2주)

- ORB 특징점 이해
- Feature Matching (BF, FLANN)
- RANSAC으로 Outlier 제거

**실습:**
- 두 이미지 간 특징점 매칭 구현
- 매칭 시각화

### 2.3 에피폴라 기하학 (3주)

- Essential Matrix 개념
- Fundamental Matrix
- 삼각측량 (Triangulation)
- PnP 문제 이해

**미니 프로젝트:**
- 두 이미지에서 3D 점 복원하기

### 2.4 광류 (Optical Flow) (1주)

- Lucas-Kanade 방법 이해
- KLT Tracker 실습
- VINS에서 특징점 추적 방식 이해

### 🎯 Phase 2 체크포인트
> "VINS-Fusion의 feature_tracker 노드가 뭘 하는지 이해"

---

## 🗺️ Phase 3: Visual Odometry & Bundle Adjustment (3개월)

### 3.1 Visual Odometry 파이프라인 (4주)

- VO 전체 흐름 이해
- 2D-2D 포즈 추정 (Essential Matrix)
- 3D-2D 포즈 추정 (PnP)
- 키프레임 선택 전략

**미니 프로젝트:**
- 간단한 Monocular VO 구현
- 궤적 시각화 및 drift 관찰

### 3.2 Bundle Adjustment (4주)

- BA 문제 정의 (재투영 오차 최소화)
- g2o 또는 Ceres로 간단한 BA 구현
- Local BA vs Global BA 개념

**실습:**
- g2o 튜토리얼 예제 실행
- 작은 규모 BA 직접 구현

### 3.3 스케일 문제 이해 (2주)

- Monocular SLAM의 스케일 모호성
- Stereo로 스케일 복원
- **IMU로 스케일 복원** (Phase 4 연결점)

### 🎯 Phase 3 체크포인트
> "왜 순수 Vision만으로는 부족하고, IMU가 필요한지 설명 가능"

---

## 🔄 Phase 4: VIO 핵심 (3-4개월) ⭐

> **이 Phase가 핵심입니다. 여기에 가장 많은 시간 투자**  
> ⚠️ Pre-integration은 예상보다 오래 걸릴 수 있음 — 여유 있게 잡기

### 4.1 IMU 기초 (2주)

- 가속도계/자이로스코프 원리
- IMU 노이즈 모델 (White noise, Random walk)
- IMU 바이어스 개념
- Allan Variance (개념만)

**실습:**
- ROS2에서 IMU 데이터 구독/시각화

### 4.2 센서 융합 기초 (3주)

- 칼만 필터 복습/학습
- Extended Kalman Filter (EKF)
- Error-State Kalman Filter 개념

**실습:**
- 간단한 1D/2D 칼만 필터 구현
- IMU + GPS 융합 예제 (개념 이해용)

### 4.3 IMU Pre-integration (4-5주)

> ⚠️ VIO의 핵심 개념입니다. VINS-Mono 논문의 핵심. 막히면 커뮤니티 질문 활용!

- Pre-integration 이론 이해
- 왜 필요한가 (키프레임 간 적분)
- 바이어스 보정 이해

**자료:**
- VINS-Mono 논문 Section III-B 정독
- "On-Manifold Preintegration" 논문 (선택)

### 4.4 VIO 초기화 (2주)

- Vision-only 초기화
- 중력 벡터 정렬
- 스케일 추정
- 바이어스 초기 추정

### 4.5 Camera-IMU 캘리브레이션 (1주)

- Kalibr로 Camera-IMU Extrinsic 캘리브레이션
- 시간 오프셋 추정

**실습:**
- Realsense D435i로 캘리브레이션 수행

### 🎯 Phase 4 체크포인트
> "VINS-Fusion의 estimator 노드에서 IMU factor가 어떻게 작동하는지 이해"

---

## 💻 Phase 5: VINS-Fusion 코드 분석 (2개월)

### 5.1 코드 구조 파악 (2주)

- 전체 노드 구조 이해
- `feature_tracker`: 특징점 추적
- `vins_estimator`: 핵심 추정기
- 데이터 흐름 파악

### 5.2 주요 모듈 분석 (4주)

- `feature_manager.cpp` 분석
- `estimator.cpp` 분석
- `factor_graph` 구조 이해
- `imu_factor.cpp` 분석 (Pre-integration)

### 5.3 파라미터 실험 (2주)

- 주요 파라미터 역할 이해
- 파라미터 튜닝 실험
- 성능 변화 관찰

### 🎯 Phase 5 체크포인트
> "VINS-Fusion을 수정하여 파라미터를 바꾸거나 로깅을 추가할 수 있다"

---

## 🤖 Phase 6: 회사 AMR 실적용 (2-3개월)

### 6.1 하드웨어 준비 (2주)

- AMR에 카메라 마운트 설계/제작
- Camera-IMU 캘리브레이션
- 외부 파라미터 측정 (로봇 좌표계 → 카메라)

### 6.2 ROS2 통합 (3주)

- VINS-Fusion ROS2 포팅 또는 ros1_bridge 설정
- TF tree 설계: `odom → base_link` vs `map → odom → base_link`
- Nav2와의 연동 방식 결정 (VIO를 odometry source로? 보조로?)
- 실시간 테스트

### 6.3 휠 오도메트리 융합 (3주)

> ⚠️ AMR에서 가장 실용적인 조합: Camera + IMU + Wheel Encoder

- 휠 오도메트리와 VIO 결과 비교
- 간단한 융합 전략 구현 (EKF 또는 가중 평균)
- Robust한 Fallback 전략 (VIO 실패 시 휠 오도만)

### 6.4 실패 모드 분석 및 대응 (2주)

> ⚠️ VIO는 실패할 때 어떻게 하느냐가 실무에서 더 중요

- 특징점 부족 환경 테스트 (빈 벽, 단조로운 바닥)
- 빠른 회전/급가감속 시 IMU saturation 대응
- 조명 변화 대응 (창문, 조명 on/off)
- VIO degradation 감지 로직 구현
- 휠 오도메트리 fallback 자동 전환

### 6.5 평가 및 문서화 (2주)

**정량적 평가 (evo 툴 사용):**
- ATE (Absolute Trajectory Error) 측정
- RPE (Relative Pose Error) 측정
- CPU/메모리 사용량 프로파일링
- 초기화 시간 측정 (로봇 시작 → VIO ready)

**문서화:**
- 기존 Cartographer/Nav2와 성능 비교 레포트
- 결과 정리 및 사내 공유
- GitHub 또는 블로그에 기록

### 🎯 Phase 6 체크포인트
> "회사 AMR에서 VIO가 돌아가고, 실패 상황에서도 안정적으로 fallback되며, 정량적 성능 지표를 제시할 수 있다"

---

## 🤝 커뮤니티 & 지원

> 독학의 한계를 넘기 위한 외부 리소스

- SLAM KR 커뮤니티 가입 (카카오톡 오픈채팅, 페이스북)
- 회사 Visual SLAM 프로젝트 (2026년 계획) 참여 의사 표명
- 막힐 때 질문할 수 있는 채널 확보
- 월 1회 학습 진행 상황 정리 및 공유

---

## 📅 현실적 주간 스케줄

```
평일:
  - 출퇴근 (30분 x 2): 영상/논문 시청
  - 점심 (30분): 개념 복습
  - 저녁 (아이 재운 후, 가능한 날만): 1시간 코딩 실습

주말:
  - 토요일 오전: 2-3시간 집중 실습
  - 일요일: 가족 시간 (학습 X)

예상: 주 7-10시간
```

---

## 📚 핵심 자료 (최소화)

### 영상

| 자료 | 용도 | 우선순위 |
|------|------|---------|
| 3Blue1Brown 선형대수 | 수학 기초 | ⭐⭐⭐ |
| Cyrill Stachniss SLAM 강의 | VO/BA 이론 | ⭐⭐⭐ |
| First Principles CV (YouTube) | 비전 기초 | ⭐⭐ |

### 논문

| 논문 | 용도 | 우선순위 |
|------|------|---------|
| VINS-Mono (Qin et al., 2018) | VIO 핵심 | ⭐⭐⭐ 필독 |
| ORB-SLAM3 (Campos et al., 2021) | 참고용 | ⭐⭐ |

### 책 (참고용)

| 책 | 용도 |
|---|------|
| State Estimation for Robotics (Barfoot) | 수학, 필터링 |
| Multiple View Geometry | 비전 기하 (사전처럼 사용) |

### 도구

| 도구 | 용도 |
|------|------|
| evo | 궤적 평가 (ATE, RPE) |
| Kalibr | Camera-IMU 캘리브레이션 |

---

## ✅ 마일스톤 체크리스트

### 3개월 후
- VINS-Fusion 빌드 및 실행 성공
- 선형대수 기초 이해
- 카메라 캘리브레이션 수행 가능
- evo 툴로 궤적 평가 가능

### 6개월 후
- 간단한 VO 구현
- BA 개념 이해 및 실습
- 에피폴라 기하학 이해

### 9개월 후
- VIO 파이프라인 전체 이해
- IMU Pre-integration 개념 파악
- VINS-Fusion 코드 읽기 시작

### 12개월 후
- VINS-Fusion 코드 수정 가능
- 회사 AMR에 카메라 마운트

### 15-18개월 후
- AMR에서 VIO 동작
- 휠 오도메트리 + VIO 융합
- 실패 모드 대응 완료
- 정량적 평가 레포트 완성
- 결과 문서화 및 공유

---

## 💡 핵심 원칙

1. **역순 학습**: 먼저 돌려보고, 모르는 것을 채운다
2. **80% 이해하면 다음으로**: 완벽 추구 X
3. **실무 연결**: 항상 "이게 AMR에 어떻게 쓰이나?" 생각
4. **기록 습관**: 배운 것을 짧게라도 기록 (블로그/노션)
5. **가족 우선**: 학습은 마라톤, 번아웃 방지
6. **커뮤니티 활용**: 막히면 혼자 끙끙대지 말고 질문하기

---

## ❌ 이 로드맵에서 **제외한 것**

| 제외 항목 | 이유 |
|----------|------|
| 딥러닝 기반 SLAM | 당장 AMR에 불필요, 나중에 |
| NeRF-SLAM | 연구 단계, 실무 적용 어려움 |
| 드론/자율주행/AR | AMR 외 도메인 |
| 논문 작성 | 블로그로 대체 |
| Loop Closure 심화 | VIO 이후 선택적으로 |
| 파티클 필터 | EKF로 충분 |

---

> 📝 **Note**: 이 로드맵은 **"AMR VIO 전문가"**를 목표로 최적화되었습니다.  
> 나중에 범용 V-SLAM 전문가로 확장하고 싶다면 Loop Closure, Relocalization 등을 추가하세요.