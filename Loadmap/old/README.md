# V-SLAM 개발자 되기 로드맵

## 📖 개요

이 로드맵은 **Visual SLAM (V-SLAM) 개발자**가 되기 위한 체계적인 학습 경로를 제시합니다. Visual SLAM은 카메라와 센서를 이용하여 실시간으로 환경을 인식하고 자신의 위치를 추정하는 기술로, 자율주행차, 드론, AR/VR, 로봇 등 다양한 분야에서 핵심적인 역할을 합니다.

### 🎯 이 로드맵의 목표

- **단계별 학습**: 수학 기초부터 실전 시스템 구현까지 체계적으로 학습
- **이론과 실습의 균형**: 각 개념을 배우면서 즉시 실습 프로젝트로 응용
- **실전 중심**: 오픈소스 분석과 실제 데이터셋 평가를 통한 실무 능력 배양
- **최신 트렌드 반영**: 전통적 방법론뿐 아니라 딥러닝 기반 접근까지 포함

### 📅 예상 학습 기간

**총 18-24개월** (주당 15-20시간 투자 기준)

- 수학 기초 & 컴퓨터 비전: 5-7개월
- SLAM 핵심 알고리즘 & 센서 융합: 6-9개월
- 오픈소스 분석 & 실전 응용: 5-6개월
- 딥러닝 기반 접근 (선택): 2-3개월

> ⚠️ **참고**: 학습 기간은 개인의 배경 지식과 투자 시간에 따라 달라질 수 있습니다.

### 💡 학습 방법

1. **순차적 진행**: 각 섹션을 순서대로 진행하되, 병행 실습은 동시에 수행
2. **체크리스트 활용**: 각 항목을 완료하면 체크박스를 표시하여 진행 상황 추적
3. **미니 프로젝트 필수**: 각 단계의 미니 프로젝트는 반드시 완성하여 개념 정착
4. **반복 학습**: 어려운 개념은 실습과 함께 여러 번 반복
5. **커뮤니티 활용**: 막히는 부분은 관련 커뮤니티나 논문을 통해 해결

### 🔧 사전 준비사항

#### 필수 지식
- **프로그래밍**: Python 기본 문법 (OpenCV 실습용)
- **수학**: 고등학교 수학 (미적분, 벡터 기초)
- **영어**: 기술 문서 읽기 수준

#### 개발 환경
- **OS**: Ubuntu 20.04/22.04 또는 macOS (Linux 권장)
- **IDE**: VS Code, CLion, 또는 선호하는 C++ IDE
- **하드웨어**: 
  - GPU: NVIDIA GPU 권장 (딥러닝 파트)
  - RAM: 최소 16GB (32GB 권장)
  - 웹캠: 실습용 카메라

---

## 📐 Phase 1: 수학 기초 (3-4개월)

### 1.1 선형대수 - 기초 (3Blue1Brown) [2주]

- 벡터 연산의 기하학적 의미 이해하기
- 행렬을 선형 변환으로 이해하기
- 고유값/고유벡터 직관 얻기
- 내적과 외적의 의미 파악하기

### 🎨 병행 실습 1단계: OpenCV 기초

- OpenCV 설치 및 환경 구축하기
- 웹캠으로 간단한 영상 캡처 테스트하기
- 이미지 읽기/쓰기/표시 기본 연산
- 색공간 변환 (RGB, Gray, HSV)
- 간단한 필터링 (가우시안, 중간값)

### 1.2 선형대수 - 심화 (SLAM 실전) [2주]

- 역행렬 계산 및 조건수 이해하기
- **SVD 분해의 기하학적 의미** 이해하기
- **QR 분해와 직교성** 학습하기
- Eigen으로 행렬 연산 구현하기
- 수치 안정성 고려한 코딩하기
- 희소 행렬 자료구조 활용하기

### 🎨 병행 실습 2단계: 특징점 기초

- ORB 특징점 검출 코드 작성하기
- 실시간 특징점 시각화 프로그램 만들기
- 두 이미지 간 특징점 매칭 구현하기
- 매칭 결과 시각화하고 분석하기
- RANSAC으로 outlier 제거 실습하기

### 1.3 확률과 통계 [2주]

- 베이즈 정리 개념 및 응용 학습하기
- 가우시안 분포 특성 완전히 이해하기
- 공분산 행렬 의미와 계산법 익히기
- 최대우도추정(MLE) 원리 파악하기
- 조건부 확률과 주변 확률 계산하기
- 다변량 가우시안 분포 다루기
- **불확실성 전파(Uncertainty Propagation)** 이해하기

### 🎨 병행 실습 3단계: 확률 기반 추정

- 이미지에 가우시안 노이즈 추가하고 필터링하기
- 1D 칼만 필터로 마우스 위치 추적 구현하기
- 2D 칼만 필터로 객체 추적하기
- 센서 융합 간단한 예제 만들기 (가상 센서 2개)
- 확률 분포 시각화 프로그램 작성하기
- 잡음이 있는 센서 데이터 필터링 실험

### 1.4 3D 기하학 [2-3주]

- 회전 표현 방법들(SO(3), 쿼터니언, 오일러각) 비교 이해하기
- 쿼터니언 연산 및 회전 변환 구현하기
- 강체 변환(SE(3)) 개념 마스터하기
- 회전과 평행이동 결합 연산 익히기
- **Lie 군과 Lie 대수 기초** 개념 학습하기
- **지수 맵과 로그 맵** 이해하기
- 3D 점 변환 및 좌표계 변환 능숙하게 다루기
- **동차 좌표(Homogeneous Coordinates)** 완전히 이해하기

### 🎮 미니 프로젝트 1: AR 마커 3D 좌표계 그리기

- ArUco 마커 생성하고 출력하기
- 웹캠으로 카메라 캘리브레이션 해보기
- 캘리브레이션 결과 행렬 출력하고 의미 파악하기
- ArUco 마커 인식 코드 작성하기
- PnP로 마커 포즈 추정하기
- 마커 위에 3D 좌표축 렌더링하기
- 여러 각도에서 테스트하고 안정성 확인하기

### 1.5 최적화 이론 [2-3주]

- **비용 함수(Cost Function) 설계 원리** 이해하기
- 최소자승법(Least Squares) 원리 이해하기
- **선형 최소자승과 정규방정식** 유도하기
- 비선형 최적화 기본 개념 학습하기
- Gauss-Newton 알고리즘 공부하기
- Levenberg-Marquardt 알고리즘 이해하기
- 목적 함수와 제약 조건 설정 연습하기
- 그래디언트 계산 및 자코비안 행렬 다루기
- **수치 미분 vs 해석적 미분** 비교하기
- **Robust 비용 함수(Huber, Cauchy)** 학습하기

### 🎨 병행 실습 4단계: 최적화 실습

- Ceres Solver 설치 및 기본 예제 실행
- 간단한 곡선 피팅 문제 풀기
- 자코비안 자동 미분 vs 수동 구현 비교
- 2D 포즈 그래프 최적화 구현하기
- 최적화 수렴 과정 시각화하기

### 1.6 그래프 이론 기초 [1주]

- **그래프 표현 방법** (인접 행렬, 인접 리스트)
- **최단 경로 알고리즘** (Dijkstra) 이해하기
- **Spanning Tree** 개념 학습하기
- **Pose Graph 구조** 이해하기
- **Covisibility Graph** 개념 파악하기

### 1.7 정보 이론 기초 [1주]

- **엔트로피(Entropy)** 개념 이해하기
- **KL Divergence** 학습하기
- **상호 정보량(Mutual Information)** 파악하기
- Loop Closure에서의 응용 이해하기

---

## 💻 Phase 2: C++ 프로그래밍 기초 (Phase 1-3과 병행, 2-3개월)

> 📌 **병행 전략**: 수학/비전 이론 학습 후 저녁 또는 주말에 C++ 학습

### 2.1 C++ 기초 (A Tour of C++ Ch 1-6) [3-4주]

- Ch1: The Basics - 기본 타입, 변수, 상수 이해하기
- Ch2: User-Defined Types - struct, class, enum 마스터하기
- Ch3: Modularity - 함수, 예외 처리, namespace 익히기
- Ch4: Classes - 생성자, 소멸자, 연산자 오버로딩 학습하기
- Ch5: Essential Operations - 복사/이동 의미론 완전히 이해하기
- Ch6: Templates - 템플릿 기본 개념과 제네릭 프로그래밍 익히기

### 🎨 C++ 기초 실습

- 간단한 2D Vector 클래스 구현하기 (복사/이동 포함)
- 템플릿으로 간단한 Stack 자료구조 만들기
- RAII 패턴으로 파일 핸들러 작성하기
- 예외 안전한 리소스 관리 코드 작성하기

### 2.2 C++ 중급 (A Tour of C++ Ch 7-12) [3-4주]

- Ch7: Concepts and Generic Programming - 제약 조건 이해하기
- Ch8: Library Overview - 표준 라이브러리 전체 구조 파악하기
- Ch9: Strings and Regular Expressions - 문자열 처리 마스터하기
- Ch10: Input and Output - 스트림 입출력 능숙하게 다루기
- Ch11: Containers - vector, map, unordered_map 활용하기
- Ch12: Algorithms - STL 알고리즘 사용법 익히기

### 🎨 C++ 중급 실습

- STL 컨테이너로 특징점 관리 시스템 구현하기
- 파일 I/O로 센서 데이터 읽고 쓰기
- 정규표현식으로 설정 파일 파싱하기
- STL 알고리즘으로 이미지 필터링 구현하기

### 2.3 현대 C++ 필수 개념 [2주]

- 스마트 포인터(unique_ptr, shared_ptr, weak_ptr) 완전 이해
- Move semantics 깊이 이해하고 적용하기
- 람다 표현식 다양하게 활용하기
- constexpr와 컴파일 타임 계산
- std::optional, std::variant 활용
- Range-based for loops 최적화

### 2.4 Eigen 라이브러리 마스터 [1-2주]

- Eigen 행렬/벡터 기본 연산 마스터하기
- Fixed-size vs Dynamic-size 행렬 선택 기준
- Eigen 메모리 정렬 이슈 이해하기
- Block 연산과 slicing 활용하기
- 선형 솔버 사용하기 (LU, QR, SVD)
- Eigen과 OpenCV 데이터 변환
- 성능 최적화 기법 적용하기

### 🎨 Eigen 실습

- 3D 변환 행렬 연산 구현
- 최소자승 문제 Eigen으로 풀기
- 대규모 희소 행렬 연산 실습

---

## 👁️ Phase 3: 컴퓨터 비전 기초 (2-3개월)

### 3.1 카메라 모델 [1주]

- 핀홀 카메라 모델 원리 완전히 이해하기
- 내부 파라미터(초점거리, 주점) 개념 파악하기
- 외부 파라미터(회전, 평행이동) 이해하기
- 렌즈 왜곡 모델(방사/접선 왜곡) 학습하기
- **투영 행렬 유도 과정** 완전 이해하기
- OpenCV로 카메라 캘리브레이션 실습하기
- 왜곡 보정 알고리즘 직접 구현해보기

### 3.2 이미지 처리 기초 [1주]

- 컨볼루션 연산 이해하고 구현하기
- 가우시안/중간값 필터링 원리 파악
- Sobel/Canny 엣지 검출기 사용하기
- 이미지 피라미드 구축하기
- 히스토그램 평활화 적용하기
- 모폴로지 연산 실습하기
- 이미지 그래디언트 계산 최적화하기

### 3.3 특징점 검출/매칭 [2주]

- **Harris Corner Detector** 원리 이해하기
- FAST 코너 검출기 구현하기
- SIFT 알고리즘 원리 및 특성 이해하기
- ORB 특징점 검출기 깊이 분석하기
- BRIEF/BRISK 디스크립터 비교 분석하기
- 특징점 매칭 알고리즘(BF, FLANN) 실습하기
- **Lowe's ratio test** 구현하기
- Outlier rejection (RANSAC) 상세 구현하기
- 다양한 환경에서 특징점 성능 비교 실험하기

### 3.4 이미지 워핑과 Homography [1주]

- **Homography 수학적 정의** 완전 이해하기
- DLT(Direct Linear Transform) 알고리즘 구현
- Perspective 변환 vs Affine 변환 비교
- 이미지 워핑 구현하기
- **양방향 매핑(Forward & Backward Warping)** 이해

### 🎮 미니 프로젝트 2: 파노라마 이미지 만들기

- 같은 장면을 조금씩 이동하며 3-4장 촬영하기
- 인접 이미지 간 특징점 매칭하기
- RANSAC으로 Homography 행렬 추정하기
- 이미지 warping으로 이어붙이기
- 블렌딩으로 이음새 자연스럽게 처리
- 결과 이미지 저장하고 품질 분석하기

### 3.5 광류(Optical Flow) [1주]

- **Lucas-Kanade 방법** 원리 이해하기
- **Horn-Schunck 방법** 학습하기
- Sparse vs Dense Optical Flow 비교
- OpenCV로 실시간 광류 추적 구현
- KLT Tracker 실습하기

### 3.6 에피폴라 기하학 [2-3주]

- **에피폴라 제약 조건** 수학적 유도하기
- Essential Matrix 개념 및 계산법 학습하기
- Fundamental Matrix 추정하기
- **5-point 알고리즘** 상세 이해하기
- **8-point 알고리즘** 구현하기
- 삼각측량(Triangulation)으로 3D 점 복원하기
- **재투영 오차(Reprojection Error)** 계산하기
- PnP (Perspective-n-Point) 문제 풀어보기
- **EPnP, P3P 알고리즘** 비교하기

### 🎮 미니 프로젝트 3: 스테레오 깊이 추정

- 같은 장면을 약간 다른 위치에서 2장 촬영하기
- Essential Matrix 추정하기
- 에피폴라 선 그려서 확인하기
- 카메라 포즈 복원하기 (R, t)
- 삼각측량으로 3D 점 복원하기
- 깊이 맵 시각화하기
- 실제 거리와 비교 분석하기

### 3.7 Direct vs Feature 방식 비교 [1주]

- **Feature-based 방식 장단점** 정리
- **Direct 방식(Photometric error)** 원리 이해
- Semi-direct 방식 개념 파악
- 각 방식의 적용 시나리오 분석

---

## 🗺️ Phase 4: SLAM 핵심 알고리즘 (4-6개월)

### 4.0 SLAM 개요 [1주]

- **SLAM 문제 정의** 완전히 이해하기
- **프론트엔드 vs 백엔드** 역할 구분
- **Filtering vs Optimization 방식** 비교
- **Keyframe-based vs Sliding Window** 비교
- **Incremental vs Batch 최적화** 차이점
- Visual SLAM 파이프라인 전체 흐름 파악

### 4.1 Visual Odometry [3-4주]

- **Monocular VO 파이프라인** 설계하기
- Stereo VO에서 깊이 계산 구현하기
- 특징점 추적 시스템 구축하기
- **2D-2D, 3D-2D, 3D-3D 포즈 추정** 비교
- 프레임 간 상대 포즈 추정하기
- **스케일 드리프트 문제** 이해하고 대응하기
- 키프레임 선택 전략 구현하기
- VO 정확도 평가 및 분석하기
- **Local map tracking** 구현하기

### 🎮 미니 프로젝트 4: Sequential VO 구현

- 연속 이미지에서 프레임별 포즈 추정
- 궤적 시각화 (2D top-view)
- Ground truth와 비교
- 드리프트 누적 관찰 및 분석
- 키프레임 선택 전략 실험

### 4.2 Bundle Adjustment [2-3주]

- **BA 문제를 최적화 문제로** 정식화하기
- **재투영 오차 최소화** 원리 이해
- g2o 라이브러리 설치 및 기본 사용법 익히기
- Ceres Solver로 간단한 BA 구현하기
- **Local BA와 Global BA** 차이 이해하기
- **스파스(희소) 행렬 구조** 활용하기
- **Schur Complement** 이해하기
- **Marginalization** 개념 학습하기
- BA 성능 최적화 기법 적용하기
- **Covariance 추정** 이해하기

### 4.3 Loop Closure [2-3주]

- **Place Recognition** 문제 정의
- Bag-of-Words 모델 이해하기
- **Visual Vocabulary 생성** 과정 학습
- DBoW2/DBoW3 라이브러리 사용하기
- 장소 인식 알고리즘 구현하기
- Loop candidate 검증 방법 학습하기
- **Geometric verification** 구현
- **Pose Graph Optimization** 구현하기
- Loop closure 후 맵 업데이트 처리하기
- False positive 감지 및 제거하기
- **정보 행렬(Information Matrix)** 설정하기

### 4.4 Map Management [2주]

- 키프레임 선택 기준 설계하기
- Local map과 Global map 구조 구축하기
- Map point 생성 및 삭제 전략 구현하기
- **Covisibility graph** 구축하기
- **Essential graph** 관리하기
- **Culling 전략** (keyframe, map point)
- 장시간 실행 시 메모리 관리 최적화하기
- Map serialization/deserialization 구현하기

### 4.5 Relocalization [1-2주]

- Tracking loss 감지 시스템 만들기
- 맵 기반 재정위 알고리즘 구현하기
- 키프레임 데이터베이스 활용하기
- PnP RANSAC으로 포즈 복구하기
- 재정위 후 추적 재개 처리하기
- 다양한 실패 케이스 테스트하기

---

## 🔄 Phase 5: 센서 융합 (2-3개월)

### 5.1 센서 기초 [1주]

- **IMU 센서 원리** (가속도계, 자이로스코프)
- **센서 노이즈 모델링** (White noise, Random walk)
- **Allan Variance** 분석하기
- 카메라 vs IMU 데이터 특성 비교

### 5.2 칼만 필터 심화 [2주]

- 기본 칼만 필터 원리 완전히 이해하기
- **Extended Kalman Filter (EKF)** 유도 및 구현
- **Error-State Kalman Filter (ESKF)** 학습
- Unscented Kalman Filter (UKF) 학습하기
- 상태 벡터 설계 연습하기
- 프로세스/측정 노이즈 모델링하기
- 센서 융합 EKF 시스템 구축하기
- 필터 성능 튜닝하기
- **Observability 분석** 수행하기

### 5.3 IMU 통합 [3-4주]

- IMU 센서 데이터 형식 이해하기
- **IMU pre-integration** 이론 학습하기
- **IMU 바이어스 추정 및 보정**하기
- 중력 벡터 정렬하기
- **VIO 초기화 과정** 구현하기
- Visual-Inertial Odometry 파이프라인 구축하기
- IMU-Camera 시간 오프셋 처리하기
- **Online vs Offline calibration** 비교

### 5.4 시간 동기화 [1주]

- 센서 타임스탬프 정렬 방법 구현하기
- 선형/스플라인 보간법 적용하기
- Hardware sync vs Software sync 이해하기
- 타임스탬프 drift 보정하기
- 비동기 센서 데이터 처리하기

### 5.5 캘리브레이션 [1-2주]

- Camera-IMU 외부 캘리브레이션 수행하기
- Kalibr 툴 사용법 익히기
- 캘리브레이션 정확도 검증하기
- 다중 카메라 시스템 캘리브레이션하기
- 온라인 캘리브레이션 기법 학습하기

### 5.6 파티클 필터 (선택) [1주]

- **Monte Carlo Localization** 이해하기
- 파티클 필터 원리 학습하기
- Resampling 전략 구현하기
- Global relocalization에 적용하기

---

## 💻 Phase 6: C++ 고급 및 시스템 프로그래밍 (Phase 4-5와 병행, 2개월)

### 6.1 C++ 고급 (A Tour of C++ Ch 13-16) [2주]

- Ch13: Utilities - 시간, 함수 객체, std::optional 활용하기
- Ch14: Numerics - 수치 연산, 난수 생성 학습하기
- Ch15: Concurrency - 스레드, mutex, future 이해하기
- Ch16: History and Compatibility - C++의 발전 과정 이해하기

### 6.2 멀티스레딩 [2-3주]

- std::thread 기본 사용법 익히기
- Mutex와 lock_guard로 동기화하기
- **std::unique_lock, std::scoped_lock** 활용
- 조건 변수(condition_variable) 사용하기
- **Producer-Consumer 패턴** 구현하기
- 트래킹/매핑 병렬 실행 구현하기
- 루프 클로저 독립 스레드로 처리하기
- Thread-safe 큐 구현하기
- **Race condition과 deadlock** 디버깅하기
- **std::atomic** 활용하기

### 🎨 멀티스레딩 실습

- 멀티스레드로 이미지 처리 파이프라인 구현하기
- Producer-Consumer 패턴으로 센서 데이터 큐 만들기
- std::future로 비동기 특징점 검출 구현하기
- 스레드풀 패턴 구현하고 성능 측정하기

### 6.3 성능 최적화 [2주]

- gprof/perf로 프로파일링하기
- Valgrind/cachegrind 사용하기
- 병목 구간 식별 및 개선하기
- **컴파일러 최적화 옵션** 이해하기
- 캐시 친화적 데이터 구조 설계하기
- 불필요한 복사 제거하기
- 인라인 함수 적절히 사용하기
- **SIMD 명령어 기초** (SSE/AVX)

### 6.4 디버깅 기술 [1주]

- GDB 기본 명령어 마스터하기
- 브레이크포인트와 워치포인트 활용하기
- Valgrind로 메모리 누수 찾기
- **AddressSanitizer, ThreadSanitizer** 사용하기
- 세그멘테이션 폴트 원인 추적하기
- Core dump 분석하기
- 로그 기반 디버깅 시스템 구축하기

### 🎮 C++ 종합 프로젝트: 미니 SLAM 프레임워크

- **시스템 아키텍처 설계** (UML 다이어그램)
- 클래스 설계: Frame, MapPoint, Camera, Map 구현하기
- 템플릿으로 제네릭 Optimizer 인터페이스 만들기
- 멀티스레드로 Tracking/Mapping 분리하기
- 스마트 포인터로 메모리 자동 관리하기
- STL 컨테이너로 맵 데이터 구조 구축하기
- 설정 파일 파싱 시스템 구현
- 전체 시스템 통합하고 테스트하기
- 단위 테스트 작성하기

---

## 🤖 Phase 7: ROS2 생태계 (2-3개월, Phase 3부터 병행 가능)

### 7.1 ROS2 기본 [2주]

- ROS2 설치 및 환경 설정하기
- 노드/토픽/서비스 개념 완전히 이해하기
- Publisher/Subscriber 노드 작성하기
- Service/Client 구현하기
- Action 서버/클라이언트 만들기
- 파라미터 관리 시스템 활용하기
- **Lifecycle 노드** 구현하기
- **QoS(Quality of Service)** 설정하기

### 7.2 메시지 타입 [1주]

- sensor_msgs (Image, Imu, PointCloud2) 능숙하게 다루기
- geometry_msgs (Pose, Transform, Twist) 활용하기
- nav_msgs (Odometry, Path) 사용하기
- 커스텀 메시지 타입 정의하기
- 메시지 직렬화/역직렬화 이해하기

### 7.3 TF2 변환 [1-2주]

- TF2 라이브러리 기본 사용법 익히기
- 정적 변환 브로드캐스팅하기
- 동적 변환 발행하기
- 좌표계 간 변환 조회하기
- TF tree 구조 설계하기
- 시간 동기화된 변환 처리하기
- **tf2_ros::Buffer, TransformListener** 활용

### 7.4 시각화 [1주]

- RViz2로 기본 마커 표시하기
- 카메라 이미지 시각화하기
- 궤적 경로 그리기
- 3D 포인트클라우드 표시하기
- 특징점 매칭 시각화하기
- 커스텀 RViz 플러그인 만들기
- 실시간 디버깅 정보 표시하기

### 7.5 런치 파일 및 패키지 관리 [1주]

- Python 런치 파일 작성하기
- 여러 노드 동시 실행 설정하기
- 파라미터 파일 로딩하기
- 조건부 노드 실행 구현하기
- 네임스페이스와 리매핑 활용하기
- colcon 빌드 시스템 이해하기
- package.xml, CMakeLists.txt 작성하기

### 7.6 ROS2 SLAM 통합 [1-2주]

- SLAM 노드를 ROS2로 포팅하기
- 센서 데이터 구독 및 발행
- Odometry 메시지 발행하기
- TF 브로드캐스팅 구현
- RViz로 실시간 시각화
- rosbag2로 데이터 기록 및 재생

---

## 📚 Phase 8: 오픈소스 SLAM 시스템 분석 (3-4개월)

### 8.1 ORB-SLAM3 분석 [6-8주]

- 전체 코드 구조 파악하기
- **System 클래스** 진입점 분석
- Tracking 모듈 상세 분석하기
    - 특징점 추출 및 매칭
    - 초기 포즈 추정
    - Local map tracking
    - Tracking loss handling
- Local Mapping 스레드 이해하기
    - 키프레임 삽입
    - Map point triangulation
    - Local BA
    - Keyframe culling
- Loop Closing 로직 분석하기
    - Loop detection
    - Loop fusion
    - Pose graph optimization
- Atlas 맵 관리 시스템 학습하기
- IMU 통합 부분 코드 읽기
- 각 설정 파라미터 영향 실험하기
- 빌드하고 데이터셋으로 실행해보기

### 8.2 VINS-Mono/Fusion 분석 [4-6주]

- VIO 초기화 과정 코드 분석하기
- Feature tracker 구현 분석하기
- Sliding window 최적화 이해하기
- Marginalization 구현 부분 학습하기
- Loop closure 통합 방식 파악하기
- 포즈 그래프 최적화 코드 읽기
- VINS-Fusion의 스테레오/GPS 통합 학습하기

### 8.3 기타 SLAM 시스템 (선택) [2-4주]

- **LSD-SLAM** (Direct 방식) 분석
- **DSO** (Direct Sparse Odometry) 학습
- **SVO** (Semi-direct) 이해하기
- **OpenVSLAM** 구조 파악하기

### 8.4 논문 구현 [지속적]

- ORB-SLAM 원본 논문 정독하기
- VINS-Mono 논문 읽고 수식 이해하기
- Direct 방식 SLAM (LSD-SLAM, DSO) 논문 학습하기
- 최신 learning-based SLAM 논문 서베이하기
- 주요 알고리즘 pseudocode 작성하기
- 간단한 알고리즘 직접 재현해보기
- 논문과 구현 차이점 분석하기

### 8.5 코드 수정 실험 [지속적]

- 파라미터 튜닝하여 성능 변화 관찰하기
- 다른 특징점 검출기로 교체해보기
- 최적화 알고리즘 변경 실험하기
- 자신만의 개선 아이디어 구현하기
- 실패 케이스 수집하고 원인 분석하기
- 개선 전후 정량적 비교하기

---

## 📊 Phase 9: 데이터셋과 평가 (1-2개월, 병행 가능)

### 9.1 표준 데이터셋 [2주]

- **EuRoC MAV** 데이터셋 다운로드 및 실행하기
- **KITTI Odometry** 데이터셋으로 테스트하기
- **TUM RGB-D** 데이터셋 사용하기
- **TartanAir** (challenging) 실험하기
- 각 데이터셋의 특성과 난이도 파악하기
- 실내/실외/드론 등 다양한 환경 테스트하기
- Ground truth 데이터 형식 이해하기

### 9.2 평가 지표 [1주]

- **Absolute Trajectory Error (ATE)** 계산하기
- **Relative Pose Error (RPE)** 구현하기
- **evo 툴**로 궤적 평가하기
- 궤적 시각화 및 비교하기
- 통계적 분석 수행하기 (mean, median, RMSE)
- 맵 품질 평가 지표 학습하기

### 9.3 자체 데이터 수집 [1-2주]

- ROS bag으로 센서 데이터 기록하기
- 카메라 이미지 수집 및 동기화하기
- IMU 데이터 로깅하기
- Ground truth 획득 방법 고려하기
- 다양한 조명/환경 조건 수집하기
- 데이터셋 문서화하기

### 9.4 벤치마킹 [1주]

- 여러 SLAM 시스템 성능 비교하기
- 실행 시간 및 자원 사용량 측정하기
- 성공률 및 실패 케이스 분석하기
- 실험 결과 표/그래프로 정리하기
- 재현 가능한 실험 프로토콜 작성하기

---

## 🧠 Phase 10: 딥러닝 기반 접근 (선택, 2-3개월)

### 10.1 PyTorch 기초 [2주]

- PyTorch 기본 문법 익히기
- Tensor 연산 마스터하기
- 데이터로더 및 전처리 파이프라인 구축하기
- CNN 모델 구현하고 학습시키기
- 학습 곡선 모니터링하기
- 체크포인트 저장/로딩하기
- GPU 활용 최적화하기
- ONNX로 모델 변환하기

### 10.2 학습 기반 특징 [2-3주]

- **SuperPoint** 논문 읽고 이해하기
- **SuperGlue** 매칭 네트워크 사용하기
- **LoFTR** (Detector-free matching) 학습
- 사전 학습 모델 다운로드 및 추론하기
- 기존 특징점과 성능 비교하기
- 자신의 데이터로 파인튜닝하기
- 실시간 성능 최적화하기

### 10.3 Depth Estimation [2주]

- Monocular depth estimation 논문 서베이하기
- **MiDaS/DPT** 모델 사용해보기
- Metric depth vs Relative depth 이해하기
- Depth map을 SLAM에 통합하기
- 추정 깊이의 불확실성 다루기
- Self-supervised learning 개념 파악

### 10.4 End-to-end SLAM [2-3주]

- **DROID-SLAM** 논문 정독하기
- **NeRF 기반 SLAM** 개념 학습하기
- 학습 기반 BA 알고리즘 이해하기
- Differentiable rendering 개념 파악하기
- **NICE-SLAM, Vox-Fusion** 논문 읽기
- 최신 연구 동향 팔로우하기
- 데모 코드 실행하고 실험하기

---

## ⚙️ Phase 11: 시스템 통합 및 최적화 (지속적)

### 11.1 임베디드 배포 [2-3주]

- Jetson Xavier/Orin에 환경 구축하기
- CUDA 최적화 적용하기
- TensorRT로 딥러닝 모델 최적화하기
- 라즈베리파이에서 경량 SLAM 실행하기
- 전력 소비 및 발열 관리하기
- 실시간 제약 조건 만족시키기

### 11.2 메모리 최적화 [1-2주]

- 메모리 프로파일링 도구 사용하기
- 장시간 실행 시 메모리 누수 방지하기
- Map pruning 전략 구현하기
- 메모리 풀 패턴 적용하기
- 불필요한 데이터 즉시 해제하기
- 제한된 메모리에서 동작하도록 튜닝하기

### 11.3 실패 케이스 처리 [2주]

- 빠른 움직임 상황 테스트하기
- 저조도 환경 대응 방법 구현하기
- 텍스처 부족 영역 처리하기
- 동적 객체가 많은 환경 다루기
- 급격한 조명 변화 견디기
- 센서 노이즈 및 오류 처리하기
- Graceful degradation 구현하기

### 11.4 로깅/디버깅 시스템 [1주]

- 상세한 로깅 시스템 구축하기
- 로그 레벨 관리하기 (DEBUG, INFO, WARN, ERROR)
- 중요 이벤트 기록하기
- 성능 메트릭 실시간 출력하기
- 재현 가능한 테스트 환경 만들기
- 자동화된 회귀 테스트 작성하기

---

## 🎯 Phase 12: 도메인 특화 개발 (목표에 따라, 2-3개월)

### 12.1 실내 로봇 (선택)

- 좁은 공간에서의 매핑 최적화하기
- 반복 패턴/텍스처 환경 대응하기
- 사람과의 안전한 상호작용 고려하기
- 정밀한 위치 정확도 달성하기
- 다층 맵 구축하기
- Semantic SLAM 통합 고려

### 12.2 자율주행 (선택)

- 대규모 야외 환경 매핑하기
- 고속 주행 시 안정적인 추적 유지하기
- GPS 융합 구현하기
- 차선 정보 통합하기
- HD Map과의 연동 고려하기
- 악천후 조건 대응하기

### 12.3 드론 (선택)

- 빠른 6DOF 움직임 처리하기
- 제한된 연산 자원으로 경량화하기
- 실시간 장애물 회피 지원하기
- 배터리 효율 고려하기
- 진동 및 블러 대응하기

### 12.4 AR/VR (선택)

- 낮은 레이턴시(<20ms) 달성하기
- 안정적인 6DOF 트래킹 구현하기
- 재정위 속도 최적화하기
- 가상 객체 정확한 배치하기
- 다양한 조명 조건 대응하기
- 실시간 occlusion handling 구현하기

---

## 🎓 Phase 13: 포트폴리오 프로젝트 (최종 2-3개월)

### 13.1 통합 프로젝트: 풀스택 SLAM 시스템

- 프로젝트 요구사항 정의
- 시스템 아키텍처 설계
- 프론트엔드 구현 (Tracking)
- 백엔드 구현 (Mapping, Loop Closure)
- 센서 융합 통합
- ROS2 인터페이스 구축
- 실시간 시각화 대시보드
- 다양한 환경에서 테스트
- 성능 벤치마킹
- 문서화 (README, Wiki)
- GitHub 공개 및 데모 영상 제작

### 13.2 논문 작성 (선택)

- 연구 주제 선정
- 관련 연구 서베이
- 실험 설계 및 수행
- 결과 분석 및 시각화
- 논문 작성 (LaTeX)
- ArXiv 업로드 또는 학회 투고

---

## 📖 추천 학습 자료 (체계적 정리)

### 📘 필수 교재

**수학:**

- 3Blue1Brown - Essence of Linear Algebra (YouTube)
- Gilbert Strang - Introduction to Linear Algebra (선택 챕터)
- Timothy Barfoot - State Estimation for Robotics (Appendix A, B)

**C++ 프로그래밍:**

- Bjarne Stroustrup - A Tour of C++ (2nd Edition) ⭐ 핵심
- Scott Meyers - Effective Modern C++ (참고용)
- C++ Core Guidelines (온라인 참고)

**컴퓨터 비전:**

- OpenCV 공식 튜토리얼
- Hartley & Zisserman - Multiple View Geometry (참고서)
- Szeliski - Computer Vision: Algorithms and Applications (온라인 무료)

**SLAM:**

- Joan Solà - A micro Lie theory for state estimation in robotics ⭐
- Cyrill Stachniss - YouTube SLAM 강의 시리즈
- Frank Dellaert - Factor Graphs (GTSAM 문서)

**ROS2:**

- ROS2 공식 튜토리얼
- Programming Robots with ROS2 (O'Reilly, 예정)

### 🎥 온라인 강의

- Cyrill Stachniss - Photogrammetry (YouTube)
- Cyrill Stachniss - Mobile Sensing and Robotics (YouTube)
- TUM - Computer Vision (YouTube)
- ETH Zurich - Autonomous Mobile Robots (YouTube)

### 📄 핵심 논문 (읽기 목록)

**기초:**

- MonoSLAM (Davison et al., 2007)
- PTAM (Klein & Murray, 2007)

**Feature-based:**

- ORB-SLAM (Mur-Artal et al., 2015) ⭐
- ORB-SLAM2 (Mur-Artal & Tardós, 2017)
- ORB-SLAM3 (Campos et al., 2021) ⭐

**Direct:**

- LSD-SLAM (Engel et al., 2014)
- DSO (Engel et al., 2018)
- SVO (Forster et al., 2014)

**VIO:**

- VINS-Mono (Qin et al., 2018) ⭐
- OKVIS (Leutenegger et al., 2015)
- MSCKF (Mourikis & Roumeliotis, 2007)

**Learning-based:**

- SuperPoint & SuperGlue (DeTone et al., 2018 & Sarlin et al., 2020)
- DROID-SLAM (Teed & Deng, 2021)
- NeRF-SLAM 계열 (최신 논문)

### 🛠️ 유용한 도구 및 라이브러리

**C++ 라이브러리:**

- Eigen3 (선형대수)
- OpenCV (컴퓨터 비전)
- g2o (그래프 최적화)
- Ceres Solver (비선형 최적화)
- DBoW2/3 (Place recognition)
- Sophus (Lie 군/대수)

**Python 라이브러리:**

- NumPy, SciPy (수치 계산)
- Matplotlib (시각화)
- OpenCV-Python
- evo (궤적 평가)

**시각화:**

- RViz2
- Pangolin
- Rerun
- Plotly

---

## 📅 학습 일정 예시 (18-20개월)

```
Month 1-2:   Phase 0-1 (환경 구축, 수학 기초 1/2)
Month 3-4:   Phase 1 (수학 기초 2/2) + Phase 2 시작 (C++ 기초)
Month 5-6:   Phase 3 (컴퓨터 비전 기초) + Phase 2 (C++ 중급)
Month 7:     Phase 7 시작 (ROS2 기초, 병행)
Month 8-11:  Phase 4 (SLAM 핵심) + Phase 6 (C++ 고급, 병행)
Month 12-13: Phase 5 (센서 융합)
Month 14-17: Phase 8 (오픈소스 분석) + Phase 9 (데이터셋, 병행)
Month 18-20: Phase 13 (포트폴리오 프로젝트)

선택 Phase:  Phase 10 (딥러닝), Phase 11-12 (최적화, 도메인)
```

---

## ✅ 학습 체크포인트

### Milestone 1 (3개월): 수학 및 C++ 기초 완성

- 선형대수 기본 개념 이해
- OpenCV로 간단한 영상 처리 가능
- C++ 클래스 설계 및 구현 가능

### Milestone 2 (6개월): 컴퓨터 비전 기초 완성

- 특징점 검출/매칭 구현 가능
- 카메라 캘리브레이션 수행 가능
- 파노라마, AR 마커 프로젝트 완성
- C++ STL 능숙하게 사용

### Milestone 3 (12개월): SLAM 핵심 이해

- Visual Odometry 직접 구현
- Bundle Adjustment 이해 및 적용
- 멀티스레딩 시스템 구축 가능
- ROS2로 센서 데이터 처리

### Milestone 4 (15개월): 센서 융합 및 시스템 통합

- VIO 파이프라인 이해
- 칼만 필터 구현 및 적용
- 미니 SLAM 프레임워크 완성

### Milestone 5 (18개월): 오픈소스 마스터

- ORB-SLAM3 코드베이스 이해
- 파라미터 튜닝 및 개선 가능
- 표준 데이터셋으로 평가 수행

### Final Milestone (20개월): 포트폴리오 완성

- 풀스택 SLAM 시스템 구현
- GitHub 공개 및 문서화
- 데모 영상 및 블로그 포스트

---

## 💡 학습 팁

### 효율적인 학습 전략

1. **이론 30% + 실습 70%**: 개념을 이해했다면 바로 코드로 구현
2. **작은 단위로 나누기**: 큰 주제를 1-2주 단위로 분할
3. **반복 학습**: 어려운 개념은 여러 번 다시 보기
4. **문서화 습관**: 학습 내용을 블로그나 노트에 정리
5. **커뮤니티 활용**: GitHub, Reddit, Discord에서 질문하기

### 좌절 방지 전략

1. **너무 완벽하게 하려 하지 말기**: 80% 이해하면 다음 단계로
2. **작은 성공 축적하기**: 미니 프로젝트로 성취감 얻기
3. **동료 찾기**: 스터디 그룹이나 온라인 커뮤니티
4. **휴식도 일정에 포함**: 번아웃 방지

### 아이 돌봄과 병행 전략

1. **짧은 세션 활용**: 20-30분 집중 학습
2. **우선순위 명확히**: 핵심 개념 먼저
3. **자동화 활용**: 빌드 스크립트, 테스트 자동화
4. **유연한 일정**: 아이 상태에 따라 조정

---

## 🎯 최종 목표 체크리스트

- 독립적으로 SLAM 시스템 설계 가능
- 오픈소스 SLAM 코드 읽고 수정 가능
- 논문 읽고 알고리즘 구현 가능
- 실제 로봇/드론에 SLAM 적용 가능
- GitHub에 포트폴리오 프로젝트 보유
- 기술 블로그 또는 논문 작성
- V-SLAM 관련 취업 또는 연구 진행
