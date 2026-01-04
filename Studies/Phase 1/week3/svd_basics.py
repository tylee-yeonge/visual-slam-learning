"""
Phase 1 - Week 3: SVD (특이값 분해) 기초
==========================================
SVD의 기하학적 의미와 SLAM 응용 실습

학습 목표:
1. SVD의 기하학적 의미 이해 (회전-스케일-회전)
2. 특이값의 의미 파악
3. SVD를 이용한 최소자승 해 구하기
4. SLAM에서의 SVD 활용 이해
"""

import numpy as np
np.set_printoptions(precision=4, suppress=True)

print("=" * 60)
print("Phase 1 - Week 3: SVD (특이값 분해) 기초")
print("=" * 60)

# ============================================================
# Part 1: SVD 기본 - A = U Σ Vᵀ
# ============================================================
print("\n" + "=" * 60)
print("Part 1: SVD 기본 분해")
print("=" * 60)

# 예제 행렬 (3x2)
A = np.array([
    [3, 2],
    [2, 3],
    [2, -2]
])

print("\n원본 행렬 A (3×2):")
print(A)

# SVD 분해
U, S, Vt = np.linalg.svd(A, full_matrices=True)

print("\n--- SVD 분해 결과 ---")
print(f"\nU (왼쪽 특이벡터, {U.shape}):")
print(U)
print(f"\nS (특이값들): {S}")
print(f"\nVᵀ (오른쪽 특이벡터의 전치, {Vt.shape}):")
print(Vt)

# 복원 검증: A = U @ Σ @ Vᵀ
Sigma = np.zeros((3, 2))
Sigma[:2, :2] = np.diag(S)
A_reconstructed = U @ Sigma @ Vt

print("\n--- 복원 검증 ---")
print("Σ (특이값 대각 행렬):")
print(Sigma)
print("\nU @ Σ @ Vᵀ =")
print(A_reconstructed)
print(f"\n원본과 같은가? {np.allclose(A, A_reconstructed)}")

print("\n💡 핵심 포인트:")
print("   - U: 직교 행렬 (Uᵀ @ U = I)")
print("   - S: 특이값들 (항상 양수, 내림차순)")
print("   - Vᵀ: 직교 행렬의 전치")

# ============================================================
# Part 2: SVD의 기하학적 의미 - 회전-스케일-회전
# ============================================================
print("\n" + "=" * 60)
print("Part 2: SVD의 기하학적 의미")
print("=" * 60)

# 2x2 행렬로 시각적 이해
M = np.array([
    [2, 1],
    [1, 2]
])

U2, S2, Vt2 = np.linalg.svd(M)

print("\n행렬 M:")
print(M)
print(f"\n특이값: {S2}")
print(f"\nU (두 번째 회전):\n{U2}")
print(f"\nVᵀ (첫 번째 회전):\n{Vt2}")

# 단위 원 위의 점들 변환
print("\n--- 단위 원의 변환 ---")
theta = np.radians(45)
unit_vector = np.array([np.cos(theta), np.sin(theta)])

print(f"\n입력 벡터 (단위 원 위): {unit_vector}")

step1 = Vt2 @ unit_vector
print(f"1단계 - Vᵀ (회전): {step1}")

step2 = np.diag(S2) @ step1
print(f"2단계 - Σ (스케일): {step2}")

step3 = U2 @ step2
print(f"3단계 - U (회전): {step3}")

direct = M @ unit_vector
print(f"\n직접 계산 (M @ v): {direct}")
print(f"결과 일치: {np.allclose(step3, direct)}")

print("\n💡 기하학적 해석:")
print("   1. Vᵀ: 입력 공간에서 '특이 방향'으로 좌표축 회전")
print("   2. Σ: 각 특이 방향으로 스케일링 (타원화)")
print("   3. U: 출력 공간에서 최종 방향으로 회전")

# ============================================================
# Part 3: 특이값의 의미
# ============================================================
print("\n" + "=" * 60)
print("Part 3: 특이값의 의미")
print("=" * 60)

# 랭크가 다른 행렬들
full_rank = np.array([
    [1, 2],
    [3, 4]
])

rank_deficient = np.array([
    [1, 2],
    [2, 4]  # 첫 번째 행의 2배 → 랭크 1
])

print("행렬 1 (풀 랭크):")
print(full_rank)
_, s1, _ = np.linalg.svd(full_rank)
print(f"특이값: {s1}")
print(f"랭크: {np.sum(s1 > 1e-10)}")

print("\n행렬 2 (랭크 부족):")
print(rank_deficient)
_, s2, _ = np.linalg.svd(rank_deficient)
print(f"특이값: {s2}")
print(f"랭크: {np.sum(s2 > 1e-10)}")

print("\n💡 특이값과 랭크:")
print("   - 0이 아닌 특이값의 개수 = 행렬의 랭크")
print("   - σ ≈ 0이면 그 방향으로 정보가 없음 (차원 축소)")
print("   - 조건수(condition number) = σ_max / σ_min")

cond = s1[0] / s1[-1]
print(f"\n행렬 1의 조건수: {cond:.4f}")

# ============================================================
# Part 4: 최소자승 해 (Least Squares with SVD)
# ============================================================
print("\n" + "=" * 60)
print("Part 4: 최소자승 해 (Ax ≈ b)")
print("=" * 60)

# 과결정 시스템 (방정식이 미지수보다 많음)
A_ls = np.array([
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4]
])
b_ls = np.array([2.1, 2.9, 4.2, 4.8])

print("직선 피팅 문제: y = a + b*x")
print("\nA (설계 행렬):")
print(A_ls)
print(f"\nb (관측값): {b_ls}")

# 방법 1: NumPy 최소자승
x_lstsq, residuals, _, _ = np.linalg.lstsq(A_ls, b_ls, rcond=None)
print(f"\n최소자승 해: a = {x_lstsq[0]:.4f}, b = {x_lstsq[1]:.4f}")

# 방법 2: SVD로 직접 계산
U_ls, S_ls, Vt_ls = np.linalg.svd(A_ls, full_matrices=False)

# 유사역행렬: A⁺ = V Σ⁺ Uᵀ
S_inv = np.diag(1 / S_ls)
A_pinv_svd = Vt_ls.T @ S_inv @ U_ls.T
x_svd = A_pinv_svd @ b_ls

print(f"SVD로 계산한 해: a = {x_svd[0]:.4f}, b = {x_svd[1]:.4f}")
print(f"\n두 방법 결과 일치: {np.allclose(x_lstsq, x_svd)}")

# 예측값과 잔차
y_pred = A_ls @ x_lstsq
print(f"\n예측값: {y_pred}")
print(f"잔차: {b_ls - y_pred}")
print(f"잔차 제곱합: {np.sum((b_ls - y_pred)**2):.6f}")

print("\n💡 SVD의 최소자승 해 공식:")
print("   x = A⁺b = V Σ⁺ Uᵀ b")
print("   여기서 Σ⁺는 각 σᵢ를 1/σᵢ로 바꾼 것")

# ============================================================
# Part 5: Null Space (영공간) 찾기
# ============================================================
print("\n" + "=" * 60)
print("Part 5: Null Space (Ax = 0의 해)")
print("=" * 60)

# 랭크 부족 행렬
A_null = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("행렬 A (랭크 부족):")
print(A_null)

U_n, S_n, Vt_n = np.linalg.svd(A_null)
print(f"\n특이값: {S_n}")

# 마지막 특이값이 0에 가까움 → 해당 V 열이 영공간
null_vector = Vt_n[-1, :]
print(f"\nNull space 벡터 (V의 마지막 행): {null_vector}")

# 검증: A @ null_vector ≈ 0
result = A_null @ null_vector
print(f"A @ null_vector = {result}")
print(f"영벡터에 가까운가? {np.allclose(result, 0)}")

print("\n💡 SLAM에서의 활용:")
print("   - 'Ax = 0' 형태의 동차 시스템에서 해 찾기")
print("   - Essential Matrix, Homography 계산에 사용")

# ============================================================
# Part 6: SLAM 응용 - Essential Matrix에서 R, t 추출
# ============================================================
print("\n" + "=" * 60)
print("Part 6: SLAM 응용 - Essential Matrix 분해")
print("=" * 60)

# Essential Matrix 예제 (실제로는 특이값이 [σ, σ, 0] 형태)
# 여기서는 개념 이해를 위한 간단한 예제
E_example = np.array([
    [0, -0.5, 0.2],
    [0.5, 0, -0.8],
    [-0.2, 0.8, 0]
])

print("Essential Matrix E (예제):")
print(E_example)

U_e, S_e, Vt_e = np.linalg.svd(E_example)

print(f"\n특이값: {S_e}")
print("(이상적인 E는 [σ, σ, 0] 형태)")

# W 행렬 (회전 행렬 추출용)
W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])

print("\nW 행렬 (90° Z축 회전):")
print(W)

# 두 가지 가능한 R
R1 = U_e @ W @ Vt_e
R2 = U_e @ W.T @ Vt_e

# t는 U의 세 번째 열
t = U_e[:, 2]

print("\n--- 분해 결과 ---")
print(f"R1:\n{R1}")
print(f"\ndet(R1) = {np.linalg.det(R1):.4f}")

print(f"\nt (평행이동 방향): {t}")

print("\n💡 Essential Matrix 분해 핵심:")
print("   - E = U diag(σ,σ,0) Vᵀ 형태")
print("   - R은 4가지 후보 (2개 R × 2개 t 부호)")
print("   - 실제 사용 시 'cheirality check'로 올바른 해 선택")

# ============================================================
# Part 7: 저랭크 근사 (이미지 압축 개념)
# ============================================================
print("\n" + "=" * 60)
print("Part 7: 저랭크 근사 (이미지 압축 개념)")
print("=" * 60)

# 8x8 "이미지" 예제
np.random.seed(42)
image = np.random.randint(0, 256, (8, 8)).astype(float)

print("원본 '이미지' (8×8):")
print(image.astype(int))

U_img, S_img, Vt_img = np.linalg.svd(image)
print(f"\n특이값: {S_img.round(2)}")

# k=2로 근사 (8개 중 2개만 사용)
k = 2
U_k = U_img[:, :k]
S_k = np.diag(S_img[:k])
Vt_k = Vt_img[:k, :]

image_approx = U_k @ S_k @ Vt_k

print(f"\n저랭크 근사 (k={k}):")
print(image_approx.astype(int))

# 압축률 계산
original_params = 8 * 8  # 64개
compressed_params = k * (8 + 1 + 8)  # U의 k열 + k개 특이값 + V의 k행

print(f"\n원본 파라미터 수: {original_params}")
print(f"압축 파라미터 수: {compressed_params}")
print(f"압축률: {compressed_params/original_params*100:.1f}%")

# 오차
error = np.linalg.norm(image - image_approx, 'fro')
print(f"프로베니우스 노름 오차: {error:.2f}")

print("\n💡 저랭크 근사의 의미:")
print("   - 상위 k개 특이값만 사용하면 '중요한' 정보 유지")
print("   - 이미지 압축, 노이즈 제거, 차원 축소에 활용")

# ============================================================
# 정리
# ============================================================
print("\n" + "=" * 60)
print("📝 Week 3 정리")
print("=" * 60)

print("""
✅ SVD 기본
   - A = U Σ Vᵀ (모든 행렬에 적용 가능)
   - U, V: 직교 행렬
   - Σ: 특이값 대각 행렬 (양수, 내림차순)

✅ 기하학적 의미
   - 모든 선형 변환 = 회전 → 스케일 → 회전
   - 특이값 = 각 방향의 스케일링 정도

✅ 핵심 응용
   - 최소자승 해: x = V Σ⁺ Uᵀ b
   - Null space: Ax = 0의 해 = σ=0에 대응하는 V의 열
   - 저랭크 근사: 상위 k개 특이값으로 압축

✅ SLAM에서의 활용
   - Essential Matrix → R, t 추출
   - Homography 분해
   - PnP 문제 해법
   - Triangulation
""")

print("🎯 다음 단계:")
print("   - svd_quiz.py로 퀴즈 풀기")
print("   - Week 4: 회전 표현 (회전 행렬, 오일러 각, 쿼터니언)")
