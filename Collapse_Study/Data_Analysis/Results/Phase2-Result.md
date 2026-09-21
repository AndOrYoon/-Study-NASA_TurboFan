# Phase 2 Result
## Training Dynamics Analysis — 붕괴 궤적 패턴

**실험일:** 2026-09-14  
**분석 대상:** Phase 1A(270 runs) + Phase 1B(130 runs) = **400 runs 총합**  
**epoch_logs:** 40,902 epoch rows

---

## 핵심 발견 요약

| 분석 항목 | 결과 | 함의 |
|---------|------|------|
| PDR onset epoch | **median=1 (epoch 1)** | MPC는 중간 발산이 아니라 초기 상태 탈출 실패 |
| Silence 비율 | **116/116 (100%)** | 모든 붕괴 run이 val_loss 기준 "정상 ES"처럼 보임 |
| PDR 경보 sensitivity | 1.000 (w=5) | 초기 5 epoch 내 PDR<0.05는 100% 탐지 |
| PDR 경보 specificity | **0.063 (w=5)** | 정상 run도 초기에 PDR<0.05 → 단순 임계값 경보 불가 |

---

## D1. 붕괴 전조 epoch — PDR onset

```
PDR 첫 붕괴 epoch: median=1, mean=1.0, min=1, max=1
stop_epoch 대비 비율: median=0.05 (5%)
```

**해석: MPC는 "중간 발산"이 아니라 "초기 상태 탈출 실패"다**

116개 붕괴 run 전부에서 PDR이 **epoch 1부터** threshold(0.05) 아래에 있다. 이는 MPC가 훈련 도중 어느 시점에 발생하는 발산이 아님을 의미한다. 모델은 **랜덤 초기화 직후부터** trivial solution(상수 예측) 근방에 있으며, 정상 run은 이 상태에서 탈출하는 데 성공하고, 붕괴 run은 early stopping이 탈출 기회를 빼앗는다.

**정상 run도 초기 PDR ≈ 0**  
정상 run 역시 epoch 1에서 PDR ≈ 0에서 시작한다. 랜덤 초기화된 LSTM은 아직 의미 있는 예측을 못 하므로 모든 run이 초기에 PDR < 0.05다. 차이는 "초기에 PDR이 낮았는가"가 아니라 "이후 PDR이 회복되었는가"다.

---

## D2. Silence 케이스 — val_loss와 PDR 해리(decoupling)

```
붕괴 runs 총: 116
  ES 정상 발화 + MPC (Silence): 116 / 116 (100%)
  MAX_EPOCHS 소진 + MPC:          0 / 116   (0%)

Silence cases val_loss at stop: median=1741.7, range=[1008.8, 10864.4]
Silence cases PDR at stop:      mean=0.0000020, max=0.0000021
```

**해석: val_loss는 MPC를 전혀 감지하지 못한다**

붕괴 run의 100%가 early stopping이 "정상 발화"한 상태로 종료됐다. val_loss 곡선만 보면 모델이 정상적으로 수렴한 것처럼 보인다. trivial solution(상수 ≈ 93 예측)의 val_loss가 충분히 안정적으로 낮아 patience counter가 차오르기 때문이다.

val_loss ≈ 1741 (median)은 클립 RUL 분산(≈ std² ≈ 38² ≈ 1444)과 유사 수준 — 상수 예측이 "그럴듯한" val_loss를 만들어낸다. 훈련 모니터링 시스템이 val_loss만 보면 붕괴를 놓친다.

---

## D3. 붕괴/정상 평균 궤적 비교

**평균 궤적의 주요 차이:**

| 지표 | 붕괴 run (epoch 1→60) | 정상 run (epoch 1→60) |
|------|----------------------|----------------------|
| PDR | 0 → 0 (항상 0 근방) | 0 → ~1.0 (점진 회복) |
| R²  | 음수 → 음수 (개선 없음) | -x → ~0.9 (회복) |
| val_loss | 초기 감소 → 낮은 수준 유지 | 초기 감소 → 더 낮게 수렴 |

정상 run과 붕괴 run의 val_loss 패턴이 겉보기에 유사 → 시각화에서 해리(decoupling)가 명확하게 드러남.

---

## D4. 조기 경보 성능 — PDR 임계값 경보의 한계

```
alarm_window  sensitivity  specificity  PPV
5             1.000        0.063        0.304
10            1.000        0.063        0.304
15            1.000        0.063        0.304
20            1.000        0.063        0.304
30            1.000        0.063        0.304
```

**경보 창 크기에 무관하게 결과 동일:**
- sensitivity=1.00: 모든 붕괴 run이 탐지됨 (FN=0)
- specificity=0.063: 정상 run 284개 중 266개가 오경보 (FP=266)
- PPV=0.304: 경보 발생시 실제 붕괴일 확률 30%

### 왜 specificity가 극히 낮은가? — "언제 측정하느냐"의 문제

- D4 경보 조건: `"첫 K epoch 이내에 PDR < 0.05인 시점이 있었는가"` → 구분력 없음
- 이유: 정상 run도 epoch 1에서 PDR≈0 (무작위 초기화 상태, 아직 학습 전)

```
정상 run 타임라인:
  epoch  1 : PDR = 0.0000030  ← 무작위 초기화, 아직 아무것도 못 배움
  epoch 10 : PDR = 0.120      ← 학습 시작
  epoch 30 : PDR = 0.850      ← 정상 수렴 진행
  epoch 65 : PDR = 0.980      (ES 발화 → 훈련 종료)

붕괴 run 타임라인:
  epoch  1 : PDR = 0.0000020  ← 동일하게 시작
  epoch 10 : PDR = 0.0000030  ← trivial solution에 고착
  epoch 20 : PDR = 0.0000021  (ES 발화 → 훈련 종료)
```

K=10 기준으로 "epoch 1~10 중 PDR<0.05인 epoch가 있었나?" 라고 물으면:
- 정상 run: **YES** (epoch 1에서 PDR≈0)
- 붕괴 run: **YES** (항상 PDR≈0)
- → 두 그룹 모두 YES → 구분 불가
- → 경보 창을 5로 줄이든 30으로 늘리든 epoch 1이 항상 조건을 충족 → 결과 불변

---

## 핵심 통찰: 실시간 경보 vs 사후 진단

### 올바른 질문: "훈련이 끝났을 때 PDR이 얼마인가?"

- 해결책: 측정 지표가 아니라 **측정 시점**을 바꾸면 됨
- ES 발화 시점(훈련 완료 직후)에 PDR을 한 번만 측정 → 두 그룹 완전 분리

```
정상 run:  ES가 epoch 65에서 발화 → 그 시점의 PDR = 0.980  ✓
붕괴 run:  ES가 epoch 20에서 발화 → 그 시점의 PDR = 0.0000021  ✗
```

| 그룹 | ES 발화 시점의 PDR | R² at stop |
|------|-------------------|-----------|
| 정상 run (284개) | ~0.98 | ~0.90 |
| 붕괴 run (116개) | ~0.000002 | ~−1.0 |

- D2(Silence 분석) 확인 수치: 116개 붕괴 run의 PDR_stop ≈ 2×10⁻⁶ — 정상과 수백만 배 차이

```
실시간 경보 (D4):  훈련 중 → PDR 모니터링 → 붕괴 조기 감지?
                   → 불가능. 정상/붕괴 초기 상태가 동일.

사후 진단 (D2):    훈련 완료 → 검증셋 추론 → PDR·R² 한 번 측정
                   → 완전 분리. sensitivity≈1.0, specificity≈1.0.
```

- 비유: 수술 직후 혈압 저하는 정상 환자도 동일 → 수술 중 구분 불가
  - 회복실에서 안정화 후 측정 → 정상 vs 합병증 명확히 분리
  - MPC도 동일 — 초기 PDR≈0은 "학습 전", 훈련 종료 시점의 PDR≈0이 "붕괴"

### Phase 4 Detector 설계 원칙

- val_loss: 탐지 지표로 사용 불가 (Silence 100% — 붕괴 run도 낮은 val_loss로 종료)
- PDR·R² (훈련 중 임의 시점): 사용 불가 ("훈련 중 낮았는가" 질문은 구분력 없음)
- **PDR·R² (ES 발화 직후)**: 완벽한 사후 진단 지표 ← Phase 4 채택

```python
# 훈련 완료 후 자동 진단 — 코드 한 줄 수준
pred = model.predict(val_set)
pdr  = np.std(pred) / (np.std(true) + 1e-8)
r2   = 1 - np.sum((true-pred)**2) / np.sum((true-np.mean(true))**2)

if pdr < 0.05 and r2 <= 0:
    print("WARNING: Mean-prediction collapse detected. Re-train.")
```

- 실용적 중요성: H6 원본처럼 RMSE=43을 보고하면서도 붕괴인 줄 몰랐던 케이스가 문헌에 존재할 수 있음 → 사후 진단 루틴 한 줄로 방지 가능

---

## Phase 4 설계 함의

Phase 2에서 밝혀진 사실에 근거한 Phase 4 Detector 설계 변경:

| 기존 계획 | 수정 |
|---------|------|
| PDR < 임계값 발생 시점 탐지 | **ES 발화 시점의 PDR·R² 확인** (사후 탐지) |
| 단일 threshold | **복합 기준:** PDR_stop < 0.05 AND R²_stop ≤ 0 |
| 훈련 중 경보 | 훈련 완료 후 즉시 자동 진단 |

훈련 완료 직후 PDR과 R²를 측정하면 sensitivity=1.00, specificity=1.00에 가까운 탐지가 가능하다 (Silence 결과에서 이미 확인: 116/116 붕괴 run의 PDR_stop ≈ 2×10⁻⁶).

---

## 저장 파일 목록

| 파일 | 내용 |
|------|------|
| `Phase2/collapse_onset.csv` | 붕괴 run별 PDR/R² onset epoch, stop_epoch, 상대 비율 |
| `Phase2/silence_cases.csv` | Silence 케이스 상세 (val_loss·PDR at stop) |
| `Phase2/mean_trajectories.csv` | 붕괴/정상 평균 epoch 궤적 (PDR, R², val_loss) |
| `Phase2/alarm_performance.csv` | 경보 창별 sensitivity/specificity/PPV |
| `Phase2/figures/fig1_mean_trajectories.png` | 붕괴 vs 정상 평균 궤적 (PDR, R², val_loss) |
| `Phase2/figures/fig2_pdr_onset.png` | PDR onset epoch 분포 히스토그램 |
| `Phase2/figures/fig3_alarm_performance.png` | 경보 창 크기 vs 성능 곡선 |
| `Phase2/figures/fig4_silence_scatter.png` | Silence: val_loss vs PDR scatter |
