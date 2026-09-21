# Collapse_Study / CLAUDE.md

이 폴더는 **합성 RUL 데이터셋에서의 평균 예측 붕괴(Mean-Prediction Collapse)** 현상을 독립 연구 주제로 발전시키기 위한 공간이다.

> 상위 폴더(`C:\BMAD_PY313\CLAUDE.md`)도 함께 로드된다 — CMAPSS 데이터 구조, 공유 LSTM 백본, 실험 결과 수치 등은 거기서 참조.

---

## 연구 동기

BMAD 논문(RESS 투고 완료, 2026-09-09) 작업 중 H6 사후 분석에서 발견:

- **원래 H6 M0 FD003 RMSE = 43.23** — 겉으로는 "기준선 성능이 나쁘다"처럼 보였음
- **실제:** 5개 시드 전부에서 100개 테스트 엔진에 대해 **상수값(~87) 하나를 예측** (std < 0.0002)
- 이는 전형적인 **mean-prediction collapse** — 모델이 trivial solution(평균 예측)으로 수렴한 상태

이 현상은 특정 프로토콜 조건 3가지가 복합 작용하여 발생했고, CMAPSS FD003의 합성 데이터 균질성이 그 취약 배경이었다.

---

## 핵심 발견 (BMAD 프로젝트로부터)

### 붕괴 트리거 조건 (세 조건 동시 충족 시 발생)

| 조건 | H6 원본 상태 | 메커니즘 |
|------|-------------|---------|
| 고정 val split (seed=42) | 5개 훈련 시드 전부에 동일한 20개 엔진이 검증셋 | 검증셋 RUL 분포 고정 → 상수 예측이 val MSE 최솟값 |
| MIN_EPOCHS 없음 | patience=15, epoch 1부터 카운팅 | trivial solution 즉시 수렴 후 15 epoch만에 종료 |
| 단일 MSE loss | `MSE(final)` 하나만 사용 | trivial solution 근방에서 gradient 매우 작음 → 탈출 불가 |

### 데이터셋 취약 배경 (FD003의 구조)

- 단일 운전 조건 → op 컬럼이 엔진 간 변별력 없음
- 2가지 결함 모드만 존재 → 엔진 궤적이 두 stereotype으로 수렴
- 열역학 시뮬레이터 생성 → 동일 결함 모드 엔진의 30-cycle 윈도우가 거의 동일
- 결과: MSE landscape에서 "평균 예측"이 넓고 안정적인 local minimum

### 붕괴 탈출 증거

M3는 동일한 붕괴 유발 조건에서도 생존했다. 이유:
- 보조 브랜치 손실 (`0.05×MSE(b0) + 0.05×MSE(b1)`) → 추가 gradient 경로 제공
- trivial solution 근방에서도 브랜치별 gradient가 탈출을 가능케 함

수정 프로토콜(per-seed random split + MIN_EPOCHS=30 + MAX_EPOCHS=300) 적용 후:
- M0 FD003 RMSE: **43.23 → 12.97 ± 0.67** (정상 수렴)
- M3 FD003 RMSE: **14.78 → 13.24 ± 1.69** (M0와 유의 차이 없음, p_BH=0.754)

---

## 연구 질문 (Research Questions)

### RQ1 — 존재와 메커니즘
어떤 훈련·검증·레이블링·최적화 조건에서 RUL 회귀 모델이 input-insensitive mean-like solution으로 수렴하는가?
- validation composition × early stopping 상호작용이 핵심 (H1a)
- collapse run은 훈련 초기 PDR이 빠르게 0에 접근하고 R² ≤ 0을 보임 (H1b)
- Phase 1A (validation/labeling factorial) + Phase 1B (optimizer/loss sweep)

### RQ2 — 취약성 및 일반화
어떤 데이터셋·레이블 분포·아키텍처 특성이 MPC 취약성을 높이는가?
- **범위: C-MAPSS FD001–FD004 × MLP·1D-CNN·GRU/LSTM**
- N-CMAPSS 및 bearing 데이터(XJTU-SY 등)는 Phase 3 범위 외 — future work (Decision_log.md D1)
- 취약성 순위를 사전에 단정하지 않음 (H2c)

### RQ3 — 조기 감지
test set 없이 훈련·검증 출력만으로 MPC를 감지할 수 있는가?
- 감지 지표: PDR, R², CBR(Constant-Baseline Ratio), ISS(Input Sensitivity Score)
- 단일 `pred_std` 임계값이 아닌 복합 기준 사용
- blind adjudication으로 순환논리 방지

### RQ4 — 예방
데이터셋·아키텍처 전반에서 MPC를 신뢰성 있게 방지하는 최소 비용 개입은?
- early-stopping warm-up vs. per-seed split vs. PDR monitor+restart 비교
- 정상 run의 RMSE non-inferiority를 함께 검증

---

## 활용 가능한 기존 자산

### 데이터
```
C:\BMAD_PY313\Dataset\
├── train_FD00{1-4}.txt   ← 재사용 가능
├── test_FD00{1-4}.txt
└── RUL_FD00{1-4}.txt
```

### 코드
```
C:\BMAD_PY313\Data_Analysis\Code\
├── shared\op_condition_utils.py   ← K-means 잔차화
├── H6_fault_mode\phase2_models\
│   ├── h6_p2_model_utils.py       ← LSTM backbone, train/eval loop
│   └── h6_p2_baseline_lstm.py     ← M0 — 붕괴 재현 가능
└── Ad-hoc_Analysis\
    ├── 01_unified_data_loader.py
    └── 04_run_h6_corrected.py     ← 수정 프로토콜 레퍼런스
```

### 붕괴 재현 데이터
```
C:\BMAD_PY313\Data_Analysis\Results\
├── H6_fault_mode\              ← 원본 붕괴 결과 (raw_predictions_M0_FD003_seed*.csv)
└── H6_corrected\               ← 수정 후 결과 (비교 기준)
    └── h6_corrected_results.csv
```

---

## 이 폴더의 구조

```
Collapse_Study\
├── CLAUDE.md                    ← 이 파일
├── Research_Plan.md             ← 실험 설계, 선행 연구, 일정 (Draft v0.3)
├── Decision_log.md              ← 주요 설계 결정 및 보류 항목 기록
├── Dataset\                     ← Junction → C:\BMAD_PY313\Dataset\ (실제 복사본 아님)
├── Hypothesis\                  ← 연구 가설 문서
├── Previous Studies\            ← 선행 연구 논문 및 메모
├── Configs\
│   ├── discovery\               ← 탐색 단계 실험 설정
│   └── confirmatory\            ← 확인 단계 실험 설정
├── Data_Analysis\
│   ├── Code\                    ← 실험 스크립트
│   │   ├── 00_reproduce_and_audit.py    ← Phase 0: 관찰 재현 및 구현 감사
│   │   ├── 01_mechanism_screen.py       ← Phase 1A: validation × early stopping × labeling
│   │   ├── 02_optimization_followup.py  ← Phase 1B: patience, LR, initialization, loss
│   │   ├── 03_training_dynamics.py      ← Phase 2: epoch trajectory 집계·시각화
│   │   ├── 04_generalization.py         ← Phase 3: dataset × architecture 확인
│   │   ├── 05_detector_validation.py    ← Phase 4: calibration 및 held-out 검증
│   │   └── 06_prevention_analysis.py    ← Phase 5: 예방책 비용-효과
│   ├── Logs\                    ← epoch별 metric 로그
│   ├── Predictions\             ← run별 예측값 파일
│   └── Results\                 ← 실험 결과 CSV, 그림 (실험 시 생성)
├── Protocols\
│   ├── split_manifests\         ← val split engine ID 기록
│   ├── preregistration.md       ← 사전 등록 (임계값, 제외 기준 등)
│   └── mpc_adjudication.md      ← blind adjudication 규칙
└── Manuscript\
    ├── Figures\
    ├── Tables\
    └── Sections\                ← 논문 섹션별 초안
```

> 신규 폴더가 추가될 수 있음 — 이 구조는 현재 기준이며 연구 진행에 따라 확장됨

---

## 환경

상위 프로젝트와 동일:
```powershell
C:\BMAD_PY313\NASA_TurboFan\Scripts\activate
```
Python 3.13, PyTorch, pandas, numpy, scipy, scikit-learn

---

## 참고 문서

- `Research_Plan.md` — 전체 실험 설계, 가설, 통계 계획, 투고 전략 (Draft v0.3)
- `Decision_log.md` — 확정 결정(D1: Phase 3 bearing 제외) 및 보류 항목(FD1: 외부 audit)
- `C:\BMAD_PY313\New_Topic.md` — 이 연구의 아이디어 원문 (섹션 I.1, II.2)
- `C:\BMAD_PY313\Data_Analysis\Ad-hoc_Analysis\Ad-hoc_Analysis_Result.md` — 붕괴 원인 분석 상세 (섹션 8.1, 8.2)
- 상위 `CLAUDE.md` — 공유 LSTM 백본, 메트릭 정의, 통계 검정 방법
