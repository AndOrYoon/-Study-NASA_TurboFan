# Decision Log — Collapse Study
**프로젝트:** Mean-Prediction Collapse in Deep RUL Regression  
**작성 시작:** 2026-09-11

이 파일은 연구 설계 과정에서 내려진 주요 결정과 그 근거를 시간 순으로 기록한다.  
실험 진행 중 방향이 흔들릴 때 이 기록을 먼저 확인할 것.

---

## 확정 결정 (Confirmed Decisions)

### D1 — Phase 3 범위: Bearing 데이터 제외
**결정일:** 2026-09-11  
**결정 내용:** Phase 3 일반화 실험의 데이터셋 범위를 C-MAPSS FD001–FD004로 한정하고, XJTU-SY / PRONOSTIA / IMS bearing 데이터는 포함하지 않는다.  
**아키텍처 범위:** MLP · 1D-CNN · GRU(또는 LSTM) 3종으로 한정.  
**근거:**
- Critique(Research_Plan_Critique.md §1d)에서 지적한 "주장 크기와 연구 규모 불일치" 위험을 줄이기 위함.
- Bearing 데이터는 RUL 정의 방식, 센서 구조, 고장 모드가 CMAPSS와 달라 MPC 발생 조건의 직접 비교가 어렵다.
- C-MAPSS 4종 × 3 architecture 조합만으로도 "LSTM 특이성", "FD003 특이성" 배제라는 일반화 목적을 충족한다.  
**향후 확장 조건:** Phase 3 결과가 2개 이상 dataset, 2개 이상 architecture에서 MPC를 재현하면, bearing 데이터로의 확장을 future work로 논문에 명시한다.

---

### D3 — 외부 논문 Audit 포함 결정 (C5: Replication Warning)
**결정일:** 2026-09-15  
**결정 내용:** 외부 audit을 연구 범위에 포함한다. FD003에서 RMSE 30–45를 보고한 published 논문 3–5편의 methods 섹션을 체크리스트(val split 방식·ES 조건·loss 설정)로 분류하여 "MPC 발생 개연성 있음/불확실/낮음"을 판정한다.

**근거 (FD1 재검토 기준 충족):**
- Phase 1A 결과: A1+B2+C2 조건에서 MPC rate = **0.80 (≥ 0.60 기준 충족)**
- A1+B2 interaction이 명확함: B1·B3는 A·C 무관하게 MPC 0%, B2는 A·C에 따라 0~90%
- 세 조건(val split × early stopping × labeling)의 상호작용이 통계적으로 구분 가능한 수준

**기여 격상:**  
이 결정으로 C5는 "자기 사례(n=1) 경고" → "재현성 취약 구역의 체계적 문헌 프로토콜 감사"로 격상된다. 코드 재실행 없이 텍스트 프로토콜 감사만으로 수행 가능.

**실행 방식:**
1. FD003 RMSE 30–45 보고 논문 3–5편 systematic search (Google Scholar, IEEE Xplore)
2. 각 논문 Methods/Experiments에서 확인: (a) validation split 방식, (b) early stopping 조건, (c) loss function
3. Phase 1A factorial 결과(A1+B2+C1 → 0.90, A1+B2+C2 → 0.80)를 기준으로 "발생 개연성" 분류
4. 주장 수위는 "붕괴했다"가 아닌 **"프로토콜 서술에 근거한 붕괴 개연성"**에 한정

---

## 추가 결정 필요 항목 (Further Decision After Phase 1)

### FD1 — 외부 논문 Audit (C5: Replication Warning) 포함 여부
**기록일:** 2026-09-11  
**현재 상태:** → **D3으로 확정 결정됨 (2026-09-15)**

---

### FD2 — 외부 Audit 대상 논문 선정 및 투고 저널 결정
**기록일:** 2026-09-15  
**현재 상태:** → **D4로 확정 결정됨 (2026-09-15)**

#### 배경
D3(외부 audit 포함)는 확정됐지만, audit 대상 논문 선정과 투고 저널은 미결이다.

#### RESS 제외 사유
RESS (Reliability Engineering & System Safety)는 현재 **신호 진단·고장 예측 관련 scope이 아닌 것으로 확인**됐다.  
→ 이 연구(MPC 메커니즘·예방·감사)의 투고 대상에서 RESS는 제외.

#### 결정 필요 항목
1. **투고 저널 결정** — 사용자가 저널을 선택하면 이 항목에 기록하고 FD2를 D4로 확정
2. **Audit 대상 논문 선정** — 저널 결정 후 scope에 맞는 FD003 고-RMSE 보고 논문 3–5편 systematic search 실시
3. **기대 조건:** 선정 논문은 peer-reviewed, FD003 RMSE 30–45 범위, Methods 섹션에 val split·ES 조건 서술 있어야 함  

#### 배경
BMAD 프로젝트(RESS 투고 완료, 2026-09-09)에서 H6 M0 FD003의 원래 RMSE는 43.23이었으나 사후 분석 결과 모든 5개 시드가 상수값(~87)을 출력하는 MPC 상태였음이 확인됐다. 수정 프로토콜 적용 후 RMSE는 12.97 ± 0.67로 회복됐다. 이 사례는 **프로토콜 조건만으로 기준선이 30 RMSE 이상 왜곡될 수 있음**을 보여준다.

#### 쟁점
Research_Plan_Critique.md(§4, C5 행)는 이 기여를 다음과 같이 분류했다:
- **현재 상태 (confession):** "우리가 낸 실수를 우리가 고쳤다" — n=1 사례, 자기 논문, RESS 투고 전에 발견·수정 완료. 외부에서 보면 "우리 논문에서 벤치마크 오염은 가정형"에 불과하다.
- **목표 상태 (finding):** published FD003 고-RMSE 논문(RMSE 30–45 범위) 3–5편을 선별하고, 그 논문이 서술한 프로토콜(val split 방식, early stopping 조건, loss 설정)을 체크리스트로 분석하여 MPC 발생 개연성을 논증. 코드 재실행 없이 **텍스트 프로토콜 감사**만으로도 가능하다. 이 경우 C5는 "단일 자기 사례 경고" → "재현성 취약 구역의 체계적 문헌 감사"로 격상되며, full-length journal 정당화의 핵심 근거가 된다.

#### 외부 Audit의 실행 방식 (보류 중이나 검토한 내용)
1. FD003에서 RMSE 30–45를 보고한 published 논문 3–5편 수집 (systematic search).
2. 각 논문의 Methods/Experiments 섹션에서 다음을 확인:
   - validation split 방식 (고정 seed 여부, engine-level 분리 여부)
   - early stopping 조건 (patience, MIN_EPOCHS 설정 여부)
   - loss function (단일 MSE 여부, auxiliary loss 여부)
3. 세 조건(Phase 1 factorial 결과 기준)과 대조하여 "MPC 발생 개연성 있음 / 불확실 / 개연성 낮음" 분류.
4. 분류 근거를 인용 가능한 형태로 기술하고, RMSE ≈ 13 대비 상대 왜곡 폭 계산.
5. 주장 수위는 "붕괴했다"가 아니라 **"프로토콜 서술에 근거한 붕괴 개연성"**에 한정.

#### Phase 1 이후로 미룬 이유
- 외부 audit의 논증 강도는 Phase 1 factorial 결과에 전적으로 의존한다.
  - Phase 1이 세 조건(val composition × early stopping × labeling)의 상호작용을 **깨끗하게** 식별하면 → audit 논증이 강력해지고 추가 작업 대비 기여 효율이 높다.
  - Phase 1 결과가 복잡하거나 조건 간 경계가 불분명하면 → audit을 넣어도 논증이 약해지므로 제외하는 것이 낫다.
- 실행 가능성: 외부 논문 systematic search + 프로토콜 텍스트 감사는 1–2주 추가 작업. Phase 1 결과를 보고 투자 여부를 결정하는 것이 합리적.

#### Phase 1 후 재검토 기준
다음 조건 중 하나 이상이 충족되면 외부 audit 추가를 강력 권장:
- Phase 1에서 "val composition × early stopping" 상호작용이 통계적으로 유의하고 효과 크기가 크다 (odds ratio ≥ 3 또는 절대 발생률 차이 ≥ 30%p).
- "fixed split + no warmup + MSE" 조건에서 MPC 발생률이 ≥ 60% (n=10 seeds 기준).
- FD003 외 데이터셋에서도 동일 조건 조합에서 MPC가 재현된다.

다음 조건이면 audit 없이 진행 권장:
- Phase 1 결과가 혼재 (조건별 MPC 발생률 차이가 크지 않거나 interaction 불명확).
- Phase 1에서 새로운 dominant factor가 발견되어 original 3-trigger 프레임 자체가 수정된다.

### D2 — Phase 4 재설계: "Detector 개발" → "예방 분류체계 + 소급 진단 도구"

**결정일:** 2026-09-15  
**결정 내용:** Phase 4의 목적을 **Option C**로 전환한다. 원래 계획("test set 없는 실시간 MPC 경고 규칙 개발")을 폐기하고, 다음 두 파트로 재구성한다.

- **Part A — 예방 분류체계 (Prevention Taxonomy):** Phase 1B + Phase 3B에서 확인된 모든 예방 개입을 체계적으로 비교. MPC 감소율 + RMSE non-inferiority + 구현 비용 3축으로 평가.
- **Part B — 소급 진단 도구 (Retrospective Audit Tool):** 이미 학습 완료된 모델의 출력(pred.csv)만으로 MPC 여부를 사후 진단하는 PDR/R² 임계값 검증. "실시간 알람"이 아닌 "사후 검진" 도구.

기존 Phase 5(예방 전략)는 Part A로 흡수 통합하고 별도 Phase로 존치하지 않는다.

**근거:**
- Phase 3B에서 H_mech 가설이 확증됨: forget gate → 0 이 MPC의 인과적 필요조건임이 V2_fg1 개입 실험으로 직접 검증됨.
- 원인이 명료한 상태에서 "실시간 detector"는 블랙박스 상황에서의 도구이므로 1차 기여로서 가치가 낮아짐.
- 반면 (1) 알려진 모든 예방책을 비용-효과 축으로 비교하는 분류체계, (2) 기존 학습 완료 모델을 출력값만으로 감사하는 소급 도구는 실무적 기여가 명확하고 Phase 3B 결과와 자연스럽게 연결됨.
- Phase 2에서 이미 확인: PDR/R² 실시간 알람은 작동 불가(sensitivity 0), 소급 진단은 완벽 분리. 이 구분이 Phase 4 재설계의 핵심 근거.

**Phase 4에 추가될 예방 방법 목록:**

| 방법 | 출처 | MPC rate 변화 | 비고 |
|------|------|:---:|------|
| GRU로 교체 | Phase 3 | 0.80 → 0.00 | 아키텍처 변경 |
| forget gate = 1 (V2_fg1) | Phase 3B | 0.80 → 0.00 | LSTM 구조 변경 |
| MAE loss | Phase 1B | 0.80 → 0.00 | loss 변경 |
| bias_init = train_mean | Phase 1B | 0.80 → 0.00 | 초기화 변경 |
| forget gate clamp [ε, 1] | Phase 4 신규 | TBD | ε 임계값 분석 포함 |
| patience = 30 | Phase 1B | 0.80 → 0.20 | 부분 개선 |
| per-seed random split | Phase 1A | 0.80 → 0.00 | 프로토콜 변경 |

**forget gate clamp [ε, 1] 관련 주석:**
- epsilon 값 정당화를 위해 Phase 3B의 fig3_fg_at_stop.png 데이터(ES 종료 시점 forget gate 분포) 활용
- V2_fg1(f=1 고정) vs clamp 방식의 차이: clamp는 forget gate 기능을 유지하면서 gradient blocking만 제거 → 더 보수적인 개입
- 최적 ε는 "collapsed run의 forget gate 최대값보다 크고 normal run의 최솟값보다 작은 값"으로 결정

**스크립트:** `05_detector_validation.py` (파일명 유지, 내부 구조 재설계)  
**기간:** 2–3주 (기존 Phase 4 + Phase 5 합산)

---

### D4 — 외부 Audit 저널 및 논문 5편 확정

**결정일:** 2026-09-15  
**결정 내용:** 투고 1순위 저널을 **Engineering Applications of Artificial Intelligence (EAAI)** 로 확정하고, audit 대상 논문 5편(P1–P5)에 대해 10-seed 실험을 완료했다.

**투고 저널:** EAAI (1순위) — 직접 확인된 CMAPSS 논문 존재, scope 적합성 최고

**Audit 논문 5편:**
| ID | 논문 | 저널 | patience | MPC rate |
|----|------|------|:---:|:---:|
| P1 | Meng et al. 2023 (Bayesian GT) | Expert Systems | 15 | 0.80 |
| P2 | Qin et al. 2024 (STAR-LSTM) | **EAAI** | 20 | 0.60 |
| P3 | Zheng et al. 2017 (LSTM RUL) | IEEE ICPHM | 10 | 0.90 |
| P4 | Pre-2020 EAAI/ESWA 대표 (unclipped) | EAAI/ESWA | 15 | 0.90 |
| P5 | Elsherif et al. 2025 (CAELSTM) | Scientific Reports | ∞/25ep | 0.90 |

**핵심 결과:** A1+B2 조합에서 patience=10–20 범위 전체에서 MPC rate 60–90%. 외부 audit의 논증 강도 충분.

**결과 파일:** `Collapse_Study/Data_Analysis/Results/Phase5-Result.md`

---

## 확인 결과 (Phase 완료 후 업데이트)

### Phase 1A/1B/2/3/3B 핵심 수치 (결정 참고용)

| Phase | 핵심 발견 |
|-------|---------|
| Phase 1A | A1+B2+C2 조합 → MPC 8/10 (FD003, LSTM) |
| Phase 1B | bias_init=train_mean, MAE loss → 각각 MPC 0/10 |
| Phase 2 | PDR 실시간 알람 불가(specificity=0.063); ES 시점 소급 진단 완벽 분리 |
| Phase 3 | LSTM만 MPC; GRU/MLP/CNN1D = 0/10 (4개 dataset 전체) |
| Phase 3B | V2_fg1 (forget gate=1) → MPC 0/10; H_mech 가설 인과 확증 |

---

*이 파일은 연구 진행에 따라 지속 업데이트한다.*  
*새 결정이 생기면 확정 결정 또는 추가 결정 필요 항목에 추가할 것.*
