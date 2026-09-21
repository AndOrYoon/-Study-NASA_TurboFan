# Research Plan 논리 검토서

**작성일:** 2026-09-11
**대상:** `Collapse_Study/Research_Plan.md` (Draft v0.1), `Collapse_Study/Manuscript/Sections/Literature_Review.md`, `New_Topic.md` I.1 / II.2
**검토 틀:** (1) 문제 정의의 학술적 가치 → (2) 방법론 적합성 → (3) 비통제 요인·대안 해석 → (4) 주장이 당면한 논리적 공격
**상태:** 실험 착수 전 사전 검토. 아래 "우선순위" 9개 항목이 180 run 실행 전 해소 대상.

---

## 0. 한 줄 결론

가장 큰 실존적 위험은 **"이건 현상(phenomenon)이 아니라 그냥 튜닝 안 된 baseline"** 이라는 공격이다.

- MSE → 평균 수렴은 교과서 내용이며 Bruna(2015), Mathieu(2016), Huang(2026) 본인이 명시적으로 인정.
- 세 트리거(고정 val split / warmup 없음 / 순수 MSE)는 모두 "하면 안 되는 관행"으로 이미 알려져 있음.
- 따라서 논문이 서 있으려면 새 현상을 주장할 게 아니라, **이 붕괴가 *조용히(silently)* 일어나 정상 수렴처럼 보이고, 그 결과 published baseline을 오염시킨다**는 것을 설계로 실증해야 한다.
- 현재 계획은 이 "silence"를 주장만 하고 실험 설계로 증명하지 않는다.

---

## 1. 학술적으로 가치 있는 문제를 어떻게 정의하고 있는가

### 1.1 잘 된 부분

- 실제 관측된 현상에 근거 (5 seeds, pred_std < 0.0002, pred ≈ 87 = 검증셋 RUL 평균). 추측이 아님.
- 인접 개념과의 구별을 Literature_Review에서 꼼꼼히 수행: mode collapse [P1], neural collapse [P17], Lee의 systematic mean bias [P7], vanishing gradient [P2]. "그거 그냥 X 아니냐"류 공격 상당수를 선제 차단.
- "연구 대상은 엔진이 아니라 실험 프로토콜의 신뢰성" (Research_Plan §12.1) 프레임은 정확하고 방어 가능.

### 1.2 공격받는 지점

| # | 문제 | 왜 위험한가 |
|---|------|------------|
| 1a | **현상의 novelty가 얇다.** 메커니즘은 기존 문헌이 전부 커버. 새로운 것은 "PHM 벤치마크에서 프로토콜 버그로 파국화"라는 훨씬 좁고 취약한 주장 | 리뷰어: "known failure mode + 세 가지 나쁜 관행 조합 = cautionary blog post이지 research contribution인가?" |
| 1b | **명명("MPC로 통합 명명") 자체가 약하다.** Huang은 이미 "mean collapse"라 부름. E[y] vs E[y\|x] 심각도 차이는 "신호가 더 약했다"이지 질적으로 새로운 메커니즘이 아님 | "왜 새 용어가 필요한가"에 대한 답이 궁색 |
| 1c | **"PHM 최초 보고"는 citation-search 주장이지 과학적 주장이 아님** | 이 논문의 진짜 계보는 *undertuned-baseline reproducibility* 문헌 — 추천시스템 "Are we really making progress? (Ferrari Dacrema et al.)", metric-learning "A Metric Learning Reality Check (Musgrave et al.)", "Are GANs Created Equal?" 등. **현재 Literature_Review에 이 계보가 통째로 빠져 있음.** 리뷰어가 "이미 알려진 문제"라고 칠 가장 위험한 누락 |
| 1d | **주장 크기와 연구 규모의 불일치.** 계획 스스로 §12.6에서 "실제 데이터에선 거의 안 일어남" 인정, RQ5(N-CMAPSS)는 optional로 강등. 그렇다면 임팩트는 "옛날 FD003 baseline 믿지 마라" 수준인데, 설계는 10주 · 180 run · full journal | short paper / letter / PHM conference 가 더 맞는 스코프. 외부 audit(항목 8) 추가 시에만 full journal 정당화 |
| 1e | **자기 참조적 기여.** 기여 5는 "우리가 낸 실수를 우리가 고쳤다" (n=1, FD003/H6). RESS 투고본은 이미 65.8% 주장 없이 수정 완료 → "벤치마크 오염"은 *우리 논문에선* 가정형 | 외부 논문 audit 없이는 confession이지 finding이 아님 |

---

## 2. 문제를 풀기 위한 방법론 제시가 적합한가

### 2.1 RQ1 (2³ factorial) — 세 factor가 깨끗한 직교 조작이 아님

- **"고정 seed=42 vs per-seed random"이 두 가지를 섞고 있음:**
  (i) val set이 훈련 seed 간 고정인지, (ii) *하필 그 20개 엔진 조합*이 병리적인지.
  계획 스스로 Ad-hoc_Analysis_Result.md §8.1 "확인 불가 사항"에서 "seed=42가 FD003에서 유독 불리한 분할인지" 명시. 구 연구는 훈련 로그가 없어 확인 불가였지만 **신규 연구는 split을 통제하므로 반드시 해소해야 함.**
  → 조치: 고정 split 여러 개(seed ∈ {0, 7, 42, 123, …})를 각각 테스트하고, 각 split의 val-set RUL 분포(평균·분산·전역 평균과의 거리)를 실측.
- **"MSE vs MSE+aux(λ=0.05)"는 loss가 아니라 아키텍처를 바꾼다.**
  aux-branch loss는 2-branch 구조(M2/M3)를 전제. 단일 branch M0에서 "MSE+aux"는 정의 불가 → (M0, aux=on) 셀이 존재하지 않음. **factorial이 고정 아키텍처 위에서 성립하지 않음.**
  → 조치: (a) M3에서 λ ∈ {0, 0.05} 로 토글 (이 경우 M0는 factorial 밖으로 분리), 또는 (b) branch 없이도 되는 architecture-neutral regularizer (output-variance penalty, output-entropy bonus, M0에 붙인 auxiliary head 등) 사용.
- **"MIN_EPOCHS 0 vs 30"이 이진 처리.** MIN_EPOCHS=5로 이미 막히면 "즉각 포획, 탈출 불가" 서사가 바뀜. → 소규모 sweep {0, 5, 10, 30} 필요 (Phase 4가 일부 다루나 Phase 1에서 분리 안 됨).
- **수정 프로토콜은 사실 4가지를 바꿨다** (val split, MIN_EPOCHS, MAX_EPOCHS 100→300, eval clip / test-RUL clip — Ad-hoc §12.2 표). Phase 1은 3개만 factorize하고 MAX_EPOCHS와 clipping을 뺌.
  → clipping이 metric에 영향이 컸던 정황(H5 N1+M0 FD003 std=12.86, Ad-hoc §6.1 관찰 1)이 있으므로 **"상수 출력 붕괴"와 "unclipped 예측이 발산해서 RMSE가 튄 것"을 분리 정의**해야 함.

### 2.2 RQ2 (데이터셋 취약성 프로파일) — rate 추정에 표본이 절대 부족

- 10 seeds로 collapse rate 추정 → 참값이 20%라면 2/10의 95% CI가 대략 3%–56%. **10 seeds로 FD001–FD004 취약성 순위를 매길 수 없음.** CI가 전부 겹침.
  → headline 비교엔 seed ≥ 30–50 필요. 불가하면 RQ2를 정성적("붕괴 관측됨 / 안 됨")으로 재프레임.
- **"Collapse Risk Score"(New_Topic II 아이디어)는 n=4 데이터셋으로 fit도 held-out validate도 불가.** deliverable에서 제거하거나 "가설 생성 수준"으로 강등.
- **FD001 falsification 위험을 정면에 둘 것.** FD001(1 op / 1 fault)이 가장 균질한데, 만약 FD001이 *안* 무너지면 "균질성 → 붕괴"(G5) 서사가 깨짐. 이걸 미결사항 D1에 묻어두지 말고 핵심 검증 지점으로 명시.

### 2.3 RQ3 (조기 감지 지표) — 순환 논리

- Phase 3는 pred_std < θ 임계값을, "collapsed = pred_std < 1.0" (Research_Plan §6)로 정의된 ground truth에 대해 캘리브레이션.
  → **pred_std로 정의한 라벨에 pred_std 탐지기를 ROC 하는 것은 거의 항진명제.**
  → 조치: 독립적 붕괴 정의 도입 — 제대로 학습된 reference 대비 RMSE 비율, 또는 "예측이 test RUL 분산의 X% 미만 설명(R² < threshold)". 그다음 pred_std가 *그것*을 잡는지 sensitivity/specificity 측정.
- "예측값이 다 똑같으면 모델이 깨진 것"은 유능한 실무자면 다 하는 sanity check. **비자명한 케이스는 부분 붕괴(FD004 bimodal seeds — Ad-hoc §13.2, seed 0·3 RMSE ~22-24 vs seed 1·2 ~15)** — 여기를 리드해야 논문값이 생김.

### 2.4 RQ4 (방어 전략) — RQ1/RQ3과 얽혀 있고, best practice 재진술 위험

- Phase 1에서 per-seed random split만으로 막히면 RQ4 답은 "Phase 1에서 이미 나옴"이고 Phase 4의 60 run은 재확인에 그침.
- "pred_std 모니터링 + 재시작"은 prevention이 아니라 detection+retry이며 RQ3이 작동한다는 전제에 의존.
- 살아남으려면 **반직관적 비대칭**을 보여야 함 (예: warmup은 안 듣는데 split 무작위화는 듣는다, 혹은 그 반대 / 가장 싼 fix가 반직관적).

### 2.5 RQ6 (Huang 경계 조건) — 방법론적으로 가장 위험한 커밋

- **arXiv preprint를 "peer-reviewed 반론"으로 반박하는 것은 취약한 기여.** Huang이 개정하거나 무명 논문이면 기여가 증발.
- 실질적으로 Huang을 **반박하지 않음.** Huang의 주장은 *truly ambiguous* 입력(같은 x, 다른 y)에 대한 것. 이 연구의 주장은 "FD003 두 fault mode는 올바른 프로토콜이면 센서 궤적으로 분리 가능" = "Huang의 전제가 여기 적용되지 않음". 이건 "정리가 틀렸다"가 아니라 "여기는 그 세팅이 아니다".
  → **진짜로 multimodal-same-x인데 MSE가 잘 되는 사례를 한 번도 제시하지 않음.**
- `arXiv:2608.25467` — ID 실재 여부, 인용 내용의 정확성, withdraw 여부를 먼저 확인.
- → 조치: 기여 6을 "contribution"에서 "discussion — 아키텍처 처방이 필요한/불필요한 경계"로 강등.

### 2.6 빠진 축

- optimizer / LR / batch 고정 (Adam 1e-3, batch 256). trivial-solution basin의 매력도는 LR 의존적. "초반 LR이 너무 높아 fit 전에 평균으로 밀린 것 아니냐"는 질문에 답할 warmup-LR arm이 없음.

---

## 3. 비통제 요인의 개입 및 다른 해석의 여지

리뷰어가 제기할 대안 해석 (강한 순):

- **(A) "그냥 undertuned baseline"** — 반박하려면 *silence*를 실증해야 함: val-loss 곡선이 건강해 보인다 / 표준 early-stopping이 "정상" 발화한다 / 흔한 레시피를 따르는 실무자면 그대로 ship한다. 이 실증이 약하면 논문은 "훈련 제대로 해라"로 붕괴.
- **(B) seed=42 특이성** — 전체 효과가 그냥 운 나쁜 20-엔진 holdout(RUL이 전역 평균 근처에 몰린 분포)일 수 있음. 다수 고정 split + val-set RUL 분포 실측으로 배제. (신규 연구는 통제 가능하므로 반드시 측정.)
- **(C) FD003 고유 아티팩트** — "합성 데이터 균질성"(G5)이 n=1에 의존. RQ2가 이걸 다뤄야 하는데 표본 부족(§2.2).
- **(D) M3 생존의 진짜 원인** — 계획은 "aux-branch gradient highway"라 함. 대안 해석:
  (i) M3는 파라미터 2×, init scale / 학습 dynamics가 다름;
  (ii) **GatingNet이 입력 의존 변동을 구조적으로 출력에 주입해서 순수 상수를 못 냄.**
  (ii)가 맞으면 "aux loss가 stabilizer"(New_Topic II 아이디어 2)는 **틀린 메커니즘 서사.**
  → 깨끗한 ablation: **M3 with λ=0 (aux loss 제거).** λ=0에서도 M3가 안 무너지면 aux-loss 설명은 폐기.
- **(E) "protocol bug(단수, 고칠 수 있는)"라는 전제 자체가 factorial이 실제로 분리해내야 성립.** Phase 1이 §2의 결함들 때문에 지금은 못 함 → 인과 주장은 아직 assertion.
- **(F) "marginal mean E[y]≈87" 정밀도** — Ad-hoc §8.1은 "*검증셋* 엔진 평균 RUL이 ~87"이라고 씀. 그렇다면 모델은 E[y | val]로 수렴한 것(val MSE가 early-stopping 신호이므로)이지 marginal E[y]가 아님. 이 뉘앙스가 Literature_Review §1.1의 "Huang보다 한 단계 더 심각(E[y] 수준)" 프레이밍을 약화시킴.
  → train-label / val-label / test-RUL 중 어느 평균인지 정확히 특정.

---

## 4. 가설 검증을 통한 본 연구 주장이 당면한 논리적 공격

기여점별 [주장 → 최강 반론 → 살아남는 조건]:

| 기여 | 최강 반론 | 살아남는 조건 |
|------|-----------|--------------|
| **C1** MPC 메커니즘 최초 체계 분석 | 메커니즘 known; "in RUL + systematic"은 incremental; 실체는 나쁜 관행 노브 2³ sweep | factorial이 *비자명한* 트리거 상호작용을 깨끗이 식별 **AND** 표준 관행 하에서 *silent*임을 실증 |
| **C2** 데이터셋 취약성 프로파일 / Risk Score | n=4, 10 seeds, CI 전부 겹침, score의 held-out 검증 없음 | "Risk Score" 삭제; "FD00x는 붕괴, FD00y는 아님, intra-mode 균질성 가설과 일치"로 재프레임 + seed ≥ 30 + CI 명시 |
| **C3** 조기 감지 프로토콜 | 순환(탐지기 = 라벨 정의); "예측값 다 같으면 broken"은 자명 | 독립 ground-truth + *부분* 붕괴(육안·val-loss로 안 잡히는 것)를 잡는다는 것을 리드 |
| **C4** 최소 비용 방어(단일 조치) | "고정 split 쓰지 마라 / warmup 써라" = 이미 표준 권고 | 반직관적 비대칭을 보여야 |
| **C5** 재현성 경고 (65.8%→0%) | n=1, 자기 논문, 투고 전 발견되어 "오염"은 우리에겐 가정형 | **외부 논문 audit** — FD003 baseline RMSE 30–45로 보고한 published 논문 3–5편을 대상으로, 그들의 서술된 프로토콜에서 붕괴 개연성을 논증하고 RMSE≈13 대비 상대 개선폭을 재계산. confession → finding 전환, **논문의 최고 버전** |
| **C6** Huang 경계 조건 / peer-reviewed 반론 | arXiv 반박은 취약; 실제론 반박 안 함(전제 부적용); 진짜 multimodal-same-x 케이스 미제시 | "contribution" → "discussion(아키텍처 처방의 경계)"로 강등; preprint 실재·인용 정확성 확인 |

### 4.1 교차 공격 — "왜 합성 데이터 신뢰성이 중요한가"

- Research_Plan §12가 잘 선제 방어함("연구 방법론의 신뢰성"). 다만 부모 논문이 TII에서 합성 데이터 fit 문제로 밀렸는데, *주제 전체가 "합성 데이터 + 나쁜 프로토콜의 아티팩트"* 인 논문은 더 노출됨.
- §12.6의 솔직한 한계("배포 위험 아님, 방법론 위험")를 **서론 초반에** 명시할 것 — 대신 이것이 논문의 상한선을 정한다는 점을 인지.

### 4.2 자기 기준 적용 (엄밀성에 *관한* 논문의 의무)

- collapse 정의·임계값을 pre-register.
- headline 비교는 seed ≥ 20–30.
- p값이 아니라 CI를 보고.
- 모든 조건의 val-set RUL 분포를 보고.
- 현재 계획의 N=5~10은 "동등성 확인"이 아니라 "검출력 부족"으로만 읽힘 (Ad-hoc §10, §13.3과 동일한 한계).

---

## 5. 180 run 실행 전 해소 우선순위

| # | 항목 | 대응 RQ / 기여 |
|---|------|----------------|
| 1 | **포지셔닝 전환**: "새 현상(MPC)" → "CMAPSS RUL의 undertuned-baseline / silent-failure audit". *undertuned-baseline reproducibility* 계보에 명시적으로 편입. MPC는 서술적 라벨로만 유지 | C1, 1c |
| 2 | **"Silence"를 핵심 실증 주장으로** 삼고 설계에 반영: 건강해 보이는 val 곡선, 표준 early-stopping 발화, 실무자가 그대로 ship함 | C1, 대안 해석 (A) |
| 3 | **Factorial 수리**: 고정 split 여러 개 + val-set RUL 분포 실측 / aux-loss는 M3 λ∈{0,0.05}로 or architecture-neutral regularizer / MIN_EPOCHS sweep {0,5,10,30} / MAX_EPOCHS·clipping을 설계에 넣거나 배제 이유 명시 / LR-warmup arm | RQ1, (B), (D), (E), §2.6 |
| 4 | **붕괴 정의를 RMSE 크기가 아닌 예측 분산·variance-explained(R²)로** 재정의. "상수 출력 붕괴" vs "unclipped 발산" 분리 | RQ1, RQ3 |
| 5 | **RQ2**: seed ≥ 30/dataset + CI, 불가하면 정성적으로 강등. "Collapse Risk Score" 삭제. FD001 falsification 위험 전면화 | RQ2, C2 |
| 6 | **RQ3**: 독립 붕괴 ground-truth, 부분 붕괴(FD004 bimodal) 리드 | RQ3, C3 |
| 7 | **RQ6**: Huang을 contribution에서 discussion으로 강등, preprint 실재·인용 정확성 확인 | RQ6, C6 |
| 8 | **외부 audit 추가**(신규 작업): published FD003 고-RMSE 논문 3–5편 분석. C5를 finding으로 전환 + full-length 정당화 | C5 |
| 9 | **스코프/venue 재결정**: 8번 없이는 short/methods paper 또는 PHM conference. 8번 추가 시 full journal 정당 | 1d |

---

## 6. 참고 — 검토에 사용한 소스

- `Collapse_Study/Research_Plan.md` (Draft v0.1)
- `Collapse_Study/Manuscript/Sections/Literature_Review.md`
- `Collapse_Study/CLAUDE.md`
- `New_Topic.md` §I.1 (합성 데이터 균질성 → 붕괴), §II.2 (aux loss as stabilizer)
- `Data_Analysis/Ad-hoc_Analysis/Ad-hoc_Analysis_Result.md` §8 (붕괴 원인 3조건), §12–15 (수정 프로토콜, C-Full 결과, op-condition 동등성)
- `CLAUDE.md` (상위) — H6 corrected 결과표, "왜 원래 65.8%가 틀렸나" 노트
