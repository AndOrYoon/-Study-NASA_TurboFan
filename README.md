# NASA CMAPSS TurboFan RUL 예측 연구

NASA C-MAPSS 데이터셋을 이용한 터보팬 엔진 잔여 수명(RUL, Remaining Useful Life) 예측 딥러닝 연구.  
4가지 설계 인자(RUL 클리핑, 정규화, 고장 모드 아키텍처, 손실 함수)를 cross-dataset controlled ablation으로 검증.  
**투고 목표: IEEE Transactions on Industrial Informatics (TII)**

---

## 연구 개요
**데이터셋:** NASA_TurboFan
**데이터셋:** (폴더) C-BMAD_PY313

**데이터셋:** NASA CMAPSS (FD001–FD004)
- 총 707개 엔진, 21개 센서, 공간-분리 `.txt` 포맷
- 운전 조건: 1가지(FD001/FD003) vs 6가지(FD002/FD004)
- 고장 모드: HPC 단일(FD001/FD002) vs HPC+Fan 복합(FD003/FD004)

**공통 백본:** Stacked LSTM (LSTM1(64)→LSTM2(64)→FC(64→32→1)), window=30, batch=256

---

## 진행 상황

| 단계 | 상태 | 산출물 |
|------|------|--------|
| EDA | ✅ 완료 | `Dataset/EDA_Report.MD`, `Dataset/Figure/` (fig01–fig12) |
| 선행 연구 조사 | ✅ 완료 | `Hypothesis/Hypothesis_PossibleValidation.MD` |
| H2: RUL 클리핑 실험 | ✅ 완료 | `Data_Analysis/Results/H2_clipping/` |
| H5: 정규화 전략 실험 | ✅ 완료 | `Data_Analysis/Results/H5_normalization/` |
| H6: 고장 모드 아키텍처 실험 | ✅ 완료 | `Data_Analysis/Results/H6_fault_mode/` |
| H7: 손실 함수 실험 | ✅ 완료 | `Data_Analysis/Results/H7_loss_function/` |
| 논문 작성 (전 섹션) | ✅ 완료 | `Manuscript/Sections/` + `manuscript_full_text.md` |
| 제출용 DOCX 생성 | ✅ 완료 | `manuscript_draft(TII Submission).docx` |
| 커버레터 작성 | ✅ 완료 | `Cover_letter(IEEE_TII).docx` |
| **IEEE TII 제출** | 🔲 미착수 | — |

---

## 실험 결과 요약

### H2: RUL 클리핑 (Ridge regression, 20 runs)

| clip | FD001 RMSE | FD002 RMSE | FD003 RMSE | FD004 RMSE |
|------|-----------|-----------|-----------|-----------|
| 75 | 33.65 | 47.37 | 32.48 | 50.83 |
| 100 | 24.51 | 37.67 | 23.00 | 40.37 |
| **125** | **21.90** | **32.39** | **21.62** | **34.61** |
| 130 | 22.06 | 31.83 | 22.24 | 34.04 |
| None | 31.90 | 33.05 | 56.09 | 46.99 |

**결론 (부분 채택):** clip=125가 전 데이터셋에서 최적 또는 통계적 동등 최적. clip=None은 FD003 NASA Score 4,014,723 (재앙적).

---

### H5: 정규화 전략 (LSTM, 5 seeds × 7 strategies × 4 datasets = 140 runs)

- **Fleet MinMax (N1)** — FD001/FD002/FD004에서 최우수: RMSE ≈ 14.1–14.6
- Per-unit (N3–N6) — 모든 데이터셋에서 N1 대비 통계적 열세 (BH p<0.05)
- RevIN (N7) — FD001에서 경쟁적(14.92), FD002/FD004에서 큰 폭 열세(18.2+)
- FD003 N1 std=12.86 → 이상 분산은 정규화 실패가 아닌 **잠재 고장 모드 이종성 신호**

**결론 (기각):** Fleet MinMax가 명확한 우승자. 인스턴스 수준 적응은 CMAPSS 같은 동질 함대에서 역효과.

---

### H6: 고장 모드 아키텍처 (LSTM, 5 seeds, FD003/FD004)

| 모델 | FD003 RMSE±std | FD003 NASA | FD004 RMSE±std | FD004 NASA |
|------|---------------|-----------|---------------|-----------|
| M0 (baseline) | 43.23±0.18 | 34,339 | 28.05±1.74 | 10,586 |
| M1 (hard routing) | 32.45±11.37 | 27,098 | 49.20±7.17 | 132,003 |
| M2 (soft gating) | 26.16±14.81 | 20,881 | 30.71±0.73 | 50,715 |
| **M3 (attention gate)** | **14.78±1.32** | **425** | **28.33±1.03** | **13,229** |

**결론 (채택 — M3):** M3 FD003 RMSE −65.8%, NASA Score −98.8% vs M0. 초기 K=10 사이클만으로 라우팅 가능 → 커미셔닝 시점 적용 가능. M1은 FD004 테스트 시 클러스터 붕괴(1:247) 실패.

---

### H7: 손실 함수 (LSTM, 5 seeds × 7 losses × 4 clips × 4 datasets = 560 runs)

- BH-FDR 보정 후 MSE(L1) 대비 통계적으로 유의하게 우수한 손실 함수 없음 (96개 비교)
- clip=None 조건에서만 Pinball(L6, τ=0.25)이 FD003 NASA Score를 9,614,000 → 3,124로 감소
- 전반적 분산의 주 원인은 손실 함수가 아닌 **RUL 클리핑**

**결론 (기각):** 커스텀 손실 함수는 통계적으로 유의한 개선을 제공하지 않음. 클리핑이 손실 함수 선택을 지배.

---

## 3계층 설계 계층구조 (핵심 기여)

```
TIER 1 — 레이블 엔지니어링  →  clip = 125 (NASA Score 최대 306,000배 차이)
TIER 2 — 아키텍처           →  FD003/FD004: M3 Attention Gate (−65.8% RMSE)
TIER 3 — 손실 함수          →  MSE 기본값 (통계적 이득 없음)
```

각 티어의 달성 가능한 레버리지가 다음 티어를 질적으로 초과.

---

## 폴더 구조

```
C:\BMAD_PY313\
├── Dataset/                        ← 원본 CMAPSS .txt + EDA 결과물
│   ├── train/test_FD00{1-4}.txt
│   ├── RUL_FD00{1-4}.txt
│   ├── EDA_Report.MD
│   └── Figure/                     ← EDA 플롯 fig01–fig12
├── Hypothesis/                     ← 선행 연구 갭 분석
├── Data_Analysis/
│   ├── Analysis_Plan.md            ← 실험 설계 문서
│   ├── 주요이슈_및_의사결정.md      ← 확정된 설계 결정 (변경 전 반드시 확인)
│   ├── Code/
│   │   ├── shared/op_condition_utils.py  ← K-means residualization (H5/H6 공용)
│   │   ├── H2_clipping/            ← 01–05 스크립트
│   │   ├── H5_normalization/       ← 01–06 스크립트
│   │   ├── H6_fault_mode/          ← phase1/phase2/phase3
│   │   └── H7_loss_function/       ← 00–08 + run_all_h7.py
│   └── Results/                    ← CSV, 그림, 모델 체크포인트
└── Manuscript/
    ├── Sections/                   ← Introduction, Methodology, Results,
    │                                  Discussion_Implication, Conclusion, Abstract
    ├── References.md
    ├── Full-Text_Manuscript/
    │   ├── manuscript_full_text.md              ← 완성 원고 (마크다운)
    │   ├── manuscript_draft(TII Submission).docx ← 제출용 DOCX (더블블라인드)
    │   └── build_tii_submission.py              ← MD → DOCX 변환 스크립트
    ├── Cover_Letter/
    │   ├── Cover_Letter_draft.md   ← 커버레터 v3.0 (TII 특화)
    │   ├── Cover_letter(IEEE_TII).docx
    │   └── convert_cover_letter.py
    ├── Figures/                    ← 최종 Figure 1–8
    └── Tables/                     ← Table 1–5 CSV
```

---

## 제출 전 잔여 항목

| # | 항목 | 비고 |
|---|------|------|
| 🚨 A | ETRI 기관 이메일로 IEEE Author Portal 계정 등록 | Gmail 미허용 |
| B | Word에서 10페이지 이내 확인 (IEEE 2단 포맷) | — |
| C | 커버레터 `[Author Name]`, `[Position/Title]` 기입 | — |
| D | References [30]–[36] 저자 정보 보완 | 현재 "(Authors not retrieved)" |
| E | COI(이해충돌) 선언 준비 | IEEE 양식 |
| F | 데이터 가용성 선언 (NASA CMAPSS 공개 URL 추가) | — |
| G | 현 TII EiC 이름 확인 → 커버레터 수신인 수정 | ieee-ies.org 확인 |

---

## 환경

```powershell
# 가상환경 활성화
NASA_TurboFan\Scripts\activate

# 실험 재실행 예시
python Data_Analysis\Code\H7_loss_function\run_all_h7.py

# 제출용 DOCX 재생성
python Manuscript\Full-Text_Manuscript\build_tii_submission.py

# 커버레터 DOCX 재생성
python Manuscript\Cover_Letter\convert_cover_letter.py
```

- Python 3.13 (TensorFlow 미지원 → PyTorch 사용)
- 가상환경: `C:\BMAD_PY313\NASA_TurboFan\`

---

*참고: A. Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08, 2008.*
