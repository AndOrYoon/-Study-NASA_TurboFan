# Final Editing Consistency Review

## 종합 판단

현재 원고는 중심 재프레임과 Related Work 구조는 안정적이지만, 아직 투고 가능한 clean version은 아니다. 문서에 22개의 검토 주석이 남아 있으며, 이를 독립적인 문제로 묶으면 핵심 차단 이슈는 네 가지다. 나머지는 동일 문제가 여러 절에 반복된 경우와 표현·편집상 수정 사항이다.

새로운 대규모 실험은 필요하지 않아 보인다. Table V-B 통계 재산출과 H4 correction family 확인이라는 두 가지 코드 확인을 끝낸 뒤, H3 validation split 설명과 반복 문구를 일괄 수정하면 최종 clean-up 단계로 넘어갈 수 있다. 현재 논문의 중심 재프레임 자체는 유지할 수 있다.

## 1. 투고 전 필수 해결 사항

| 우선순위 | 위치 | 남은 문제 | 권장 조치 |
|---|---|---|---|
| CRITICAL | Table V-B 및 unified 결과 전체 | One-sided 검정인데 Methods는 two-sided이며, 실제 비교는 4개인데 “six comparisons”로 BH 보정 | Two-sided·4개 데이터셋 기준으로 재산출 |
| CRITICAL | H3 Results/Discussion | Methods는 고정 validation split(seed=42)인데 본문은 per-run/random split으로 설명 | 고정 split 기준으로 전체 통일 |
| CRITICAL | Introduction | Reference [13]을 근거로 FD001 RMSE “below 13” 주장 | “low-to-mid teens”로 수정 |
| MAJOR/코드 확인 | H4 | 96-test BH family가 RMSE용인지 NASA용인지 불명확 | 통계 코드 기준으로 endpoint와 correction family 명시 |

## 2. Unified normalization 통계

현재 Table V-B는 다음과 같이 보고한다.

- `one-sided ranksums`
- FD001–FD003: `p_raw = 0.0045`, `p_BH = 0.009`
- “BH correction across six comparisons”

그러나 unified 분석은 문서상 `2 strategies × 4 datasets`이므로 데이터셋 수준 비교는 네 개다. 또한 §III.J의 H2 설명은 two-sided이다. Unified 분석 전에 단측 가설이 명시됐다는 근거가 없다면 two-sided로 통일하는 것이 가장 방어적이다.

완전한 rank separation에서 `scipy.stats.ranksums(..., alternative='two-sided')`를 적용하면 다음과 같다.

- FD001–FD003: `p_raw = 0.009023`
- 4개 비교 BH 보정: `p_BH = 0.012031`
- FD004: 약 `p_raw = p_BH = 0.458`

유의성 결론은 바뀌지 않는다. 따라서 통계값을 수정하고 이를 Abstract, Contribution (ii), §IV.B.2, Table V-B, §V.F, Conclusion에 일괄 반영하면 된다. 원래 seven-normalizer screening의 `p_BH = 0.009`는 별개의 분석이므로 변경하지 않는다.

## 3. H3 validation split 모순

Methods, Table III, Limitations에서는 H3를 다음과 같이 정의한다.

> deterministic engine-level shuffle, seed = 42, fixed across model seeds

그러나 다음 위치에는 아직 `random/per-run split` 또는 `training-split sensitivity`가 남아 있다.

- §III.D unified protocol 설명
- §IV.B.2
- §IV.C.2 첫 문단
- Fig. 7 caption
- §V.C
- §V.D

FD003의 M1 표준편차 8.74는 우선 다음과 같이 해석해야 한다.

> substantial training-seed instability under a fixed validation partition

GMM assignment가 seed마다 실제로 달라졌다는 로그가 없다면 “GMM labels produced different branch assignments across seeds”도 삭제해야 한다. Branch initialization, minibatch ordering, GMM initialization 또는 checkpoint dynamics 중 원인을 분리하지 못했다고 제한하는 것이 안전하다.

§III.J의 “non-overlapping engine pools”도 정확한 독립성 근거가 아니다. M0의 full pool은 M1의 cluster-specific pool을 구성하는 엔진을 포함한다. Independent Mann–Whitney test를 유지한다면 “M0 and M1 were not blocked on identical branch-level validation partitions”처럼 설계를 사실대로 설명하고, 실제 engine ID와 split 순서를 코드로 확인해야 한다.

## 4. Reference [13]

§II.A는 이미 “low-to-mid teens”로 적절히 수정됐지만, Introduction에는 다음 문장이 남아 있다.

> reported RMSE on FD001 has fallen ... to below 13 [13]

Elsherif et al.은 FD001 RMSE를 14.44, FD003 RMSE를 13.40으로 보고하므로 이 문장은 틀리다. 다음 문장으로 교체하는 것이 적절하다.

> As model architectures have evolved from recurrent baselines to attention-based and hybrid encoders, recent studies have reported FD001 RMSE values in the low-to-mid teens under heterogeneous evaluation protocols [12, 13, 25].

Source: https://www.nature.com/articles/s41598-025-09155-z

## 5. H4의 96-test family

`6 losses × 4 clips × 4 datasets = 96`은 하나의 outcome에 해당한다. 그런데 논문은 RMSE와 Mean NASA Penalty를 모두 추론적으로 해석한다. 실제 통계 스크립트가 다음 중 어느 방식인지 확인해야 한다.

1. RMSE만 96-test inferential family이고 NASA는 descriptive인지
2. RMSE와 NASA에 각각 별도의 96-test family를 적용했는지
3. 두 metric을 하나의 192-test family로 합쳤는지

`minimum p_BH = 0.176`이 어느 metric과 family에서 나온 값인지도 밝혀야 한다. 이 부분은 문구만으로 결정하지 않고 코드 출력에 맞춰 Methods, Table VII, Results, Discussion, Conclusion을 통일한다.

## 6. 추가로 남은 주요 표현 문제

1. Introduction의 H1 설명에서 `independent of backbone architecture`를 삭제한다. OLS 한 종류로 architecture independence를 입증할 수 없다.
2. §II.D와 §V.E의 Reference [37] 인용은 인용 논문의 결과와 본 원고의 해석을 분리한다.
3. `priority ordering`은 §V.A의 “strict hierarchy가 아니다”라는 설명과 충돌하므로 `attribution-oriented staged checklist`로 통일한다.
4. §V.D의 “training and test are identically distributed”는 `M3 applies the same information horizon and feature construction at training and test time`으로 수정한다.
5. Conclusion의 `evaluated in isolation`과 `resolved prerequisite throughout the remaining analyses`는 H4의 clipping–loss interaction과 충돌하므로 factor-specific experiments와 reference condition 표현으로 바꾼다.
6. Limitations의 “H2 and H4 used the LSTM backbone”에는 H3가 누락됐다. `H2–H4 used stacked-LSTM models, with different capacities`가 정확하다.
7. “all hypotheses except H1 used a common two-layer stacked LSTM backbone”은 동일 backbone으로 오해될 수 있다. 동일 architecture family를 사용했지만 H2와 H3/H4의 capacity가 달랐음을 명시한다.
8. §V.I의 “N-CMAPSS includes maintenance resets”는 Reference [14]에서 확인되지 않는다. 별도 근거가 없다면 `real-flight profiles and richer operating-condition variation` 정도로 제한한다. Reference [14] 자체는 Arias Chao et al. (2021)이 맞다.

N-CMAPSS source: https://www.mdpi.com/2306-5729/6/1/5

## 7. 반영이 잘 된 사항

- H1의 `all subsequent hypotheses` 오류는 H2/H3 고정, H4 reference condition으로 적절히 수정됐다.
- Unified run count는 `40 runs`로 정확하다.
- §II.E의 `shared backbone` 주장은 제거됐다.
- Gate confidence는 threshold나 routing accuracy로 해석하지 않고 기술 통계로 제한됐다.
- Mean NASA Penalty를 test-engine 수로 나눈 지표라는 설명이 추가됐다.
- Related Work의 구조와 연구 공백 연결은 현재 상당히 안정적이다.

## 8. 최종 편집 체크리스트

- Fig. 3의 `mean over 20 runs`를 `20 deterministic configurations`로 수정한다.
- §IV.B에 `B.1 Original Seven-Strategy Screening`을 추가한다.
- Mean NASA Penalty의 ratio/order 보존 설명은 `within the same dataset`으로 제한한다.
- `normalization/normalisation`, `labeling/labelling` 중 하나로 통일한다.
- Funding의 `No. XXXX`를 실제 과제번호로 교체하고 공식 영문 사업명을 확인한다.
- 실제 투고본에 Tables IV–VII, Table V-B, Figures 1–8이 삽입됐는지 확인한다.
- Target template에서 `Table V-B` 표기가 허용되는지 확인하고 필요하면 순차 번호 또는 V-A/V-B 체계로 정리한다.
- 모든 표·그림 번호를 확정한 뒤 cross-reference를 다시 검사한다.
- 모든 `<!-- REVIEW ... -->` 주석을 제거한 clean version을 생성한다.

## 최종 결론

현재 원고의 핵심 연구 프레임은 유지 가능하다. 남은 핵심 작업은 새로운 실험이 아니라 두 가지 통계·코드 확인과 그 결과의 문서 전체 전파다. 특히 Table V-B의 검정 방향·BH family와 H3 validation split 설명을 먼저 확정해야 한다. 이후 Reference [13], H4 multiplicity, priority-ordering 용어와 metric 명칭을 정리하면 투고 직전 수준으로 접근할 수 있다.
