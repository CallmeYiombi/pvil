---
paths:
  - "src/pvil/eval/**"
  - "scripts/evaluate*.py"
  - "scripts/embed*.py"
  - "configs/**"
---

# 평가 코드 규칙

## 테스트 배치 가드

- 데이터 로더와 평가 함수는 `split="test"`에 접근할 때 `allow_test=True` 인자가 없으면 예외를 던지도록 구현한다. 이 인자는 최종 평가 스크립트에서만 켠다.
- 모든 결과 파일에 `protocol`(P-orig | P-strict | P-LOBO), `fold`, `seed`, `git_hash`, `config_path`를 기록한다.
- 테스트 배치 평가를 실행하면 날짜와 run 목록을 `docs/decisions.md`에 남긴다.

## 고정 정의 (바꾸려면 사용자 승인 + decisions.md 기록)

- 후처리: L2 → well 평균 → 대조군에 fit한 ZCA-cor sphering → PCA 50 → Harmony(`harmonypy==0.2.0`, θ = 2, λ = 1, σ = 0.1, max_iter_harmony = 100, 나머지 0.2.0 기본값. 수렴 반복 횟수와 장치 기록). 전후 모두 저장. 구현은 `src/pvil/eval/harmony.py`.
- scIB Batch: iLISI, kBET, BRAS, graph connectivity, PCR comparison 5개 비가중 평균. Harmony 전은 PCR = 0. 개별 지표와 PCR 제외 4개 평균도 저장.
- 대조군 well은 scIB를 제외한 모든 평가에서 제외.
- Cross-batch recall: 쿼리 = held-out 배치의 비대조군 well, 갤러리 = 다른 배치의 비대조군 well, 코사인, macro R@1…10.
- Replicate mAP: copairs. positive = 다른 plate의 같은 perturbation, negative = 같은 plate의 처리군–대조군, 같은 plate 처리군 쌍 제외.
- CP feature prediction: ridge α = 1, 표준화는 학습 세트에만 fit, median Pearson r.
- 세부 사양은 `docs/tasks/P0_reproduction.md` §5를 따른다.

## 통계와 보고

- 3 seeds 평균 ± 표준편차. 계층적 bootstrap(seed → 쿼리 perturbation, 1,000회) 95% CI.
- 베이스라인 비교: perturbation별 R@10에 Wilcoxon signed-rank + Holm 보정.
- 저장 위치: `results/<exp>/<method>/<dataset>/seed<k>/metrics.json`, `per_perturbation.csv`.
- 요약표는 스크립트로 생성한다. 수치를 손으로 옮겨 적지 않는다.
