# PVIL 연구 계획 (저장소 기준본)

- 작성: 2026-10-14
- 원본: Claude Docs "연구 계획서: Paired-View Shortcut-Aware Representation Learning for Cell Painting"
- 이 파일이 저장소의 기준본이다. 내용을 바꿀 때는 `docs/decisions.md`에 날짜와 사유를 남긴다.
- 출처 표기: **[확인]** 원문 직접 확인 · **[추정]** 명시한 가정에서 계산한 값 · **[미확인]** 착수 전 확인 필요

---

## 1. 개요와 최종 목표

### 1.1 제목 후보

1. (주안) **Not All Context Is a Shortcut: Paired-View Interventional Learning for Cross-Batch Biological Retrieval in Cell Painting**
2. Keep the Neighbors, Drop the Batch: Learning Transferable Context from Paired-View Interventions in Cell Painting
3. From Diagnosis to Training: Paired-View Interventions for Shortcut-Aware Cell Painting Representations

방법 가칭은 PVIL (Paired-View Interventional Learning)이다. 제출 전 이름 중복 검색이 필요하다 [미확인].

### 1.2 연구 질문

- **Main RQ:** Can a learning objective that exploits paired-view interventions separate *batch-transferable* context from *batch-specific* shortcuts, thereby improving cross-batch biological retrieval while preserving performance across all four CP-BG-Bench protocols?
- **RQ1 (진단):** crop 맥락의 어떤 성분(이웃 세포 표현형, 국소 밀도, 배경 픽셀 통계)이 배치를 넘어 전이되는가?
- **RQ2 (방법):** paired view를 학습 신호로 쓰면 전이 맥락은 남기고 배치 특이 신호만 억제할 수 있는가?
- **RQ3 (일반화):** 그 효과가 고정된 held-out 배치 하나를 넘어 여러 배치와 다른 세포주에서도 유지되는가?

### 1.3 가설과 판정 규칙 (사전 등록)

| ID | 가설 | 실험 | 지지 조건 |
| --- | --- | --- | --- |
| H1 | 맥락 = 이웃 세포 표현형(전이) + 국소 밀도(부분 전이) + 배경 픽셀 통계(배치 특이) | E2 | 3개 데이터셋 중 2개 이상에서 N view의 cross-batch R@10이 S보다 유의하게 높음 |
| H2 | PVIL은 동일 백본·데이터의 통제 비교 대비 cross-batch R@10을 3개 데이터셋 모두에서 높인다 | E1 | PVIL − A1-C와 PVIL − (B·C·D 중 최강)의 95% CI가 3개 데이터셋 모두에서 0 초과 |
| H3 | 학습 목적함수가 within/across retention 구조를 바꾼다. CP-BG-Bench의 "배경 이득은 인코더가 아니라 실험 설계가 결정한다"는 결론에 대한 반론이다 | E1, E3 | 같은 PVIL 모델에 S와 C를 넣어 잰 ρ_a가 두 기준보다 모두 높음. 기준 1: CP-BG-Bench 방식 ρ_a(S로 학습한 모델 ÷ C로 학습한 모델)의 3개 백본 최댓값. 기준 2: A1-C 모델에 S와 C를 넣어 잰 ρ_a. 두 정의는 논문에서 구분 표기 |
| H4 | view는 학습에만 쓰고 추론은 C view만으로 충분하다(테스트 시 segmentation 불필요) | E8 | C-only 추론이 E1의 PVIL과 비열등이고 A1-C보다 높음 |

**2차 주장 (사전 등록):** C→S 격차가 큰 데이터셋에서는 우위, 격차가 작은 데이터셋(JUMP-CP)에서는 비열등. H2가 기각돼도 이 주장은 별도로 판정해 보고한다.

### 1.4 기여 (목표)

1. **조작적 정의와 진단:** within/across retention으로 shortcut과 transferable context를 정의하고, N view로 맥락을 세 성분으로 분해한다.
2. **방법:** PVIL. 임베딩을 세포 성분 z_c와 맥락 성분 z_x로 나누고, view 일관성과 배치 조건부 불변성을 성분별로 건다.
3. **평가:** CP-BG-Bench 4개 프로토콜 전부, leave-one-batch-out, 외부 검증. 코드와 체크포인트를 공개한다.
4. **분석:** 모델이 남긴 맥락과 지운 맥락을 attribution과 view별 성능으로 보인다.

### 1.5 성공 기준

| 등급 | 기준 |
| --- | --- |
| 필수 | 동일 백본 통제 비교(A·B·C 그룹)에서 cross-batch R@10이 3/3 데이터셋 모두 유의하게 우세 (3 seeds, 계층적 bootstrap 95% CI) |
| 목표 | 데이터셋별로 CP-BG-Bench 36개 구성 중 최고 성능보다 우세. 동시에 replicate mAP, CP feature prediction, scIB Bio에서 비열등 (δ 사전 등록) |
| 확장 | PVIL을 OpenPhenom, SubCell 백본에도 적용해 각각 개선 (특정 백본 전용이 아닌 범용 목적함수) |

기전 지표: 임베딩의 배치 예측도(linear probe) 감소, S/C cross-batch retention 증가.

---

## 2. 선행연구 핵심: CP-BG-Bench (arXiv 2609.12761) [확인]

### 2.1 설정 요약

- **View:** Cellpose-SAM 분할, 세포 중심 150×150 px crop → 224×224. S는 DNA 채널을 핵 마스크 밖에서, 나머지 채널을 세포 마스크 밖에서 0으로 만든다. CD/SD는 네 모서리에 22×22 px 패치를 그리고 강도로 FOV 세포 수(min-max)를 나타낸다.
- **인코더:** DINOv3 ViT-B/16, OpenPhenom, SubCell. 백본 고정 + LoRA, 128차원 공유 projection.
- **학습:** 이미지 임베딩 ↔ perturbation 임베딩(ECFP4 / ESM2) 대칭 multi-positive contrastive. 미니배치 512 = 64 perturbation × 8, imaging batch 간 층화 샘플링(cross-batch positive가 이미 포함됨).
- **후처리:** L2 → well 평균 → 대조군 기준 ZCA-cor sphering → PCA 50 → Harmony.
- **Held-out:** JUMP-CP source_4, RxRx1 HEPG2-07, RxRx3-core plate 4. 같은 배치를 early stopping에도 사용했다. 구성당 1회 실행.
- **공개:** 데이터, 재구성 파이프라인, 체크포인트 36개, 평가 스위트를 공개 예정이나 2026-10-14 기준 미공개("링크는 추후 개정판에 추가").
- 전체 재현 사양은 `docs/tasks/P0_reproduction.md`에 있다.

### 2.2 설계에 영향을 주는 결과 (post-Harmony)

| 결과 | 수치 | 함의 |
| --- | --- | --- |
| cross-batch R@10은 9개 인코더×데이터셋 조합 전부에서 C > S | RxRx1 DINO 0.742 vs 0.322 · RxRx3-core SubCell 0.313 vs 0.100 · JUMP-CP SubCell 0.170 vs 0.140 | 맥락에 배치를 넘어 전이되는 신호가 있다. 배경 제거는 해법이 아니다 |
| 밀도 증강은 within-study 격차의 84%를 회복하지만 cross-batch 격차는 8%만 회복 | RxRx3-core SubCell R@10: S 0.100 → SD 0.117 (C 0.313) | 명시적 밀도는 within-study shortcut의 1순위 후보 |
| CP feature prediction은 JUMP-CP·RxRx3-core에서 S > C | RxRx3-core OpenPhenom: S 0.432 vs C 0.311 | C의 추가 신호는 고전 형태 특징이 아니다. 형태 보존을 보는 비열등성 지표로 쓴다 |
| C→S replicate mAP 격차는 데이터셋마다 다르고 인코더 간에는 비슷 | JUMP-CP 0.03 ~ RxRx1 0.36 | H3의 반박 대상 |
| RxRx3-core는 Harmony 전후 recall 차이가 0.003 이내 | 사후 보정 효과가 사실상 없음 | 학습 단계 개입의 여지가 가장 크다. 방법 개발을 여기서 시작한다 |

### 2.3 해석상 주의

- 원문 2.2절 제목과 결론 문장은 배경 신호가 배치를 넘어 전이되지 않는다고 요약한다. 그러나 같은 절의 표에서 cross-batch R@10은 9/9 조합 모두 C가 S보다 높다. 초록의 표현("segmentation 후에도 남는 within-study 신호가 대부분 전이되지 않는다")이 수치와 맞다. 따라서 "일부 맥락 성분(밀도 등)은 전이되지 않고 다른 성분은 전이된다"로 읽는다. 논문에서는 원문 수치를 정확히 인용한다.
- 저자들은 RxRx1·RxRx3-core가 단일 시설 데이터여서 plate layout, 밀도, 이웃 효과를 통해 잔여 변동이 perturbation과 상관될 수 있다고 지적했다. JUMP-CP 화합물 plate는 10개 source에 걸쳐 layout이 무작위화되어 있다.

### 2.4 저자가 남긴 열린 질문 = 본 연구의 출발점

원문 Discussion은 "within-study 전용 신호를 벌점화하는 학습 목적함수로 cross-batch 격차를 줄일 수 있는가"를 열린 질문으로 남겼다. 예로 든 두 방법(cross-batch positive, 배치 상관 맥락에 대한 adversarial 정규화)은 각각 베이스라인 B3, B4로 포함하고, PVIL은 이 둘을 넘어서야 한다.

본 연구가 해소하는 원문의 한계:

- 구성당 단일 실행 → 3 seeds와 bootstrap 신뢰구간
- held-out 배치로 early stopping → P-strict 병행
- 단일 batch correction(Harmony) → Harmony 전후, sphering만, 배치별 정규화(PCA-CS) 병기

### 2.5 관련 연구 지형과 차별점

- Cell Painting 배치 강건 표현학습: CWA-MSN(다른 well의 같은 perturbation 정렬), ABRA(적대적 배치 표현 증강), MICON(화합물을 처치로 모델링하는 perturbation-aware contrastive)
- 일반 domain generalization·shortcut: DANN, Deep CORAL, VREx, GroupDRO, IRM, 배경 증강
- 차별점: 위 방법들은 배치 라벨만 쓴다. 같은 세포에서 맥락만 바꾼 paired view를 **개입 감독 신호**로 쓰는 방법은 없다.

---

## 3. 문제 정의

### 3.1 표기와 작업 가정

- 단일 세포 tile x, well w, imaging batch b, perturbation p. 배치 키는 CP-BG-Bench의 Harmony 키를 따른다: JUMP-CP는 source, RxRx1은 source(experiment), RxRx3-core는 plate.
- 작업 가정: x = g(m, n, ρ, t_b). m은 중심 세포 형태, n은 이웃 세포 표현형, ρ는 국소 밀도, t_b는 배치 의존 기술 요인(조명, 염색 강도, 초점, 잔해)이다.
- t_b는 세포 내부 강도에도 작용한다. 따라서 S view도 배치 효과에서 자유롭지 않다.

### 3.2 View는 입력 수준의 개입이다

| View | 포함 성분 | 비고 |
| --- | --- | --- |
| C | m, n, ρ(암묵), t_b(세포+배경) | CP-BG-Bench |
| S | m, t_b(세포 내부) | CP-BG-Bench |
| CD / SD | C 또는 S + 명시적 ρ(모서리 패치) | CP-BG-Bench |
| **N (신규)** | m, n, ρ(암묵), t_b(세포 내부). crop 안 모든 세포 마스크 밖의 배경 픽셀만 0, 이웃 세포는 유지 | crop 내 전체 instance mask 필요 |
| **B (선택)** | C에서 배경 픽셀의 강도 분포만 다른 배치의 분포로 교체(histogram matching) | 배경 통계의 인과 효과 측정 |

대비별 의미: C − N = 배경 픽셀 기여(기술 요인 우세 예상), N − S = 이웃 세포 기여(같은 well의 이웃은 같은 perturbation을 받았으므로 생물학적 요인 우세 예상), SD − S와 CD − C = 명시적 밀도 기여.

### 3.3 조작적 정의

- within-study 지표 M_w = replicate mAP, cross-batch 지표 M_a = R@10
- retention: ρ_w = M_w(S) / M_w(C), ρ_a = M_a(S) / M_a(C) (CP-BG-Bench의 94%·32%가 이 정의)
- 밀도 회복률: r_dens = (M(SD) − M(S)) / (M(C) − M(S))
  - 원문 검산: R@10 (0.117 − 0.100) / (0.313 − 0.100) ≈ 0.08, replicate mAP (0.753 − 0.716) / (0.760 − 0.716) ≈ 0.84
- 맥락 성분 k의 판정 (예: N 대 S): g_w = M_w(N) − M_w(S), g_a = M_a(N) − M_a(S). perturbation 단위 bootstrap 95% CI 기준
  - transferable: g_a > 0이고 CI가 0을 포함하지 않음
  - within-only shortcut: g_w > 0이고 g_a의 CI가 0을 포함
  - harmful: g_a < 0
- 표현 수준 목표: z_k ⊥ b | p 이면서 I(z_k; p) > 0 (성분별 조건부 독립)

### 3.4 비생물학적 "전이 신호" 통제

plate layout이 배치 간에 반복되면, well 위치와 상관된 인공물도 transferable로 판정될 수 있다.

1. **비이미지 베이스라인(G1–G3):** well당 세포 수, well 위치, FOV 강도 통계만으로 cross-batch 검색. 세포 수만으로 bioactivity 벤치마크가 상당 부분 예측된다는 보고(Seal et al., Nat Commun 2026)를 반영한다.
2. **위치 층화 평가:** 쿼리와 plate 위치가 같은 갤러리 well을 제외하고 R@k를 다시 계산한다.
3. **위치로 설명할 수 없는 생물학적 정답:** 유전자 관계 recall(RxRx3-core), 화합물 표적·MoA 검색(JUMP-CP).

### 3.5 최적화 목표

추론 입력은 C view 하나다. held-out 배치 b*에서의 cross-batch 검색 성능을 최대화하되, CP feature prediction, replicate mAP, scIB에서 동일 백본 베이스라인보다 δ 이상 나빠지지 않아야 한다.

---

## 4. 방법: PVIL

### 4.1 설계 원칙

1. **통제 비교 가능성:** 백본, LoRA 설정, perturbation 임베딩, 미니배치 샘플러, 후처리는 CP-BG-Bench와 동일하다. 성능 차이는 손실 설계에서만 나와야 한다.
2. **성분별 차등 정규화:** 강한 배치 불변성은 맥락 성분에만 걸고, 세포 성분에는 약하게(0 포함) 건다. 미세한 형태 정보를 지우지 않기 위해서다.
3. **맥락 보존:** 맥락을 지우지 않고 별도 성분으로 남긴다. 추론 시 맥락 가중치 α로 사용량을 조절하고 분석한다.

### 4.2 구조

| 구성 요소 | 설정 | 근거 |
| --- | --- | --- |
| 백본 | DINOv3 ViT-B/16 (timm `vit_base_patch16_dinov3.lvd1689m`), 1×1 conv 채널 어댑터, LoRA r=8, α=16 | CP-BG-Bench와 동일 |
| 공유 trunk 출력 | h = f(x), 768차원 CLS 토큰 | 동일 |
| 세포 head φ_c | h → z_c (기본 96차원) | 신규 |
| 맥락 head φ_x | h → z_x (기본 32차원) | 신규. 분할 비율은 ablation |
| 최종 임베딩 | z = [z_c ; α·z_x], 총 128차원 | CP-BG-Bench의 128차원과 맞춤 |
| perturbation 인코더 ψ | ECFP4(화합물), ESM2(유전자) → 128차원 | 동일 |
| 확장 | 같은 손실을 OpenPhenom, SubCell 백본에 적용 | 범용성 검증 (E7) |

### 4.3 손실 함수

```
L = L_task + λ_v·L_view + λ_n·L_null + λ_x·L_inv(z_x) + λ_c·L_inv(z_c)  [+ λ_ρ·L_dres]
```

1. **L_task (과제 손실):** CP-BG-Bench의 대칭 multi-positive contrastive loss를 그대로 쓴다. 세포 성분도 단독으로 perturbation을 예측하도록 S view에도 가중치 β로 적용한다(β = 0은 ablation).
   `L_task = L_mpc(z(x^C), ψ(p)) + β · L_mpc(z(x^S), ψ(p))`
2. **L_view (개입 일관성):** C와 S는 중심 세포가 같고 맥락만 다르다. 세포 성분은 맥락 제거라는 개입에 대해 변하지 않아야 한다. N view 변형에서는 z_c(x^N)도 같은 방식으로 맞춘다.
   `L_view = [1 − cos(z_c(x^C), sg[z_c(x^S)])] + [1 − cos(z_c(x^S), sg[z_c(x^C)])]`
3. **L_null (맥락 성분의 범위 한정):** 맥락이 제거된 S view에서는 맥락 성분이 비어 있어야 한다. 이 제약으로 z_x는 세포 바깥 정보만 담고, 세포 내부의 배치 효과는 z_x로 새지 않는다.
   `L_null = ||z_x(x^S)||²`
4. **L_inv (배치 조건부 불변성):** z ⊥ b | p를 구현한다. 기본은 perturbation 임베딩을 조건으로 받는 배치 판별기 D와 gradient reversal이다. CP-BG-Bench 샘플러가 한 미니배치 안에서 같은 perturbation을 여러 배치에 걸쳐 뽑으므로 조건부 판별이 미니배치 안에서 가능하다. 대안 구현(조건부 평균 정렬, VREx)은 ablation으로 비교한다.
   `L_inv(z_x) = −CE(D(GRL(z_x), sg[ψ(p)]), b)`
   λ_x는 크게, λ_c는 작게(0 포함) 둔다. 세포 내부 염색 강도 같은 배치 효과는 λ_c로 다루되, CP feature prediction으로 형태 정보 손실을 감시한다.
5. **L_dres (선택: 밀도 잔차 제거):** 같은 perturbation 안에서의 밀도 편차 ρ − ρ̄(p)만 판별기로 제거한다. perturbation에 의한 밀도 변화(세포독성)는 남고 well마다 다른 seeding 잡음만 지워진다.

### 4.4 학습 절차

1. 미니배치: 64 perturbation × 8 세포, imaging batch 간 층화(CP-BG-Bench와 동일). 각 세포에 대해 C, S view(선택적으로 N)를 함께 불러온다.
2. 증강은 CP-BG-Bench와 동일하다. 같은 세포의 view 쌍에는 **같은 기하 변환**을 적용해 개입 쌍의 대응을 유지한다.
3. 판별기 D의 GRL 계수는 warmup 동안 0에서 목표값까지 선형 증가시킨다.
4. 하이퍼파라미터는 테스트 배치가 아닌 별도 validation 배치에서 고른다. 선택 기준은 validation cross-batch R@10이고, validation replicate mAP의 비열등을 조건으로 둔다. 모든 방법에 같은 탐색 예산을 준다.

### 4.5 추론

- 입력은 C view 하나. z = [z_c(x^C) ; α·z_x(x^C)]. 기본 α = 1, 분석용으로 α ∈ {0, 0.5, 1}을 보고한다.
- 집계는 CP-BG-Bench와 동일하며 Harmony 전후 모두 평가한다.

### 4.6 대안 설명 배제

| 대안 설명 | 배제 방법 |
| --- | --- |
| 계산량이 늘어서 좋아졌다 | B1: C view만으로 2배 step |
| view 증강 효과일 뿐이다 | B2: C/S/CD/SD를 무작위로 섞어 같은 L_task로만 학습 |
| cross-batch positive 때문이다 | B3: positive를 다른 배치 샘플로만 제한 |
| 적대적 배치 제거 때문이다 | B4: 임베딩 전체에 조건부 적대적 배치 제거(분해 없음) |

### 4.7 이론적 동기 (보장 아님)

내용이 고정되고 스타일만 바뀐 view 쌍의 정렬 학습이 내용 블록을 식별한다는 결과(von Kügelgen et al., NeurIPS 2021)와 같은 구조다. 다만 S는 무작위 스타일 변화가 아니라 결정적 마스킹이므로 정리의 가정이 그대로 성립하지 않는다. 논문에서는 동기로만 제시한다.

---

## 5. 데이터셋과 분할

### 5.1 주 데이터셋 (CP-BG-Bench 구성) [확인]

| 데이터셋 | 교란 / 세포 | wells / perturbation | 배치 키 / 원 held-out | plate 레이아웃 | 라이선스 |
| --- | --- | --- | --- | --- | --- |
| JUMP-CP (cpg0016 부분집합; 본 연구는 근사 부분집합 28,003 / 907, `docs/decisions.md` 2026-10-14) | 화합물 / U2OS | 28,003 / 1,466 | source 10개 / source_4 | 화합물 plate는 10개 source에 걸쳐 무작위화 | paired-view를 CC0로 재배포 예정 |
| RxRx1 (HepG2) | siRNA / HepG2 | 12,581 / 1,108 | experiment 11개 / HEPG2-07 | siRNA 위치는 experiment·plate마다 무작위. 단 277개씩 묶인 4개 그룹은 항상 같은 plate에 함께 배치 | CC BY-NC-SA 4.0 (CP-BG-Bench는 CC BY-NC로 표기) |
| RxRx3-core | CRISPR-KO / HUVEC | 104,166 / 735 | plate 9개 / plate 4 | 각 CRISPR 실험은 같은 perturbation 묶음을 9회 반복하며 plate마다 무작위화 | Recursion Commercial EULA (HF `LICENSE`, CC 아님. 귀속 표기 필수, 신경과학 분야 사용 금지, 파생물 공유 시 동등 조건) |

- wells·perturbation·배치 수는 CP-BG-Bench Table 4 기준(대조군 포함).
- RxRx3-core 원 데이터: 222,601 wells, 유전자 736개(HF 카드는 735개), 화합물 1,674개 × 8 농도. 원본 2048×2048에서 512×512 중앙 crop, uint8, JPEG 2000 압축. 압축 손실이 배경 통계에 영향을 줄 수 있다.
- RxRx3-core의 배치 키 "plate"는 실험 내 반복 번호(1–9)다. CRISPR wells의 plate 열은 1–9이고 176개 gene 실험이 모두 1–9를 가진다. 1–48 범위는 화합물 실험이다 (2026-10-14 메타데이터로 확인).
- RxRx1: 384-well plate의 내부 308 wells, well당 512×512 두 장, 6채널.

### 5.2 분할 프로토콜

| 프로토콜 | 내용 | 용도 |
| --- | --- | --- |
| P-orig | CP-BG-Bench와 동일. held-out 배치로 early stopping과 평가를 모두 한다 | 원 논문 수치와 직접 비교 (E0, 부록) |
| **P-strict** | 학습 배치 중 하나를 validation으로 떼어 early stopping과 하이퍼파라미터 선택에만 쓴다. validation 배치는 "학습 배치 중 well 수가 중앙값인 배치"로 사전 등록. 테스트 배치는 최종 1회만 평가 | 주 결과 |
| P-LOBO | 데이터셋마다 3개 fold. 원 held-out 배치 + 고정 seed로 뽑은 배치 2개 | 강건성 (E4) |

### 5.3 외부 검증 데이터

| 데이터 | 구성 | 목적 |
| --- | --- | --- |
| RxRx1 전체 4개 세포주 | experiment 51개: HUVEC 24, RPE 11, HepG2 11, U2OS 5. 공식 batch-separated split은 학습 33개(16/7/7/3), 테스트 18개(8/5/5/2) | 세포주 간 일반화 (E5). HepG2 외 세포주는 view 재구성 필요 |
| RxRx3-core 공식 벤치마크 (EFAAR 저장소) | 유전자 간 알려진 관계 recall(CORUM, hu.MAP, Reactome, StringDB; 코사인 상·하위 5%; 무작위 기준 약 0.10), 화합물–유전자 DTI(AP, AUC, 음성 샘플링 100 seeds) | 위치 교란으로 설명할 수 없는 생물학적 정답 (E6). DTI는 화합물 wells 재구성 필요 |
| JUMP-CP 화합물 주석 | 표적·MoA 주석 가용성 [미확인] | 화합물 쪽 생물학적 정답. 불가하면 RxRx3-core DTI로 대체 |
| RxRx1-WILDS (선택) | 3채널 변형, 배치 기준 domain generalization split | DG 문헌(예: ABRA)과의 비교용 부록 |

### 5.4 전처리

- CP-BG-Bench 부록 A.1 사양을 그대로 따른다(`docs/tasks/P0_reproduction.md`).
- N view를 위해 crop 안 **모든** 세포 instance mask를 보존한다. 원 파이프라인은 중심 세포 마스크만 쓰므로 수정이 필요하다.
- 저장은 150×150 crop + mask만 한다(데이터셋당 약 200 GB [추정]). view는 학습 중에 생성한다. 모든 view를 224×224×6채널로 저장하면 데이터셋당 약 1.8 TB [추정].

### 5.5 데이터 가용성 리스크

CP-BG-Bench 아티팩트는 2026-10-14 기준 미공개다. (1) 제1저자에게 사전 공개를 요청하고, (2) 부록 A.1–A.4 사양대로 자체 구현을 병행한다. JUMP-CP 부분집합의 정확한 plate·화합물 목록은 원문에 없으므로 저자 요청의 1순위다.

---

## 6. 베이스라인 (필수 15종: A 2, B 5, C 4, D1, G 3)

| 그룹 | ID | 내용 | 구분 |
| --- | --- | --- | --- |
| A 통제 | A1 | DINOv3 ViT-B/16 + LoRA, C·S·CD·SD view 각각, 3 seeds | 필수 |
| | A2 | SubCell, OpenPhenom 백본으로 같은 4개 view | 필수 (E7의 짝) |
| B 대안 설명 배제 | B1 | C view만으로 2배 step | 필수 |
| | B2 | C/S/CD/SD 무작위 혼합 + 같은 L_task | 필수 |
| | B3 | positive를 다른 배치 샘플로만 제한한 샘플러 (저자 제안 1) | 필수 |
| | B4 | 임베딩 전체에 조건부 적대적 배치 제거, 분해 없음 (저자 제안 2) | 필수 |
| | B5 | 배경 픽셀을 다른 배치 crop의 배경으로 바꾸는 증강 (Xiao et al., ICLR 2021 계열). H1의 대조군 | 필수 |
| C 일반 DG | C1 | DANN (비조건부 적대적 배치 제거) | 필수 |
| | C2 | Deep CORAL (배치 간 2차 통계 정렬) | 필수 |
| | C3 | VREx (배치별 손실 분산 벌점) | 필수 |
| | C4 | GroupDRO (최악 배치 손실 최소화) | 필수 |
| | C5 | IRM | 선택 |
| D CP 전용 | D1 | CWA-MSN. 원문에 코드 링크 없음 → cross-well 샘플링·정렬을 같은 백본에 재구현 | 필수 |
| | D2 | MICON. 화합물 전용이므로 JUMP-CP에만 적용 | 권장 |
| | D3 | CellCLIP (NeurIPS 2025, github.com/suinleelab/CellCLIP). 입력 단위가 달라 통제 비교에서는 제외 | 외부 참조 |
| | D4 | ABRA (Knowledge-Based Systems 2026). 원문에 코드 링크 없음 | 선택 (RxRx1 부록) |
| | D5 | Weakly supervised CNN (Moshkov et al., Nat Commun 2024) | 외부 참조 |
| E 고정 인코더 | E1 | OpenPhenom-S/16. RxRx3-core 논문에 따르면 RxRx3와 cpg0016(source 4·13의 CRISPR·ORF)으로 사전학습 → 평가 데이터 중첩 가능성을 표에 명시 | 참조 |
| | E2 | SubCell (HPA 사전학습, 4채널 입력) | 참조 |
| | E3 | cp_measure 고전 특징 (CP-BG-Bench의 CPM) | 참조 |
| F 사후 보정 | F1 | 대조군 기준 ZCA-cor sphering + PCA 50 | 모든 방법 공통 |
| | F2 | Harmony (전후 모두 보고) | 모든 방법 공통 |
| | F3 | 배치별 대조군 중심화·스케일링(PCA-CS). 임베딩 수준의 AdaBN 대응물 | 모든 방법 공통 |
| G 비이미지 하한선 | G1 | well당 세포 수 | 필수 |
| | G2 | well 위치(행·열) | 필수 |
| | G3 | FOV 수준 채널별 강도 통계 | 필수 |

**선행 증거 [확인]:** RxRx1 원 논문에서 gradient reversal은 배치 분류 정확도를 1.8%까지 낮췄지만 처음 보는 배치로의 일반화는 개선하지 못했다. AdaBN은 개선했다. 따라서 C1의 실패는 예상된 결과이고, PVIL은 성분 한정과 perturbation 조건부로 차이를 보여야 한다. PVIL의 이득은 F1–F3 적용 후에도 남아야 한다.

---

## 7. 평가

### 7.1 CP-BG-Bench 4개 프로토콜 (원 정의 유지)

| 프로토콜 | 정의 | 역할 |
| --- | --- | --- |
| Cross-batch recall | 쿼리 = held-out 배치의 비대조군 well, 갤러리 = 나머지 배치의 비대조군 well. 코사인, macro R@k (k = 1, 5, 10) | **주 지표** |
| Replicate mAP | copairs. positive = 다른 plate의 같은 perturbation well, negative = 같은 plate의 처리군–대조군 쌍. 처리군 well macro 평균, 순열 10,000회 + BH | 비열등성 |
| scIB | Batch(iLISI, kBET, BRAS, graph connectivity, PCR comparison), Bio(cLISI, KMeans NMI, ARI, silhouette label, isolated labels), Total = 0.4·Batch + 0.6·Bio | Bio 비열등, Batch 개선 기대 |
| CP feature prediction | 50차원 임베딩 → cp_measure 특징 ridge(α = 1). 비held-out으로 학습, held-out에서 median Pearson r. 특징 그룹별로도 보고 | 비열등성 |

원 논문은 scIB가 replicate mAP와 거의 같이 움직인다고 지적했다. 비열등성 판단의 핵심은 replicate mAP와 CP feature prediction이다.

### 7.2 생물학적 정답 지표

- 유전자 간 알려진 관계 recall (RxRx3-core, 공식 EFAAR 코드와 PCA-CS 정렬)
- 화합물–유전자 DTI (RxRx3-core, AP·AUC, 무작위 대비 z-score)
- 화합물 주석 검색 (JUMP-CP, 가용 시)

### 7.3 Shortcut 진단 지표

| 지표 | 정의 | 기대 방향 (PVIL) |
| --- | --- | --- |
| ρ_w, ρ_a | §3.3 정의 | ρ_a 증가 |
| r_dens | §3.3 정의 | cross-batch 밀도 의존 감소 |
| 배치 예측도 | 학습 배치 안 5-fold linear probe (well 임베딩 → 배치 ID) | 감소. 일반화를 보장하지 않으므로 성공 기준은 아님 |
| 영역별 attribution | 전체 세포 마스크로 crop을 중심 세포·이웃 세포·배경으로 나눠 occlusion attribution 비율 측정 | 배경 비율 감소, 이웃 세포 비율 유지 |
| 하한선 대비 이득 | R@10(방법) − max(G1, G2, G3) | 증가 |

### 7.4 통계 검정

- 주 매트릭스의 모든 구성은 3 seeds. 평균 ± 표준편차를 보고하고 최고 seed를 고르지 않는다.
- 계층적 bootstrap(seed → 쿼리 perturbation 재표집, 1,000회)으로 R@10 차이의 95% CI.
- perturbation별 R@10(seed 평균)에 Wilcoxon signed-rank. 데이터셋 안에서 베이스라인 수만큼 Holm 보정.
- 비열등성: 단측 검정 (베이스라인 − PVIL) < δ. δ = Phase 0에서 잰 A1 seed 표준편차 × 2, E1 실행 전 고정.

### 7.5 평가 위생

1. 테스트 배치는 최종 구성마다 한 번만 평가한다(P-strict).
2. 하이퍼파라미터는 validation 배치에서만 고르고, 모든 방법에 같은 탐색 예산을 준다.
3. 모든 방법에 같은 후처리를 적용한다.
4. perturbation별 원시 결과와 평가 스크립트를 공개한다.

---

## 8. 실험 매트릭스 (학습 run 총 528회)

| ID | 질문 (가설) | 구성 | 데이터셋·프로토콜 | 학습 run | 우선 |
| --- | --- | --- | --- | --- | --- |
| E0 | 원 논문 재현 | A1(DINO) × 4 view × 3 seeds, G1–G3 | 3개 · P-orig, P-strict | 36 | P1 |
| E1 | PVIL vs 통제 (H2) | PVIL, B1–B5, C1–C4, D1 = 11개 방법 × 3 seeds. A1은 E0 재사용 | 3개 · P-strict, 4개 프로토콜 + 진단, F1–F3 전후 | 99 | P1 |
| E2 | 맥락 분해 (H1) | A1을 N view로 학습. A1-C와 PVIL에 테스트 시 C·N·S·CD·SD·B 입력. 위치 층화 평가 | 3개 · P-strict | 9 | P1 |
| E3 | 구성 요소 기여 | PVIL 변형 10종 × 3 seeds (아래 목록) | RxRx3-core, RxRx1 · P-strict | 60 | P1 |
| E4 | 고정 배치 의존성 (RQ3) | PVIL, A1-C, B·C 중 최강, D1 × 추가 fold 2개 × 3 seeds | 3개 · P-LOBO | 72 | P2 |
| E5 | 세포주·대규모 (RQ3) | (a) HepG2 학습 모델을 HUVEC·RPE·U2OS에 평가(학습 없음) (b) 공식 split으로 PVIL, A1-C × 3 seeds | RxRx1 전체 | 6 | P2 |
| E6 | 생물학적 타당성 | E1 체크포인트로 유전자 관계 recall, DTI, 화합물 주석 검색, CP 특징 그룹별 예측 | RxRx3-core, JUMP-CP | 0 | P1 |
| E7 | 백본 범용성, 데이터 효율 | (a) SubCell·OpenPhenom에서 PVIL vs A2-C × 3 seeds (b) 학습 데이터 25%·50%에서 PVIL vs A1-C × 3 seeds | (a) 3개 (b) RxRx3-core | 48 | P2 |
| E8 | segmentation 없는 추론, 마스크 강건성 (H4) | α 스윕과 C-only 추론(학습 없음). 마스크 침식·팽창 잡음 2단계 × 3 seeds | RxRx3-core | 6 | P3 |
| HP | 공정한 탐색 | 조정 항이 있는 8개 방법(PVIL, B4, B5, C1–C4, D1) × 8 trials × 3개 데이터셋 × 1 seed | validation 배치만 | 192 | P1 |

**E3 변형 10종:** (1) L_view 제거 (2) L_null 제거 (3) L_inv 제거 (4) λ_c = 0 (5) β = 0 (6) L_dres 추가 (7) L_inv = 조건부 평균 정렬 (8) L_inv = VREx (9) z_c/z_x 분할 64/64 (10) N view 일관성 추가.

**실험 순서의 논리**

1. E0으로 측정 도구를 검증하고 δ를 고정한다.
2. E2(진단)를 E1보다 먼저 일부 돌린다. N view 결과가 PVIL 설계(이웃 세포를 맥락 성분에 남길지)를 확정한다.
3. 방법 개발은 RxRx3-core에서 시작해 RxRx1, JUMP-CP로 넓힌다.
4. E1과 E3의 validation 결과를 확인한 뒤에만 테스트 배치를 연다.

---

## 9. 일정, 관문, 연산 예산

### 9.1 단계

| 단계 | 기간 | 세부 작업 | 관문 |
| --- | --- | --- | --- |
| P0 재현 | 2026-10-15 ~ 11-30 | 재구성 파이프라인 확보(공개 요청 또는 재구현), N view용 전체 마스크 보존, E0, G1–G3, T 실측, δ 확정, 배치 키·주석 가용성 확인 | G0 |
| P1 진단 | 2026-12-01 ~ 2027-01-15 | E2(N view, 테스트 시 개입, 위치 층화), 워크숍 원고 | G1 |
| P2 방법 개발 | 2027-01-16 ~ 03-15 | RxRx3-core → RxRx1 → JUMP-CP 순서로 PVIL 개발, HP 탐색, E3(validation 배치) | G2 |
| P3 본 실험 | 2027-03-16 ~ 04-20 | E1(테스트 배치 개봉), E4, E5, E6, E7 | — |
| P4 집필 | 2027-04-21 ~ 마감 | 집필, 그림, 코드·체크포인트 공개 준비 | 제출 |

목표: NeurIPS 2027 본 트랙(통상 5월 마감, 공식 일정 [미확인]). 백업: ICLR 2028. P1 결과는 ICLR 2027 워크숍(LMRL 계열이 열릴 경우)에 먼저 낼 수 있다.

### 9.2 관문 기준 (사전 등록)

| 관문 | Go 조건 | No-go 시 대응 |
| --- | --- | --- |
| G0 | **(2026-10-16 정성 재현으로 변경, 사용자 승인)** DINO × 4 view × 3 데이터셋(seed 0, P-orig, post-Harmony)에서 원문 현상 Q1–Q6 모두 성립: Q1 R@10 C > S(3개 모두) · Q2 RxRx1·RxRx3-core에서 (SD−S)/(C−S) < 0.5 · Q3 \|CD−C\| < \|C−S\|(3개 모두) · Q4 C의 R@10 RxRx1 > RxRx3-core > JUMP · Q5 Harmony 전후 R@10 차이가 RxRx3-core에서 가장 작음 · Q6 JUMP·RxRx3-core에서 CP r S ≥ C. Q7 replicate mAP 격차(C−S) RxRx1 > JUMP는 mAP 정의 확정 후 판정(정의는 학습 임베딩이 무작위보다 확실히 높아야 함). 수치 크기 차이(예: S/C 비율)는 기준이 아니며 부록에 그대로 보고. 이전 기준(R@10·mAP가 원문과 절대 0.03 이내)은 폐기 | 미달 현상을 부록에 보고하고, 해당 현상에 의존하는 가설(H1–H3)의 해석을 재검토. 원 저자 아티팩트가 공개되면 기준선을 대체하고 차이를 부록에 보고 |
| G1 | N view 결과로 맥락 성분의 전이 여부가 판정됨(H1 지지 또는 기각) | H1 기각(N ≈ S) 시 B view로 배경 통계의 기여를 규명하고, 맥락 성분 정의를 "배치 간 일관된 배경 신호"로 수정 |
| G2 | validation 배치에서 PVIL이 A1-C와 B4를 3개 중 2개 이상 데이터셋에서 이기고, validation replicate mAP는 비열등 | §10의 전환 경로 |

### 9.3 연산 예산 [추정]

기준 run(BRE) = A1 한 번의 학습 비용. view 두 개를 넣는 방법은 2 BRE.

| 블록 | 학습 run | BRE |
| --- | --- | --- |
| E0 | 36 | 36 |
| E1 | 99 | 약 126 |
| E2 | 9 | 9 |
| E3 | 60 | 120 |
| E4 | 72 | 약 117 |
| E5 | 6 | 약 30 (데이터 약 3.3배) |
| E7 | 48 | 약 61 |
| E8 | 6 | 12 |
| HP | 192 | 약 233 |
| **합계** | **528** | **약 744** (P1만 약 524) |

- **T 추정 가정:** 데이터셋당 학습 세포 최대 약 1.1–1.3M(샘플링 규칙상 상한, 예: RxRx3-core 735 유전자 × 학습 plate 8개 × 200), 25 epoch, ViT-B/16 224 px 순전파 약 17.6 GFLOPs, LoRA 학습 비용 = 순전파의 2–3배, A100 bf16 활용률 30–50%.
- 이 가정에서 T ≈ 2–5 A100 GPU-시간. 전체 약 1,500–3,700, P1만 약 1,050–2,600 A100 GPU-시간.
- **B200에서 E0로 T를 실측해 이 표를 갱신한다.** Cellpose-SAM 분할과 cp_measure 계산 비용도 Phase 0에서 따로 잰다.
- 절감 수단: E3·E4를 2 seeds로(약 −80 BRE), HP trial 8 → 6(약 −58 BRE), E3를 RxRx3-core 하나로(−60 BRE).

---

## 10. 리스크와 대응 (우선순위순)

| # | 리스크 | 조기 신호 | 대응 |
| --- | --- | --- | --- |
| 1 | 적대적 항이 배치 예측도만 낮추고 cross-batch 검색은 개선하지 못함 | E3에서 L_inv를 빼도 R@10 차이 없음 | L_inv를 조건부 평균 정렬 또는 VREx로 교체. 배치별 정규화(F3)와의 결합 검토 |
| 2 | 전이 이득의 원천이 비생물학적 신호(레이아웃, RxRx1의 277개 고정 그룹) | 위치 층화나 같은 그룹 갤러리 제외 분석에서 성능 급락 | E6의 생물학 정답 지표를 1차 증거로 격상. 그룹 구조 통제 분석을 본문에 |
| 3 | CP-BG-Bench 아티팩트 공개 지연, 재현 실패 | G0 미달 | 저자 요청과 자체 구현 병행, 재현 차이를 부록에 보고 |
| 4 | H1 기각(이웃 세포가 전이 이득의 원천이 아님) | G1 | B view로 배경 통계 기여 규명, 맥락 성분 정의 수정 |
| 5 | JUMP-CP에서 이득이 작음(원래 C→S mAP 격차 0.03) | E1 JUMP-CP 결과 | 사전 등록한 2차 주장으로 보고. H2는 기각으로 보고 |
| 6 | 비열등성 실패(CP feature prediction 하락) | E3의 λ_c 변형 | λ_c = 0, β 증가로 세포 성분 보호. cp_measure 특징을 보조 학습 목표로 쓰는 것은 평가 누수이므로 금지 |
| 7 | 경쟁 연구 선점 | arXiv 모니터링 | 차별점(N view 분해, 성분별 차등 정규화, 비이미지 통제) 유지, G1 결과 워크숍 선공개 |
| 8 | 연산 부족 | Phase 0의 T 실측 | §9.3 절감 수단 |
| 9 | 마스크 품질 의존(S·N view 오류) | E8 마스크 잡음 실험 | 잡음 강건성 보고, Cellpose-SAM QC 기준 명시 |
| 10 | 라이선스 | Phase 0 | RxRx1 파생 이미지는 재배포 시 같은 조건이 붙음. 이미지 대신 재구성 코드 공개 |
| 11 | OpenPhenom 사전학습 데이터 중첩 | 이미 확인된 사실 | 표에 명시, 결론은 DINO 백본 결과에 근거 |

**G2 실패 시 전환 경로:** (1) 진단 논문으로 전환(E0, E2, 비이미지 통제, 위치 층화; 벤치마크·데이터셋 트랙이나 워크숍) (2) 방법 수정(배치 수준 정규화·테스트 시 적응, 또는 N view를 주 입력으로) (3) 범위 축소(단일 시설 데이터셋 전용 방법).

---

## 11. 논문 구조

- 서사: 진단 → 정의 → 방법 → 검증. 결과 절은 진단(무엇이 전이되는가)을 주 비교보다 먼저 둔다.
- 섹션과 분량: Abstract(약 200 단어) → Introduction(1.25쪽) → Related Work(0.75쪽) → Problem Formulation(1쪽) → Method: PVIL(1.5쪽, Algorithm 1) → Experimental Setup(1쪽) → Results(2.5쪽) → Discussion & Limitations(0.5쪽) → Conclusion, Reproducibility statement → Appendix A–H (A 재현, B 구현·HP 공간, C Harmony 전후 전체 결과, D CP 특징 그룹별, E α 스윕·마스크 잡음·데이터 효율, F 통계, G 라이선스·전처리, H 연산량).
- Results 순서: 무엇이 전이되는가(E2) → 주 비교(E1) → 3축 분해(H3) → ablation(E3) → 강건성·일반화(E4·E5·E7) → 생물학적 타당성(E6).

| ID | 내용 | 근거 실험 | 전달할 한 문장 |
| --- | --- | --- | --- |
| Fig 1 | 같은 세포의 C·N·S·CD·SD view + 베이스라인과 PVIL의 within·cross retention 막대 | E1, E2 | 맥락은 지우지 말고 나눠야 한다 |
| Fig 2 | PVIL 구조와 손실 배치도 | — | 세포 성분에는 일관성, 맥락 성분에는 조건부 불변성 |
| Fig 3 | 데이터셋별 C·N·S·CD·SD·B의 cross-batch R@10과 비이미지 하한선 | E2 | 무엇이 전이되고 무엇이 전이되지 않는가 (결과에 따라 문장 수정) |
| Fig 4 | 모든 방법의 ρ_w 대 ρ_a 산점도 | E1 | 기존 인코더는 한 패턴에 모이고 PVIL만 벗어난다 (H3) |
| Fig 5 | P-LOBO fold별 R@10 차이, 세포주 간 결과 | E4, E5 | 이득은 특정 배치나 세포주에 국한되지 않는다 |
| Table 1 | 3개 데이터셋 × {R@1, R@10, replicate mAP, scIB Bio·Batch, CP feature r} × 방법(그룹별), 평균 ± 표준편차, 유의성 | E1 | cross-batch에서 우위, 나머지에서 비열등 |
| Table 2 | ablation | E3 | 이득은 성분 분해와 조건부 불변성의 결합에서 나온다 |
| Table 3 | 백본 범용성, 데이터 효율 | E7 | 특정 백본 전용이 아니다 |
| Table 4 | 유전자 관계 recall(CORUM, hu.MAP, Reactome, StringDB), DTI | E6 | 늘어난 신호는 생물학적 정답과 맞는다 |

리뷰어 예상 질문과 대응 위치: 계산량·증강 효과(B1, B2) · 저자 제안과의 차이(B3, B4, Table 2) · 테스트 배치 early stopping(P-strict, 부록 A에 P-orig) · 위치 교란(G1–G3, 위치 층화, Table 4) · OpenPhenom 사전학습 중첩(부록 G) · 단일 seed(3 seeds, 계층적 bootstrap, Holm).

---

## 12. 착수 전 확인 항목

- [x] CP-BG-Bench 아티팩트: 저자 문의 안 함, 자체 구현 (2026-10-14 결정)
- [x] JUMP-CP 부분집합: 근사 규칙 확정(907 화합물, 28,003 wells, `configs/data/jump_cp_wells.csv.gz`). G0는 "근사" 표기 (2026-10-14)
- [x] RxRx3-core 배치 키 "plate" = 실험 내 plate 번호 1–9 (2026-10-14)
- [x] RxRx3-core 라이선스 = Recursion Commercial EULA (2026-10-14). 파생물(학습된 모델 포함) 공유 시 동등 조건 적용 여부는 공개 전에 검토
- [ ] JUMP-CP 화합물 표적·MoA 주석 가용성
- [ ] CWA-MSN, ABRA 코드 공개 여부
- [ ] SubCell 가중치 라이선스와 입력 사양
- [x] 재구성 파이프라인에서 crop 내 전체 세포 마스크 확보 가능 여부 (N view 전제 조건): crop zarr에 crop 안 모든 세포의 핵·세포 instance mask 저장 (2026-10-15, RxRx1·RxRx3-core)
- [ ] B200 기준 1회 학습 비용 T 실측
- [ ] NeurIPS 2027 마감일과 분량 규정
- [ ] 방법 이름 PVIL 중복 검색

---

## 13. 참고문헌 (핵심)

**[확인] 원문 직접 확인**

- CP-BG-Bench (Treis et al., 2026): https://arxiv.org/abs/2609.12761
- RxRx3-core (Kraus et al., 2025, LMRL@ICLR 2025): https://arxiv.org/abs/2503.20158 · 데이터: https://huggingface.co/datasets/recursionpharma/rxrx3-core · 벤치마크 코드: https://github.com/recursionpharma/EFAAR_benchmarking
- RxRx1 (Sypetkowski et al., 2023): https://arxiv.org/abs/2301.05768 · WILDS 데이터셋 페이지: https://wilds.stanford.edu/datasets/
- CWA-MSN: https://arxiv.org/abs/2509.19896
- ABRA (Knowledge-Based Systems 2026): https://arxiv.org/abs/2603.05622
- CellCLIP (NeurIPS 2025): https://neurips.cc/virtual/2025/poster/117626 · 코드: https://github.com/suinleelab/CellCLIP
- MICON: https://arxiv.org/abs/2504.09544

**[인용 경유] 원문 확인 문헌의 참고문헌 목록에서 서지 확인**

- copairs (Kalinin et al., Nat Commun 16:5181, 2025) · scIB (Luecken et al., Nat Methods 19:41–50, 2022) · Harmony (Korsunsky et al., Nat Methods 16:1289–1296, 2019) · Arevalo et al. (Nat Commun 15:6516, 2024) · Seal et al. (Nat Commun 2026, doi:10.1038/s41467-026-68725-5) · Moshkov et al. (Nat Commun 15:1594, 2024) · OpenPhenom/MAE (Kraus et al., CVPR 2024, arXiv:2404.10242) · SubCell (Gupta et al., bioRxiv 2025) · DINOv3 (arXiv:2508.10104) · Cellpose-SAM (bioRxiv 2025, doi:10.1101/2025.04.28.651001) · cp_measure (arXiv:2507.01163) · EFAAR/PCA-CS (Celik et al., PLoS Comput Biol 20(10):e1012463, 2024) · CLOOME (Nat Commun 14:7339, 2023) · SupCon (Khosla et al., NeurIPS 2020) · LoRA (arXiv:2106.09685) · DANN (Ganin et al., JMLR 2016) · IRM (arXiv:1907.02893) · AdaBN (arXiv:1603.04779)

**[미확인] 표준 문헌, 제출 전 서지 확인 필요**

- Deep CORAL (arXiv:1607.01719) · VREx/REx (arXiv:2003.00688) · GroupDRO (arXiv:1911.08731) · von Kügelgen et al. 2021 (arXiv:2106.04619) · Xiao et al. 2021 (arXiv:2006.09994) · Geirhos et al. 2020 (arXiv:2004.07780)
