# P0 작업 지시서: CP-BG-Bench 재현

- 기간: 2026-10-15 ~ 2026-11-30 · 관문: G0
- 출처: 아래 사양과 참조값은 모두 CP-BG-Bench(arXiv 2609.12761) 본문과 부록 A.1–A.4에서 확인한 값이다.
- 원문에 없는 값은 **[원문 없음]**, 설치·형식을 직접 확인해야 하는 항목은 **[확인 필요]**로 표시했다. 이런 값을 임의로 정하지 말고, 선택과 이유를 `docs/decisions.md`에 기록한다.

---

## 1. 목표와 완료 조건

- **목표:** 원 논문의 데이터 → view → 학습 → 평가 파이프라인을 재현해, 이후 모든 실험의 측정 도구로 쓴다.
- **G0 (Go 조건):** DINO ViT-B/16의 C·S 구성에서 cross-batch R@10과 replicate mAP(post-Harmony, P-orig)가 3개 데이터셋 모두 §6 참조값과 절대 0.03 이내.
- **산출물**
  1. 재구성 파이프라인 코드와 단위 테스트
  2. crop + instance mask 저장소 (3개 데이터셋)
  3. A1 체크포인트: DINO × 4 view × 3 데이터셋 × 3 seeds (36개)
  4. 평가 스위트 (4개 프로토콜 + 비이미지 베이스라인 G1–G3)
  5. `results/E0/summary.md`: 참조값 대비 재현표
  6. `docs/decisions.md`: δ, validation 배치, 근사 규칙 등
  7. `docs/progress.md`: B200 기준 1회 학습 비용 T

---

## 2. 블로커 (가장 먼저 처리)

1. **원 저자 아티팩트:** 저자 문의는 하지 않는다(2026-10-14 결정). 부록 A.1–A.4 사양대로 자체 구현한다. 나중에 아티팩트가 공개되면 그때 비교·대체를 검토한다.
2. **JUMP-CP 부분집합 정의 [원문 없음]:** 원문은 "cpg0016 부분집합, U2OS, 화합물 1,466개, source 10개, wells 28,003(대조군 포함)"만 밝힌다. 저자 문의는 하지 않으므로(2026-10-14 결정) 공개 정보로 근사 규칙을 만들어 진행한다. JUMP-CP의 G0 판정은 "근사"로 표기한다. **[근사 규칙 확정, 2026-10-14]** source_2–11, COMPOUND·TARGET2 plate, 화합물 = TARGET2 비대조군 293개 ∪ 6개 이상 source에 나오는 화합물 = 907개, (화합물, source)당 최대 4 wells(seed 0) = 처리군 19,706 + DMSO 8,297 = 28,003 wells, plate 1,149개, held-out source_4(1,339 wells, 쿼리 화합물 293). 정의 파일 `configs/data/jump_cp_wells.csv.gz`, 코드 `src/pvil/data/jump_subset.py`. source_4에는 COMPOUND plate가 없어 화합물이 TARGET2뿐이다.
3. **RxRx3-core 배치 키 [해결, 2026-10-14]:** CRISPR wells(`perturbation_type == "CRISPR"`, 126,900개)의 `plate`는 1–9이고, 176개 gene 실험이 모두 plate 1–9를 가진다(1–48은 화합물 실험). 따라서 배치 = plate 번호(1–9)이고 held-out은 모든 CRISPR 실험의 plate 4 wells다. **미결:** 원문 well 수 104,166 / 735 perturbation과 맞는 well_type 조합이 없다(Query guides 76,629 / 734, 전체 126,900 / 736). 원문이 세포 수 등 추가 필터를 썼을 가능성이 있다. 분할·QC 뒤에 다시 비교한다.

---

## 3. 데이터·view 사양 (부록 A.1)

| 항목 | 값 |
| --- | --- |
| 분할 | Cellpose-SAM, 기본 파라미터. **가중치 = `cpsam`(v1, 2025-04) 고정, 정밀도는 cellpose 기본값 bf16** (2026-10-14 결정). `cpsam_v2`는 E8 마스크 강건성 실험의 추가 조건으로만 쓴다. 핵 = DNA 채널, 세포 = AGP+DNA 또는 Phalloidin+DNA (데이터셋별 채널 라우팅) |
| 핵-세포 매칭 | 마스크 겹침으로 매칭. 매칭되지 않은 객체는 제거 |
| crop | 세포 centroid 중심 150×150 px. centroid가 FOV 경계에서 75 px 이내면 제외 |
| 리사이즈 | 224×224. 형광 채널은 bilinear, 마스크는 nearest |
| QC | 핵 면적, 세포질 면적, 핵/세포질 비율 중 하나라도 [0.025, 0.975] 분위수 밖이면 제거 |
| 샘플링 | `uniform_per_compound_source`. perturbation×source당 목표 100개(JUMP-CP, RxRx1), 200개(RxRx3-core). well당 세포 수는 크게 다를 수 있음 |
| S view | DNA 채널은 핵 마스크 밖을 0, 나머지 형광 채널은 세포 마스크 밖을 0 |
| CD / SD view | 모든 채널에 22×22 px 정사각 패치 4개(각 모서리에서 7 px 안쪽). 강도 = 원 FOV의 세포 수를 세포 풀 전체 기준 min-max로 [0, 255]에 스케일 |
| 채널 수 | 5 또는 6 (JUMP-CP 5, RxRx 6) |
| **본 연구 추가** | crop 안 **모든** 세포의 instance mask 저장(N view용). 원 FOV ID, plate, well, site, centroid 좌표, FOV 세포 수를 메타데이터로 저장 |

### 3.1 원본 데이터 확인 결과 (2026-10-14)

| 항목 | RxRx1 | RxRx3-core | JUMP-CP |
| --- | --- | --- | --- |
| 형식 | 512×512 uint8 PNG, `images/<exp>/Plate<k>/<well>_s<site>_w<ch>.png` | 512×512 uint8 JPEG 2000, parquet 35개, 키 `<exp>/Plate<k>/<well>_s1_<ch>` | 1080×1080 uint16 TIFF(source_4 기준), S3 키 구조 그대로 |
| site / well | 2 | 1 | source별 4–9 (microscope_config) |
| 커버리지 | 메타데이터 125,510 site 전부 6채널 | 222,601 wells 전부 6채널(CRISPR 126,900 포함) | 다운로드 중 (manifest로 확인) |
| 색인 | `raw/rxrx1/image_index.parquet` | `raw/rxrx3_core/image_index.parquet` (shard·row group 위치) | `raw/jump_cp/manifests/*.json` |
| 핵 채널 | w1 (몽타주로 확인) | ch1 (확인) | DNA (load_data 열 이름) |
| 세포 경계 채널 | w3 = Phalloidin으로 가정. HepG2는 세포가 밀집해 시각적으로 불분명 **[확인 필요]** | ch3 = 액틴 섬유가 뚜렷해 Phalloidin으로 확인 | AGP (load_data 열 이름) |
| 나머지 채널 | 염색 이름 미확인. README는 번호만 제공 | ch6에 핵소체 신호(RNA 계열로 보임), 나머지 미확정 | RNA, ER, Mito (열 이름) |
| 주의 | — | JPEG 2000 압축 흔적이 저신호 채널(ch5)에서 계단 모양으로 보인다 | illum `.npy` 보관(사용 여부는 원문에 없음) |

- 색인은 `scripts/index_raw.py {rxrx3_core|rxrx1}`로 다시 만든다. 분할 채널 라우팅: RxRx는 핵 = w1, 세포 = w3+w1. JUMP-CP는 핵 = DNA, 세포 = AGP+DNA.

---

## 4. 인코더·학습 사양 (부록 A.2)

| 항목 | 값 |
| --- | --- |
| 백본 | timm `vit_base_patch16_dinov3.lvd1689m`, 768차원 CLS |
| 채널 어댑터 | 학습 가능한 1×1 conv(bias 없음)로 5·6채널 → 3채널, 이어서 ImageNet 통계로 초기화한 학습 가능 affine 정규화 |
| LoRA | r = 8, α = 16, dropout 0.1. 모든 transformer block의 attention·MLP projection에 적용, 나머지 trunk 고정 |
| 이미지 projection head | GeGLU 2-layer MLP + skip connection → 128차원 |
| perturbation 임베딩 | 화합물: ECFP4 (Morgan radius 2, 2048 bit, chirality-aware, log(1+count) 변환), InChIKey별 사전 계산. 유전자: ESM2 `facebook/esm2_t33_650M_UR50D`, 토큰 평균, 1280차원, 유전자 기호별 사전 계산. 대조군(DMSO, EMPTY) = 전체 perturbation 평균 임베딩 |
| perturbation head | SiLU 2-layer MLP + skip connection → 128차원. LoRA 없음 |
| 손실 | 이미지·perturbation 임베딩을 각각 L2 정규화. logit ℓ_ij = z_i^img · z_j^pert / τ. τ는 학습(초기 0.1, clamp [0.01, 100]). P(i) = 같은 label 전부. L_i = −log Σ_{j∈P(i)} exp(ℓ_ij) / Σ_k exp(ℓ_ik). image→pert와 pert→image를 같은 가중치로 대칭 적용 (in-log multi-positive) |
| 미니배치 | 512 = 64 perturbation × 8. imaging batch 간 층화해 perturbation마다 cross-batch 커버리지 확보. 대조군은 매 배치 고정 포함 |
| 최적화 | AdamW β = (0.9, 0.95), cosine schedule + 10% linear warmup. LR: projection head 3e-4, LoRA 1e-4, τ 1e-5. weight decay 5e-4(projection head만). bf16 mixed precision, gradient clipping 5 |
| 기간 | 최대 25 epoch. early stopping patience 5, 감시 지표 = validation macro image→perturbation R@1 |
| 증강 | 좌우·상하 flip, 0–90° 무작위 회전, 밝기·대비 ±10% |
| E7용 다른 백본 | OpenPhenom: `recursionpharma/OpenPhenom`, 5채널 256×256 네이티브, 384차원, 어댑터 없음. SubCell: HPA로 MAE 사전학습한 ViT-B/16, 2-head attention pooling 1536차원, 4채널(미세소관·ER·DNA·단백질) 입력 → 나머지 Cell Painting 채널을 1×1 conv로 단백질 슬롯에 결합, 이미지별 min-max 정규화, 224×224 |

---

## 5. 후처리와 평가 사양 (부록 A.3–A.4, 본문 §3.6)

### 5.1 후처리

1. 단일 세포 임베딩 L2 정규화
2. well 평균
3. pycytominer로 ZCA-cor sphering (대조군 well에만 fit, 모든 well에 적용)
4. PCA 50
5. Harmony (θ = 2, λ = 1, σ = 0.1, max_iter 100). **구현 = `harmonypy==0.2.0` 고정. θ·λ·σ와 `max_iter_harmony=100`(원문 A.3의 명시값)을 명시적으로 넘기고, 나머지는 0.2.0 기본값을 쓴다. 실제 수렴 반복 횟수와 실행 장치를 결과에 저장한다. G0가 RxRx1 또는 JUMP-CP에서 어긋날 때만 `max_iter_harmony=10`으로 재실행해 비교한다** (2026-10-14 결정. 2.0.0부터 λ 기본값이 자동 추정이고, 2.1.0은 원 논문보다 늦게 배포됨). batch key: JUMP-CP source(10개), RxRx1 source(11개), RxRx3-core plate(9개). Harmony 후 재정규화 없음
6. Harmony 전(`X_pca`)과 후(`X_pca_harmony`)를 모두 저장

### 5.2 평가

- 대조군 well은 batch integration(scIB)을 제외한 모든 평가에서 뺀다.
- **Replicate mAP:** copairs. positive = 다른 plate의 같은 perturbation well, negative = 같은 plate의 처리군–대조군 쌍. 같은 plate의 처리군–처리군 쌍은 reference-index 기능으로 제외. 처리군 well의 AP 평균 → perturbation별 mAP → macro 평균. 유의성: 순열 10,000회, α = 0.05, Benjamini-Hochberg.
- **scIB:** Batch = iLISI, kBET, BRAS, graph connectivity, PCR comparison. Bio = cLISI, KMeans NMI, KMeans ARI, silhouette label, isolated labels. 각 성분 = 지표의 비가중 평균, Total = 0.4·Batch + 0.6·Bio. batch key는 위와 같고 label key = InChIKey(화합물) 또는 유전자 기호. **Batch = 5개 지표(iLISI, kBET, BRAS, graph connectivity, PCR comparison)의 비가중 평균** (2026-10-14 결정). 원문 Table A2·A3 수치를 검산하면 5개 평균과 일치하므로, A.6 캡션의 4개 표기는 오기로 본다. Harmony 전 평가에서는 PCR comparison = 0이다(기준이 자기 자신). 개별 지표 5개와 PCR 제외 4개 평균도 함께 저장한다. 구현 = scib-metrics 0.5.10. Bio의 NMI·ARI는 KMeans만 쓴다.
- **CP feature prediction:** cp_measure로 세포별 특징 계산(강도·텍스처·granularity·radial distribution·radial Zernike × 채널 × 핵/세포질 마스크, 크기·모양·Zernike·Feret × 구획, 모든 채널 쌍의 Pearson·Manders fold·RWC) → well 평균 → NaN 비율 > 50% 특징과 대조군 분산 0 특징 제거 (원문 기준 데이터셋당 1,358–1,443개). 50차원 임베딩에서 multi-output ridge(α = 1). 임베딩 표준화는 학습 세트에만 fit. 학습 = 비대조군·비held-out well, 평가 = held-out 배치의 비대조군 well. 학습 세트 분산 < 1e-8 특징 제외. median Pearson r 보고.
- **Cross-batch perturbation recall:** 쿼리 = held-out 배치의 비대조군 well, 갤러리 = 나머지 배치의 비대조군 well. 코사인 유사도, macro R@k (k = 1…10).

---

## 6. 참조값 (원문, post-Harmony) [확인]

원문은 구성당 1회 실행이므로 참조값 자체에 seed 변동이 들어 있다. G0의 허용치 0.03은 이를 감안한 값이다.

### 6.1 G0 대상: DINO ViT-B/16

| 데이터셋 | 지표 | C | CD | S | SD |
| --- | --- | --- | --- | --- | --- |
| RxRx1 | R@1 | 0.552 | 0.519 | 0.137 | 0.154 |
| RxRx1 | R@10 | 0.742 | 0.716 | 0.322 | 0.368 |
| RxRx1 | replicate mAP | 0.756 | 0.773 | 0.251 | 0.209 |
| RxRx1 | CP pred. median r | 0.487 | 0.484 | 0.409 | 0.521 |
| RxRx3-core | R@1 | 0.075 | 0.071 | 0.025 | 0.023 |
| RxRx3-core | R@10 | 0.296 | 0.302 | 0.123 | 0.131 |
| RxRx3-core | replicate mAP | 0.626 | 0.611 | 0.469 | 0.484 |
| RxRx3-core | CP pred. median r | 0.217 | 0.215 | 0.283 | 0.302 |
| JUMP-CP | R@1 | 0.054 | 0.051 | 0.029 | 0.034 |
| JUMP-CP | R@10 | 0.156 | 0.158 | 0.108 | 0.130 |
| JUMP-CP | replicate mAP | 0.276 | 0.263 | 0.225 | 0.243 |
| JUMP-CP | CP pred. median r | 0.278 | 0.263 | 0.332 | 0.322 |

### 6.2 전체 인코더 (E7 참조용)

**Cross-batch R@10**

| 데이터셋 | 인코더 | C | CD | S | SD |
| --- | --- | --- | --- | --- | --- |
| RxRx1 | SubCell | 0.676 | 0.653 | 0.409 | 0.369 |
| RxRx1 | DINO | 0.742 | 0.716 | 0.322 | 0.368 |
| RxRx1 | OpenPhenom | 0.568 | 0.564 | 0.258 | 0.162 |
| RxRx3-core | SubCell | 0.313 | 0.300 | 0.100 | 0.117 |
| RxRx3-core | DINO | 0.296 | 0.302 | 0.123 | 0.131 |
| RxRx3-core | OpenPhenom | 0.121 | 0.093 | 0.035 | 0.033 |
| JUMP-CP | SubCell | 0.170 | 0.163 | 0.140 | 0.150 |
| JUMP-CP | DINO | 0.156 | 0.158 | 0.108 | 0.130 |
| JUMP-CP | OpenPhenom | 0.094 | 0.101 | 0.050 | 0.059 |

**Replicate mAP** (CPM = cp_measure 특징, 분할 세포 기준이라 S 열에만 값)

| 데이터셋 | 인코더 | C | CD | S | SD |
| --- | --- | --- | --- | --- | --- |
| RxRx1 | SubCell | 0.787 | 0.843 | 0.330 | 0.290 |
| RxRx1 | DINO | 0.756 | 0.773 | 0.251 | 0.209 |
| RxRx1 | OpenPhenom | 0.239 | 0.233 | 0.122 | 0.077 |
| RxRx1 | CPM | — | — | 0.108 | — |
| RxRx3-core | SubCell | 0.760 | 0.725 | 0.716 | 0.753 |
| RxRx3-core | DINO | 0.626 | 0.611 | 0.469 | 0.484 |
| RxRx3-core | OpenPhenom | 0.312 | 0.296 | 0.136 | 0.127 |
| RxRx3-core | CPM | — | — | 0.011 | — |
| JUMP-CP | SubCell | 0.285 | 0.270 | 0.252 | 0.256 |
| JUMP-CP | DINO | 0.276 | 0.263 | 0.225 | 0.243 |
| JUMP-CP | OpenPhenom | 0.209 | 0.222 | 0.214 | 0.257 |
| JUMP-CP | CPM | — | — | 0.205 | — |

**CP feature prediction (median Pearson r)**

| 데이터셋 | 인코더 | C | CD | S | SD |
| --- | --- | --- | --- | --- | --- |
| RxRx1 | SubCell | 0.464 | 0.455 | 0.485 | 0.511 |
| RxRx1 | DINO | 0.487 | 0.484 | 0.409 | 0.521 |
| RxRx1 | OpenPhenom | 0.526 | 0.546 | 0.474 | 0.500 |
| RxRx3-core | SubCell | 0.183 | 0.189 | 0.233 | 0.247 |
| RxRx3-core | DINO | 0.217 | 0.215 | 0.283 | 0.302 |
| RxRx3-core | OpenPhenom | 0.311 | 0.335 | 0.432 | 0.425 |
| JUMP-CP | SubCell | 0.290 | 0.307 | 0.332 | 0.324 |
| JUMP-CP | DINO | 0.278 | 0.263 | 0.332 | 0.322 |
| JUMP-CP | OpenPhenom | 0.293 | 0.283 | 0.335 | 0.319 |

---

## 7. 작업 순서

1. **환경 구축.** Python 가상환경을 만들고 버전을 `requirements.txt`(또는 `environment.yml`)로 고정한다. 필요한 구성 요소: torch(B200 인식 확인), timm, LoRA 구현(peft 또는 자체 구현), Cellpose-SAM을 지원하는 cellpose 버전, cp_measure, copairs, pycytominer, harmonypy, scanpy, scIB 지표 구현(BRAS 포함 여부), rdkit(ECFP4), transformers(ESM2). 패키지 이름과 버전은 설치하면서 확인한다 [확인 필요].
2. **데이터 다운로드.**
   - RxRx1: https://rxrx.ai 에서 HEPG2-01 ~ HEPG2-11 (6채널, 512×512, well당 2 sites)
   - RxRx3-core: Hugging Face `recursionpharma/rxrx3-core`. `metadata_rxrx3_core.csv`로 CRISPR wells를 고른다. 이미지 키는 뷰어 기준 `<experiment>/Plate<k>/<address>_s<site>_<channel>` 형태로 보인다 [확인 필요].
   - JUMP-CP cpg0016: `s3://cellpainting-gallery/cpg0016-jump` 무인증. `scripts/download_jump.py`가 plate별 `load_data_with_illum.csv`로 선택 well의 모든 site × 형광 5채널 TIFF와 illum `.npy`를 받는다. 원본 약 2.8 TB는 overlay `/tmp/pvil_raw/jump_cp`(휘발성)에 둔다.
3. **분할·crop·QC·샘플링** (`src/pvil/data/`). crop 내 전체 instance mask를 저장한다.
4. **view 생성기** (`src/pvil/data/views.py`): C, S, CD, SD 구현, N과 B는 스텁. 단위 테스트:
   - S에서 마스크 밖 픽셀이 0인지 (DNA 채널은 핵 마스크 기준)
   - CD·SD 패치의 위치, 크기, 강도
   - 같은 세포 view 쌍에 같은 기하 변환이 적용되는지
5. **비이미지 베이스라인 G1–G3** → cross-batch R@10 (지표 오염 하한선).
6. **학습 루프(A1), 임베딩 추출, 후처리, 평가 스위트** 구현.
7. **seed 0으로 12 run** (DINO × 4 view × 3 데이터셋, P-orig) → `results/E0/summary.md`에 §6.1 참조값 대비 차이 표.
8. **G0 판정 (2026-10-16부터 정성 기준 Q1–Q7, `research_plan.md` §9.2).** 파이프라인은 현재 설정으로 고정하고 원문 수치에 맞추려고 바꾸지 않는다(decisions.md 2026-10-16). 아래 진단 순서는 정성 현상이 미달할 때만 쓴다: (a) 세포 수·샘플링·QC → (b) 채널 순서·정규화 → (c) 후처리(sphering fit 대상, PCA, Harmony batch key) → (d) 평가 정의(copairs 설정, 갤러리 구성).
9. **seed 1, 2 추가** (24 run) → A1 seed 표준편차 → δ = 2 × 표준편차를 decisions.md에 고정.
10. **P-strict validation 배치 선정** (학습 배치 중 well 수 중앙값) → decisions.md.
11. **B200 기준 T(GPU-시간) 실측** → progress.md에 기록하고 `docs/research_plan.md` §9.3 예산 갱신을 제안한다.
12. 블로커와 확인 항목 결과를 `docs/research_plan.md` §12 체크리스트에 반영한다.
