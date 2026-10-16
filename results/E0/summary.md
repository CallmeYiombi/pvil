# E0 요약: CP-BG-Bench 재현 (DINO ViT-B/16, seed 0, protocol: P-orig)

`scripts/summarize_e0.py`가 생성한다. 손으로 고치지 않는다.

- 참조값 = 원문(P0_reproduction.md §6.1), post-Harmony. G0 = R@10과 replicate mAP가 |Δ| ≤ 0.03.
- seed 0 단일 실행이다. 원문도 구성당 1회 실행이라 양쪽 모두 seed 변동을 포함한다.
- CP feature prediction과 scIB는 아직 측정하지 않았다(—).

## 참조값 대비 (post-Harmony)

| 데이터셋 | view | 지표 | 원문 | 우리 | Δ | G0 |
| --- | --- | --- | --- | --- | --- | --- |
| rxrx1 | C | R@1 | 0.552 | 0.666 | +0.114 |  |
| rxrx1 | C | R@10 | 0.742 | 0.858 | +0.116 | 미달 |
| rxrx1 | C | mAP | 0.756 | 0.993 | +0.237 | 미달 |
| rxrx1 | C | CP r | 0.487 | — | — |  |
| rxrx1 | S | R@1 | 0.137 | 0.309 | +0.172 |  |
| rxrx1 | S | R@10 | 0.322 | 0.571 | +0.249 | 미달 |
| rxrx1 | S | mAP | 0.251 | 0.949 | +0.698 | 미달 |
| rxrx1 | S | CP r | 0.409 | — | — |  |
| rxrx3_core | C | R@1 | 0.075 | 0.154 | +0.079 |  |
| rxrx3_core | C | R@10 | 0.296 | 0.458 | +0.162 | 미달 |
| rxrx3_core | C | mAP | 0.626 | 0.951 | +0.325 | 미달 |
| rxrx3_core | C | CP r | 0.217 | — | — |  |
| rxrx3_core | S | R@1 | 0.025 | 0.108 | +0.083 |  |
| rxrx3_core | S | R@10 | 0.123 | 0.361 | +0.238 | 미달 |
| rxrx3_core | S | mAP | 0.469 | 0.949 | +0.480 | 미달 |
| rxrx3_core | S | CP r | 0.283 | — | — |  |

## Harmony 전 (참고)

| 데이터셋 | view | R@1 | R@10 | mAP |
| --- | --- | --- | --- | --- |
| rxrx1 | C | 0.499 | 0.755 | 0.987 |
| rxrx1 | S | 0.164 | 0.372 | 0.930 |
| rxrx3_core | C | 0.151 | 0.459 | 0.945 |
| rxrx3_core | S | 0.110 | 0.374 | 0.944 |

## 학습 비용 T (상한)

공유 GPU(1·2번, 다른 작업과 동시 실행)에서 잰 값이라 상한이다. 측정 중 점유는 각 run의 `gpu_log.csv`.

| 데이터셋 | view | epoch (best) | step | samples/s | 학습 h | val h | T (GPU-h, 벽시계) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rxrx1 | C | 15 (10) | 25,155 | 315 | 11.37 | 0.38 | 12.03 |
| rxrx1 | S | 16 (11) | 26,832 | 302 | 12.62 | 0.42 | 13.35 |
| rxrx3_core | C | 24 (19) | 56,328 | 543 | 14.74 | 0.81 | 15.82 |
| rxrx3_core | S | 25 (25) | 58,675 | 454 | 18.40 | 0.92 | 19.63 |

## 원문과의 차이

원문과 같은 원본 데이터셋을 쓰지만, 원문이 밝히지 않은 부분을 우리가 정해 채운 재구현이다. 아래 차이는 모든 방법에 똑같이 적용되므로 방법 간 비교(E1 이후)에는 영향을 주지 않고, 원문 수치와의 비교(G0)를 해석할 때만 고려한다. G0가 어긋나면 이 표가 진단 순서의 출발점이다.

### 데이터 규모

| 데이터셋 | 원문 wells | 원문 perturbation | 우리 wells (제외 역할 뺀 정의) | 우리 perturbation | crop이 있는 wells | crops | 상태 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rxrx1 | 12,581 | 1,108 | 12,229 | 1,108 | 12,223 | 941,362 | Stage B 완료 |
| rxrx3_core | 104,166 | 735 | 81,314 | 734 | 80,879 | 1,352,958 | Stage B 완료 |
| jump_cp | 28,003 | 1,466 | 28,003 | 907 | 27,722 | 943,614 | Stage B 완료 |

### 원문에 없거나 원문과 다를 수 있는 결정 (`docs/decisions.md`에서 자동 수집)

| 날짜 | 구분 | 내용 | 영향 범위 |
| --- | --- | --- | --- |
| 2026-10-14 | 원문 차이 | LoRA 구현 = peft 0.21.2. target_modules = 정규식 `blocks\.\d+\.(attn\.(qkv\|proj)\|mlp\.(fc1\|fc2))` (48개 모듈, 학습 파라미터 1,179,648) | 모델 |
| 2026-10-14 | 원문 차이 | Harmony = `harmonypy==0.2.0` 고정. θ = 2, λ = 1, σ = 0.1, `max_iter_harmony` = 100을 명시적으로 넘기고, 나머지(max_iter_kmeans 20, epsilon_cluster 1e-5, epsilon_harmony 1e-4, device 자동)는 0.2.0 기본값. 실제 수렴 반복 횟수와 장치를 결과에 저장. | 후처리 전체 (사용자 결정) |
| 2026-10-14 | 원문 차이 | Cellpose-SAM 가중치 = `cpsam`(v1, HF `mouseland/cellpose-sam`, 2025-04-29 업로드, SHA-256 `e1440429…abe2`) 고정. 모든 실험에서 같은 마스크를 쓴다. `cpsam_v2`는 E8 마스크 강건성 실험의 추가 조건으로만. 정밀도는 cellpose 기본값(bf16)을 쓰고 로그에 기록 | 데이터·E8 (사용자 결정) |
| 2026-10-14 | 원문 차이 | scIB Batch = iLISI, kBET, BRAS, graph connectivity, PCR comparison 5개의 비가중 평균. Harmony 전 평가에서는 PCR comparison = 0(기준이 자기 자신). 개별 지표 5개와 PCR 제외 4개 평균도 함께 저장 | 평가 (사용자 결정) |
| 2026-10-14 | 원문 차이 | JUMP-CP 근사 부분집합: source_2–11(10개), plate 유형 COMPOUND·TARGET2. 화합물 = TARGET2 비대조군 화합물 ∪ 10개 source 중 6개 이상에 나오는 화합물(907개, 양성대조군·empty 제외). | JUMP-CP 전체 (사용자 결정) |
| 2026-10-14 | 원문 차이 | 분할 구현: Cellpose-SAM `cpsam`을 FOV마다 2회 실행(핵 = 핵 채널 단독, 세포 = [세포 채널, 핵 채널], `channel_axis=-1`, 그 밖에는 기본값). 핵-세포 매칭 = 겹침 픽셀 수 기준 상호 최대 겹침 쌍만 유지. 채널 라우팅: RxRx 핵 w1/ch1, 세포 w3/ch3; JUMP 핵 DNA, 세포 AGP | 데이터 |
| 2026-10-14 | 원문 없음 | `fov_n_cells` = FOV의 매칭된 세포 수(경계 제외·QC 이전). CD/SD view 강도와 G1 베이스라인에 쓴다 | view, G1 |
| 2026-10-14 | 원문 없음 | JUMP-CP illumination correction은 적용하지 않는다(illum `.npy`는 보관). crop은 원본 dtype(RxRx uint8, JUMP uint16)으로 저장하고 강도 정규화는 view 단계에서 한다 | 데이터·view |
| 2026-10-14 | 원문 없음 | 대조군 샘플링 = well 단위: 대조군 well마다 처리군 well당 평균 샘플 세포 수만큼. 처리군은 (perturbation, source)당 100개(JUMP·RxRx1), 200개(RxRx3-core). JUMP는 선택 well의 모든 site를 분할 | 데이터·평가 (사용자 결정) |
| 2026-10-14 | 원문 차이 | well 역할: RxRx1 perturbation = treatment siRNA(1,108, 원문과 일치), 대조군 = negative_control(EMPTY), positive_control 제외. RxRx3-core는 기본안(Query guides / Intron controls)으로 두고 Stage A 후 확정 [확인 필요] | 데이터 |
| 2026-10-14 | 원문 차이 | RxRx3-core 역할 확정: perturbation = `well_type_label == "Query guides"`(76,629 wells, 유전자 734개), 라벨 = 유전자 기호(guide 4–10개, 대부분 6개를 유전자로 묶음). 음성 대조군 = Intron controls(4,685 wells, sphering fit·copairs negative). | RxRx3-core 전체 (사용자 결정) |
| 2026-10-14 | 원문 없음 | RxRx1 siRNA perturbation 임베딩 = siRNA별 학습 가능 임베딩 표 → perturbation head(SiLU MLP). 대조군(EMPTY) = 전체 siRNA 임베딩 평균 | RxRx1 학습 (사용자 결정) |
| 2026-10-14 | 원문 없음 | ESM2 입력: 유전자 기호 → UniProt 인간 참조 프로테옴(UP000005640) reviewed 정규 서열. 1,022 aa를 넘으면 겹치는 1,022 aa 창으로 나눠 창별 토큰 평균을 토큰 수로 가중 평균 | 유전자 임베딩 (사용자 결정) |
| 2026-10-14 | 원문 없음 | 1 epoch = ⌈학습 crop 수 / 512⌉ step. 최대 25 epoch, early stopping patience 5는 epoch 단위 | 학습 (사용자 결정) |
| 2026-10-14 | 원문 없음 | 미니배치 512 = perturbation 63개 × 8 + 대조군 8개(64 슬롯 중 1개를 대조군에 고정). 대조군은 같은 대조군 라벨을 공유하는 positive 집합 | 학습 (사용자 결정) |
| 2026-10-14 | 원문 없음 | A1 구현 세부값 : 평가 임베딩 = image projection head 출력 128차원(research_plan.md "CP-BG-Bench의 128차원과 맞춤"). head hidden = 입력 차원, skip = 선형 사영. 어댑터 1×1 conv 초기값 = 1/C, affine은 ImageNet mean/std로 초기화. 입력 = uint8/255. | 학습·평가 (사용자 승인: E0 계획) |
| 2026-10-15 | 원문 차이 | ViT activation checkpointing 켬(`train.grad_checkpointing: true`), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 학습·T |
| 2026-10-15 | 원문 없음 | CD/SD 밀도 패치는 224×224 입력 해상도에 그린다(22×22 px, 모서리에서 7 px 안쪽, 4개, 모든 채널). 순서 = 리사이즈 → 밝기·대비 → C/S 마스킹 → 패치 → 기하 증강 | view, E0 CD/SD |
| 2026-10-15 | 원문 차이 | CD/SD 패치 강도 = round((fov_n_cells − min) / (max − min) × 255) / 255. min·max는 학습 배치 세포에서만 구한다 | view, E0 CD/SD |
| 2026-10-15 | 원문 없음 | CD/SD 패치는 기하 증강 전에 그린다(패치 → flip·회전). 0–90° 연속 회전에서 학습 샘플의 약 52%(22–68°)는 패치가 화면 밖으로 밀려 사라지고, 0–90° 균등일 때 패치 픽셀은 평균 29%만 남는다(합성 측정). 평가에는 증강이 없어 영향 없음 | view, E0 CD/SD, E2 |

### 계산 환경 (run별 `run_info.json`)

원문의 하드웨어와 패키지 버전은 확인되지 않아 비교하지 않고, 우리 값만 기록한다. GPU는 다른 작업과 공유했다.

| 데이터셋 | view | GPU | git | 패키지 |
| --- | --- | --- | --- | --- |
| rxrx1 | C | NVIDIA B200 | 0bf56f99da | torch 2.9.1+cu130, timm 1.0.30, peft 0.21.2, harmonypy 0.2.0, copairs 0.5.5, pycytominer 1.7.1, cellpose 4.2.1.1 |
| rxrx1 | S | NVIDIA B200 | 0bf56f99da | torch 2.9.1+cu130, timm 1.0.30, peft 0.21.2, harmonypy 0.2.0, copairs 0.5.5, pycytominer 1.7.1, cellpose 4.2.1.1 |
| rxrx3_core | C | NVIDIA B200 | 6116fcfdea | torch 2.9.1+cu130, timm 1.0.30, peft 0.21.2, harmonypy 0.2.0, copairs 0.5.5, pycytominer 1.7.1, cellpose 4.2.1.1 |
| rxrx3_core | S | NVIDIA B200 | 6116fcfdea | torch 2.9.1+cu130, timm 1.0.30, peft 0.21.2, harmonypy 0.2.0, copairs 0.5.5, pycytominer 1.7.1, cellpose 4.2.1.1 |

- seed 변동: 원문과 이 표 모두 구성당 1회 실행이라, 차이가 seed 변동인지 구분할 수 없다. seed 1, 2 추가 후(P0 작업 9) 다시 판단한다.
