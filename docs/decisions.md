# 결정 로그

> 설계·프로토콜·데이터 결정을 날짜순으로 기록한다. 사전 등록 항목을 바꿀 때는 반드시 사유를 적고 사용자 승인을 받는다.

| 날짜 | 결정 | 근거 | 영향 범위 |
| --- | --- | --- | --- |
| 2026-10-14 | 주 지표 = held-out 배치 cross-batch R@10 (post-Harmony) | CP-BG-Bench 헤드라인 표와 일치하고 배치 일반화를 직접 측정 | 전체 |
| 2026-10-14 | 주 프로토콜 = P-strict(별도 validation 배치). P-orig는 재현·부록용 | 원 논문은 held-out 배치를 early stopping에도 사용 | 전체 |
| 2026-10-14 | 보고용 결과는 모두 3 seeds | 원 논문의 단일 실행 한계 | 전체 |
| 2026-10-14 | 비열등성 마진 δ = A1 seed 표준편차 × 2, E1 실행 전 고정 | 사전 등록 | E1 이후 |
| 2026-10-14 | 방법 개발 시작 데이터셋 = RxRx3-core | Harmony 전후 recall 차이 ≤ 0.003으로 학습 단계 개입 여지가 가장 큼 | Phase 2 |
| 2026-10-14 | 데이터 저장 = 150×150 crop + instance mask, view는 학습 중 생성 | 모든 view 저장 시 데이터셋당 약 1.8 TB, crop+mask는 약 200 GB | 데이터 |
| 2026-10-14 | 저장소 = `PVIL/pvil/` 독립 git repo, 원격 `CallmeYiombi/pvil`. 이 repo에만 쓰는 gh 인증은 `/home/genetics/yiombi/.gh_pvil`(credential helper는 repo `.git/config`) | 바깥 MorphoShift repo와 이력 분리. `~/` 사용 금지 | 전체 |
| 2026-10-14 | 저장 위치 = `/home/genetics/yiombi` 아래만. venv `venv_pvil`, 캐시 `hf_cache`(기존 공용), 데이터 `pvil_data`(repo `data/`는 심볼릭 링크). 경로는 `scripts/env.sh`와 `configs/paths.yaml`로 관리 | `~/`(/home/jovyan) 사용 금지(사용자 지시). gpfs는 서버 재생성 뒤에도 남음 | 전체 |
| 2026-10-14 | 환경 = uv + Python 3.11.15(uv 관리) + torch 2.9.1+cu130. 버전은 `uv.lock`과 `requirements.txt`로 고정 | B200(sm_100)은 CUDA 12.8 이상 필요. base conda에서 2.9.1+cu130 동작 확인 | 전체 |
| 2026-10-14 | LoRA 구현 = peft 0.21.2. target_modules = 정규식 `blocks\.\d+\.(attn\.(qkv\|proj)\|mlp\.(fc1\|fc2))` (48개 모듈, 학습 파라미터 1,179,648) | 이름 목록 `[qkv, proj, fc1, fc2]`을 쓰면 `patch_embed.proj`(Conv2d)에도 LoRA가 붙는다(+12,288). 사양은 "모든 transformer block의 attention·MLP projection" | 모델 |
| 2026-10-14 | scIB 구현 = scib-metrics 0.5.10 (BRAS, PCR comparison 포함). Bio의 NMI/ARI는 KMeans만 쓰고 기본값의 Leiden은 끈다 | 사양 §5.2 | 평가 |
| 2026-10-14 | JAX는 CPU로 고정(`JAX_PLATFORMS=cpu`) | torch CUDA와 충돌 방지. scIB 계산량은 CPU로 충분하다고 판단(실데이터에서 재확인) | 평가 |
| 2026-10-14 | **Harmony = `harmonypy==0.2.0` 고정.** θ = 2, λ = 1, σ = 0.1, `max_iter_harmony` = 100을 명시적으로 넘기고, 나머지(max_iter_kmeans 20, epsilon_cluster 1e-5, epsilon_harmony 1e-4, device 자동)는 0.2.0 기본값. 실제 수렴 반복 횟수와 장치를 결과에 저장. G0가 RxRx1 또는 JUMP-CP에서 어긋날 때만 max_iter_harmony = 10으로 재실행해 비교 | 2.0.0(2026-04-26)부터 λ 기본값이 자동 추정으로 바뀌었다. 2.1.0(2026-10-06)은 원 논문(2026-09-14)보다 늦게 배포됐다. 0.2.0은 2026-01-09 배포. max_iter 100은 원문 A.3의 명시값. 참고: 0.2.0은 PyTorch 구현이라 CPU와 GPU 결과가 약간 다르다(합성 데이터에서 최대 절대차 0.015, GPU끼리 반복 3e-6) | 후처리 전체 (사용자 결정) |
| 2026-10-14 | **Cellpose-SAM 가중치 = `cpsam`(v1, HF `mouseland/cellpose-sam`, 2025-04-29 업로드, SHA-256 `e1440429…abe2`) 고정.** 모든 실험에서 같은 마스크를 쓴다. `cpsam_v2`는 E8 마스크 강건성 실험의 추가 조건으로만. 정밀도는 cellpose 기본값(bf16)을 쓰고 로그에 기록 | 원문이 인용한 Cellpose-SAM 프리프린트의 모델. cellpose 4.2.1.1의 기본값은 `cpsam_v2`(2026-05)라 명시적으로 지정해야 한다 | 데이터·E8 (사용자 결정) |
| 2026-10-14 | **scIB Batch = iLISI, kBET, BRAS, graph connectivity, PCR comparison 5개의 비가중 평균.** Harmony 전 평가에서는 PCR comparison = 0(기준이 자기 자신). 개별 지표 5개와 PCR 제외 4개 평균도 함께 저장 | 원문 Table A2·A3 수치를 검산하면 5개 평균과 일치한다. A.6 캡션의 4개 표기는 오기로 본다 | 평가 (사용자 결정) |
