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
| 2026-10-14 | scIB 구현 = scib-metrics 0.5.10 (BRAS, PCR comparison 포함). Bio 지표는 KMeans NMI/ARI만 쓰고 기본값의 Leiden은 끈다 | 사양 §5.2. Batch를 PCR 포함 5개로 할지 4개로 할지는 미결 [확인 필요] | 평가 |
| 2026-10-14 | JAX는 CPU로 고정(`JAX_PLATFORMS=cpu`) | torch CUDA와 충돌 방지. scIB 계산량은 CPU로 충분하다고 판단(실데이터에서 재확인) | 평가 |
