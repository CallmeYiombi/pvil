# Phase 0 환경 구축 계획 (P0 §7-1)

## Context

`docs/progress.md` 다음 할 일 2번이자 P0 작업 지시서 §7-1에 해당한다. 이후 모든 단계(분할, view, 학습, 평가)가 쓸 재현 가능한 환경과 저장소 골격을 만든다.

현재 상태(확인함):
- PVIL 문서는 `PVIL/pvil/`에 있다(CLAUDE.md, docs/, .claude/rules/, .gitignore). 독립 git repo가 아니고, 바깥 `PVIL/`의 MorphoShift repo(main, 미커밋 삭제 다수) 안에 있다. `PVIL/CLAUDE.md`는 `pvil/CLAUDE.md`와 내용이 같다.
- GPU: B200 × 4 (183 GB), 드라이버 580.95, CUDA 13.0. conda base에서 torch 2.9.1+cu130 동작. `uv`, `mamba`, `tmux` 있음. CPU 288코어, RAM 2.2 TB.
- 저장소: gpfs `/home/genetics` 5.6 TB 여유(영구), overlay `/` 38 TB(휘발성).
- 기존 `PVIL/.venv`는 MorphoShift용(awscli만)이라 재사용하지 않는다.

사용자 결정: (1) `pvil/`을 독립 git repo로 만든다. (2) 데이터는 gpfs에 두고 repo의 `data/`를 심볼릭 링크로 연결한다.

## 단계

### 0. 저장 위치 원칙: `~/`(/home/jovyan) 사용 금지
- 모든 산출물(venv, Python 인터프리터, 패키지·모델 캐시, 데이터, 체크포인트)은 `/home/genetics/yiombi/` 아래에 둔다.
- 기존 관례를 따른다. 캐시는 이미 설정된 `/home/genetics/yiombi/hf_cache`(`HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`)를 재사용하고, venv는 `venv_*` 관례에 맞춰 `/home/genetics/yiombi/venv_pvil`에 만든다.
- 기본값이 `~`로 가는 경로를 `scripts/env.sh`에서 모두 덮어쓴다:
  - `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`(uv가 받는 Python 3.11), `UV_PROJECT_ENVIRONMENT=/home/genetics/yiombi/venv_pvil`
  - `XDG_CACHE_HOME`, `CELLPOSE_LOCAL_MODELS_PATH`(기본값 `~/.cellpose`), `TRITON_CACHE_DIR`(기본값 `~/.triton`), `TORCHINDUCTOR_CACHE_DIR`, `MPLCONFIGDIR`
  - 위 경로는 모두 `/home/genetics/yiombi/hf_cache/` 아래에 둔다.
- `check_env.py` 마지막에 `~/.cache`, `~/.cellpose`, `~/.triton`, `~/.local/share/uv`에 이번 실행으로 새로 생긴 파일이 없는지 확인한다.
- git 사용자 정보는 `~/.gitconfig`에 의존하지 않고 repo의 `.git/config`에 직접 넣는다(MorphoShift repo와 같은 `CallmeYiombi <mynameisgithub@gmail.com>`).
- 계획 문서는 `docs/tasks/P0_env_setup.md`로 repo 안에 복사한다(plan 파일은 `~/.claude`에 있으므로).

### 1. 저장소 분리와 원격
- `pvil/`에서 `git init -b main` 실행 후 기존 문서를 첫 커밋으로 넣는다.
- 원격 `origin = https://github.com/CallmeYiombi/pvil.git`(현재 비어 있음 확인). 커밋 단위마다 push한다.
- 커밋 메시지·PR에는 Claude 관련 문구, Co-Authored-By, 워터마크를 넣지 않는다. 이 규칙은 `CLAUDE.md` 작업 방식 절에 한 줄로 추가해 다음 세션에도 유지한다.
- 바깥 MorphoShift repo는 건드리지 않는다. `pvil/`은 `PVIL/.git/info/exclude`(로컬 전용, 커밋 안 됨)에 추가해 untracked 표시만 없앤다.
- 바깥 `PVIL/CLAUDE.md` 사본은 그대로 둔다. 정리 여부는 사용자가 정한다. 앞으로 작업 디렉터리는 `pvil/`이다.

### 2. 데이터 루트 (gpfs)
- `/home/genetics/yiombi/pvil_data/{raw,crops,ckpt,embeddings}`를 만들고, `pvil/data`를 이 경로에 대한 symlink로 둔다(`data/`는 이미 .gitignore에 있음).
- 모델 가중치(ESM2, DINOv3, Cellpose-SAM)는 §0의 공용 `hf_cache`에 둔다. 별도 cache 디렉터리는 만들지 않는다.

### 3. Python 환경 (uv, Python 3.11)
- `pyproject.toml`: `src/pvil` 패키지를 editable로 설치하고 의존성 그룹을 나눈다. 버전은 `uv.lock`으로 고정하고, 지시서가 요구하는 `requirements.txt`도 `uv export`로 만든다.
- torch는 PyTorch cu130 인덱스에서 받은 2.9.1이다(base에서 B200 동작 확인됨).
- 구성 요소와 확인 항목(설치하면서 실제 버전을 확정):

| 용도 | 패키지 | 확인할 것 |
| --- | --- | --- |
| 백본 | timm | `vit_base_patch16_dinov3.lvd1689m`이 있는 버전 |
| LoRA | peft | timm ViT의 qkv/proj/fc1/fc2를 target_modules로 쓸 수 있는지. 안 되면 자체 구현으로 바꾸고 decisions.md에 기록 |
| 분할 | cellpose ≥ 4 | Cellpose-SAM(`cpsam`) 기본 모델 로드, GPU 추론 |
| CP 특징 | cp_measure | 설치 가능 여부, 의존성 충돌 |
| 평가 | copairs, pycytominer, harmonypy, scanpy, anndata | pycytominer의 ZCA-cor 함수 위치 |
| scIB | scib-metrics | BRAS·PCR comparison 포함 여부. JAX는 우선 CPU 빌드(torch CUDA와 충돌 방지) |
| perturbation 임베딩 | rdkit, transformers | Morgan count fp, chirality 옵션, `facebook/esm2_t33_650M_UR50D` 로드 |
| 데이터 | huggingface_hub, boto3, pyarrow/polars, zarr, tifffile, imagecodecs(JPEG 2000) | RxRx3-core JP2 디코딩 |
| 기타 | omegaconf, pytest | — |

- 설치가 안 되거나 충돌하는 패키지는 임의로 대체하지 않고 보고한 뒤 결정을 받는다.

### 4. 저장소 골격 (CLAUDE.md 목표 구조)
- `src/pvil/{data,models,losses,eval,baselines}/__init__.py`, `scripts/`, `tests/`, `configs/`, `results/`. 디렉터리와 빈 모듈만 만들고 구현은 이후 단계에서 한다.
- `configs/paths.yaml`: data_root 등 경로. `configs/base_cpbg.yaml`: §4 학습 사양을 값만 옮긴다. 원문에 없는 값은 `TODO`로 남긴다.
- `src/pvil/eval/run_info.py`: git hash, config, seed, 패키지 버전을 기록하는 헬퍼. 규칙 4(모든 run 기록)의 기반이다.

### 5. 환경 점검 스크립트 `scripts/check_env.py`
GPU 4장 각각에서 다음을 확인하고 결과를 JSON으로 저장한다(`results/E0/env_check/env_report.json`, 커밋하지 않음).
1. torch/CUDA 버전, 장치명, bf16 matmul
2. timm DINOv3 로드 → 6채널 1×1 conv 어댑터 → 224 입력 forward, peft LoRA 적용 후 학습 가능 파라미터 수, bf16 forward+backward 1 step의 시간과 메모리
3. Cellpose-SAM을 합성(또는 공개 샘플) 이미지에 GPU로 추론
4. rdkit ECFP4 count fp(2048), ESM2 토큰 평균 1280차원 1개
5. harmonypy, pycytominer ZCA, copairs, scib-metrics를 랜덤 데이터로 각각 1회 호출

`tests/test_env.py`에는 import와 B200 인식 같은 가벼운 검사만 넣는다.

### 6. 기록과 커밋
- `docs/decisions.md`: 환경 관리 방식(uv), torch/CUDA 버전, LoRA 구현 선택, scIB 구현 패키지와 BRAS 포함 여부, 데이터 루트 경로.
- `docs/progress.md`: 완료 항목, ViT-B LoRA step 시간(T 추정의 첫 근거, 정식 T는 E0에서 측정), 새로 생긴 이슈.
- 커밋은 의미 단위로 나눈다: (a) 기존 문서 첫 커밋, (b) 환경과 골격, (c) 점검 스크립트와 기록. 각 커밋 뒤 `origin main`에 push한다.
- `docs/progress.md`는 마지막에 한 번이 아니라 단계가 끝날 때마다 갱신한다.
- 결과 경로는 `results/E0/env_check/`로 바꾸고 `run_info` 헬퍼는 `src/pvil/eval/run_info.py`에 둔다(목표 구조 안).

## 범위 밖 (다음 단계)
- 데이터 다운로드(RxRx3-core 메타데이터 확인이 다음 순서), 분할·crop, view 생성기
- 블로커 1·2(저자 문의)는 사용자가 진행

## 검증
- `source scripts/env.sh && python scripts/check_env.py`: 모든 항목 PASS, GPU 4장 인식
- `pytest -q tests/` 통과
- `uv sync --frozen`으로 새 venv를 다시 만들어 같은 버전이 설치되는지 확인(재현성)
- `git -C pvil log`에 커밋 3개, 바깥 MorphoShift repo의 `git status`는 작업 전과 같음(pvil/만 사라짐)
