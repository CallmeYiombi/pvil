# 진행 상황 (세션 인수인계)

> 세션 시작 시 읽고, 끝내기 전에 갱신한다. 최신 항목을 위에 쓴다.

- 현재 단계: Phase 0 (CP-BG-Bench 재현)
- 마지막 갱신: 2026-10-14

## 세션 시작 방법

```bash
cd /home/genetics/yiombi/PVIL/pvil
source scripts/env.sh          # venv 활성화와 모든 캐시 경로 설정 (~/ 사용 안 함)
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

- 환경을 다시 만들 때: `source scripts/env.sh && uv sync --frozen`
- 환경 점검: `python scripts/check_env.py` → `results/E0/env_check/env_report.json`

## 다음 할 일

1. JUMP-CP 다운로드 완료 확인 (manifest 1,149개, 누락 well 확인)
2. Stage A 분할 완료 확인(RxRx1 44 plates, RxRx3-core 1,584 plates) → JUMP-CP 분할(다운로드 완료 후)
3. RxRx3-core well 역할 정의 확정(사용자) → Stage B(`scripts/make_crops.py`, `--dry-run`으로 수치 먼저 확인): QC → 샘플링 → crop zarr
4. view 생성기 결정 사항: 중심 세포 핵 마스크의 11.5%가 세포 마스크 밖으로 일부 나온다(RxRx3 시험). S view에서 핵을 세포 안으로 자를지 정한다

## 진행 중

- JUMP-CP 원본 다운로드 (아래 실행 중인 작업)

## 완료

- 2026-10-14 Stage B 구현(`scripts/make_crops.py`) + RxRx3-core plate 2개 시험: 3,392 crops, 중앙 픽셀 = 중심 라벨 100%, crop당 세포 중앙값 15, 약 53 KB/crop (`pvil_data/crops/rxrx3_core_partial/`, 시험용)
- 2026-10-14 분할 파이프라인(Stage A) 구현: `src/pvil/data/{readers,segment,qc_sample,crops}.py`, `scripts/segment.py`, 테스트 6개. RxRx3-core plate 2개 시험: 0.62 s/FOV, FOV당 약 90 세포, 윤곽 미리보기 확인 (`pvil_data/seg/rxrx3_core/_preview/`)
- 2026-10-14 RxRx1·RxRx3-core 원본 확인: 형식, 메타데이터-이미지 1:1 커버리지, 핵·세포 채널 라우팅, 색인 (`P0_reproduction.md` §3.1, `scripts/index_raw.py`)
- 2026-10-14 RxRx3-core 원본 다운로드 완료 (17.4 GB, 35 shard SHA-256 일치, `MANIFEST.json`)
- 2026-10-14 JUMP-CP 근사 부분집합 확정(907 화합물, 28,003 wells, plate 1,149) + `scripts/download_jump.py`(source_4 plate 2개 시험: 176 MB/s, 파일 수 일치)
- 2026-10-14 RxRx1 다운로드(49 GB, MD5 일치)와 압축 해제(753,060 파일) 완료
- 2026-10-14 RxRx3-core 메타데이터·LICENSE 확인: 배치 키 plate = 1–9 해결, 라이선스 = Recursion EULA. 원문 well 수와의 차이는 미결 (`P0_reproduction.md` §2-3)
- 2026-10-14 미결 사양 3건 확정 (`docs/decisions.md`): Harmony = harmonypy 0.2.0 + max_iter 100, Cellpose-SAM = `cpsam` v1, scIB Batch = 5개 평균. `src/pvil/eval/harmony.py`(수렴 반복 횟수·장치 기록) + `tests/test_harmony.py`
- 2026-10-14 환경 구축 (`docs/tasks/P0_env_setup.md`): 독립 repo + GitHub push, uv 환경, 패키지 골격, `configs/base_cpbg.yaml`, `scripts/check_env.py`, `tests/test_env.py`(15 passed). 점검 결과는 아래 측정값
- 2026-10-14 연구 계획 수립 (`docs/research_plan.md`)

## 실행 중인 작업

| tmux 세션 | 명령 | 로그 경로 | 시작 시각 |
| --- | --- | --- | --- |
| `pvil_dl_jump` | `python scripts/download_jump.py configs/data/jump_cp.yaml` (1,149 plates, 약 2.8 TB → /tmp/pvil_raw) | `/home/genetics/yiombi/pvil_data/logs/download_jump_cp_20261014_1034.log` | 2026-10-14 10:34 UTC |
| `pvil_seg` (창 s0–s7) | `scripts/segment.py` RxRx1 → RxRx3-core, `--shard i/8`, GPU i%4 | `/home/genetics/yiombi/pvil_data/logs/segment_20261014_1111_shard{0..7}.log` | 2026-10-14 11:11 UTC |

## 측정값

| 항목 | 값 | 측정일 | 비고 |
| --- | --- | --- | --- |
| T (B200, A1 1회 학습) | | | 계획서 추정은 A100 기준 2–5 GPU-시간. E0에서 측정 |
| δ (비열등성 마진) | | | A1 seed 표준편차 × 2, E1 전에 고정 |
| ViT-B/16 DINOv3 + LoRA(block 한정, 학습 파라미터 1,179,666), bf16 fwd+bwd | 0.105 s/step (batch 64, 약 609 img/s), peak 7.03 GB | 2026-10-14 | GPU 0, torch.compile 없음, 다른 작업과 GPU 공유 중에 잰 참고값 |
| bf16 matmul (4096²) | GPU0 16 / GPU1 180 / GPU2 79 / GPU3 371 TFLOPS | 2026-10-14 | 1차 측정(GPU0 7 / GPU1 52 / GPU2 470 / GPU3 238)과 크게 다르다. 사용량 채우기 작업이 도는 상태라 우리 작업이 실제로 받을 성능과 다를 수 있다 |
| Cellpose-SAM `cpsam` 추론 (256², 합성) | 0.55 s, 가중치 bf16, SHA-256 일치 | 2026-10-14 | 실데이터 분할 비용은 Phase 0 데이터 처리에서 따로 측정 |
| Cellpose-SAM 2회 + 매칭 (RxRx3-core 512²) | 0.62 s/FOV (plate 80 FOV ≈ 50 s) | 2026-10-14 | GPU 1장 1프로세스, 공유 상태 |
| Cellpose-SAM 2회 (JUMP 1080²) | 1.7 s/FOV (핵 0.74 + 세포 0.94) | 2026-10-14 | 매칭 제외, 공유 상태 |

## 이슈·블로커

1. **GPU 공유:** `nvidia-smi`에 다른 프로세스가 여럿 보인다. 그중 `/opt/conda/bin/python3`(multiprocessing spawn 4개, GPU당 40–143 GB, 2026-09 말부터 실행)은 다른 사람이 사용량을 채우려고 올린 작업으로, 우리가 GPU를 쓰면 자원이 유동적으로 우리에게 할당된다(사용자 확인, 2026-10-14). 따라서 점유 메모리·사용률만 보고 GPU가 부족하다고 판단하지 않는다. 그 밖의 학습 작업(다른 사용자의 train_val.py, run_brca.py 등)은 실제 경합 대상이다. T를 잴 때는 측정 시점의 GPU 점유 상태를 함께 기록한다
2. **HF CDN 속도:** 단일 연결이 약 45 KB/s로 떨어지는 파일이 있다(cpsam_v2 1.23 GB). 64분할 병렬 range 다운로드 + SHA-256 확인으로 해결
3. CP-BG-Bench 아티팩트 미공개. 저자 문의는 하지 않기로 함 (2026-10-14)
4. JUMP-CP는 근사 부분집합(화합물 907 vs 원문 1,466). G0는 "근사" 표기
5. RxRx3-core well 수 불일치: 원문 104,166 wells / 735 perturbation, 메타데이터 CRISPR 126,900 / 736
6. RxRx1 HepG2 well 수 불일치: 원문 12,581 wells / 1,108 perturbation, 메타데이터 13,547 wells(처리군 12,179 + 양성대조군 1,318 + 음성대조군 50) / siRNA 라벨 1,139. 분할·QC 뒤 재비교
