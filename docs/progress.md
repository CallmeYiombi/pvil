# 진행 상황 (세션 인수인계)

> 세션 시작 시 읽고, 끝내기 전에 갱신한다. 최신 항목을 위에 쓴다.

- 현재 단계: Phase 0 (CP-BG-Bench 재현)
- 마지막 갱신: 2026-10-15 12:30 UTC (E0 4 run 오류 없음. RxRx1 C는 ep 15에서 early stopping(patience 5, best val R@1 0.345 @ ep 10) → 학습 종료 후 후속 단계 진행 중. RxRx1 S ep 15 진행, best 0.063(ep 11). RxRx3-core C ep 7 진행, best 0.117(ep 6) · S ep 6 진행, best 0.048(ep 5), 아직 상승 중·epoch당 약 53분. 처리량 run당 약 380–540 samples/s)
- 사용자 결정(2026-10-15): 학습 시간이 길어도 E0은 계획대로 전체(36 run) 진행한다. 축소안(G0용 6 run)은 기각
- **GPU는 1, 2번만 쓴다**(사용자 지시 2026-10-14). `scripts/env.sh`가 기본값 `CUDA_VISIBLE_DEVICES=1,2`를 설정한다

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

1. **세션 시작 시 먼저 확인:** `tmux ls`, 아래 "실행 중인 작업"의 로그(Traceback, OOM). E0 run은 `scripts/run_e0_job.sh`가 GPU 여유(≥ 25 GB) 확인 후 시작하고 OOM이면 최대 3회 재시도한다. 여유 확인: `python3 scripts/gpu_free_gb.py 1`
2. E0 4 run이 끝나면(train.py → evaluate.py 자동 실행): `python scripts/summarize_e0.py` → `results/E0/summary.md` 커밋, T·처리량·GPU 점유를 아래 측정값에 기록, held-out 평가 실행 기록을 decisions.md에 남긴다
3. JUMP 분할 완료(2026-10-15 06:59). 원본이 overlay `/tmp/pvil_raw/`라 서버 재생성 전에 Stage B를 끝내야 한다. 다만 지금 cgroup 594/597 GiB(대부분 page cache)라 RxRx1 run이 끝난 뒤 시작 권장: `preview_seg.py` → `make_crops.py --dry-run` → Stage B → `cache_crops.py` → ECFP4 임베딩(아직 없음) → E0 JUMP C/S
4. 결정 완료(2026-10-15): S-clip은 E2 민감도 분석에 포함, JUMP 부분집합은 현행 907개 유지
5. 처리량 개선 후보(미승인, 결과 수치 불변): 메모리 여유가 있으면 activation checkpointing 끄기(약 +27%), torch.compile, step마다의 `loss.item()` 동기화 제거. 실행 중인 run에는 적용하지 않는다. 비교용으로 경합 없는 처리량을 잴 기회가 생기면 측정
6. CD/SD view 구현 완료(2026-10-15, 224 기준, `configs/E0/dino_{cd,sd}_{rxrx1,rxrx3_core}.yaml` 추가). **사용자 결정 대기:** 0–90° 연속 회전 증강이 모서리 패치를 밀어낸다(합성 측정: 0–90° 균등일 때 패치 픽셀 평균 29%만 남고, 학습 샘플의 52%(약 22–68°)는 패치가 완전히 사라진다). 평가에는 증강이 없어 영향이 없다. 선택지: (a) 현행 유지(패치 → 회전) (b) CD/SD만 패치를 기하 증강 뒤에 그린다. CD/SD run 시작 전에 정한다
7. CP feature prediction(cp_measure 특징)과 scIB 평가 구현

## 진행 중

- E0 학습 4 run: RxRx1 C/S, RxRx3-core C/S (아래 실행 중인 작업)

## 완료

- 2026-10-15 06:59 JUMP-CP Stage A 분할 완료(1,149 plates, `pvil_data/seg/jump_cp`). `pvil_wait_rxrx3`가 RxRx3-core E0 C/S를 GPU 1·2에 띄움
- 2026-10-15 E0 파이프라인 구현: view(C, S; CD/SD는 TODO), 샘플러(63×8 + 대조군 8, 배치 층화, held-out 가드), A1 모델, multi-positive 손실, `scripts/{train,evaluate,summarize_e0,cache_crops,make_pert_embeddings}.py`, `scripts/run_e0_tmux.sh`, 테스트 7개(전체 36 passed). 스모크 학습·평가 확인
- 2026-10-15 Stage B 완료: RxRx1 941,362 crops(QC 통과율 experiment별 0.74–0.94, HEPG2-05·09가 낮음), RxRx3-core 1,352,958 crops(plate별 0.89–0.90). memmap 캐시 `/tmp/pvil_cache/{rxrx1,rxrx3_core}`
- 2026-10-14 ESM2 유전자 임베딩 734개(`pvil_data/pert_emb/rxrx3_core_esm2.npz`, 1,022 aa 초과 105개는 창 평균, C7orf26→INTS15·MPP5→PALS1은 NCBI 동의어로 해결)
- 2026-10-14 핵 overflow 감사(`scripts/audit_nucleus_overflow.py`, held-out 배치 제외): 핵이 세포 밖으로 1픽셀 이상 나온 세포 RxRx3-core 16.7%·RxRx1 32.0%, 핵 면적의 5% 넘게 나온 세포 4.1%·7.1%. 나온 픽셀이 이웃 세포 위에 놓인 비율(픽셀 기준) 87.7%·43.6%. RxRx1은 밀도가 높을수록 overflow가 많다(23.5% → 42.6%)
- 2026-10-14 RxRx3-core 역할 확정과 원문 차이 분해(decisions.md, `results/E0/data_audit/rxrx3_core_roles.json`)
- 2026-10-14 JUMP 부분집합 규칙 재검토: 현행 907개는 사실상 "6개 이상 source에 나오는 화합물"(TARGET2 293개가 모두 여기에 포함). 대안 "source_4와 다른 source 1곳 이상" = 293개(= TARGET2 전부, 모두 9–10개 source에 있음), 처리군 11,716 wells. 현행과의 차이는 갤러리 방해 화합물 614개 유무
- 2026-10-14 16:23 JUMP-CP 원본 다운로드 완료(1,149 plates, 1,103,705 파일, 2.51 TB, 평균 120 MB/s) + RxRx3-core Stage A 완료(1,584 plates). `pvil_wait_jump`가 JUMP 분할을 자동 시작
- 2026-10-14 RxRx1 Stage A 완료: 44 plates, 27,094 FOV, 매칭 세포 431만. experiment별 FOV당 세포 중앙값 51–385(밀도 차이 최대 7.5배, 과분할 아님, 윤곽 확인). `scripts/preview_seg.py`, `pvil_data/seg/rxrx1/_preview/`. Stage B에서 experiment별 QC 통과율을 보고한다(핵/세포질 비율 QC가 밀집 experiment에 몰릴 수 있음)
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
| `pvil_e0_rxrx1_c` | 같은 형식, `dino_c_rxrx1.yaml` (GPU 1) | `.../pvil_e0_rxrx1_c_20261015_0037.log` | 2026-10-15 00:37 UTC |
| `pvil_e0_rxrx1_s` | 같은 형식, `dino_s_rxrx1.yaml` (GPU 2) | `.../pvil_e0_rxrx1_s_20261015_0037.log` | 2026-10-15 00:37 UTC |
| `pvil_e0_rxrx3_c` | `scripts/run_e0_tmux.sh pvil_e0_rxrx3_c 1 configs/E0/dino_c_rxrx3_core.yaml` (GPU 1, `pvil_wait_rxrx3`가 띄움) | `.../pvil_e0_rxrx3_c_20261015_0659.log` | 2026-10-15 06:59 UTC |
| `pvil_e0_rxrx3_s` | 같은 형식, `dino_s_rxrx3_core.yaml` (GPU 2) | `.../pvil_e0_rxrx3_s_20261015_0659.log` | 2026-10-15 06:59 UTC |

## 측정값

| 항목 | 값 | 측정일 | 비고 |
| --- | --- | --- | --- |
| T (B200, A1 1회 학습) | 측정 중 | | 각 run의 `timing.json`·`gpu_log.csv`. 공유 GPU라 상한으로 표기 |
| E0 학습 처리량(초기, 배치 512, checkpointing, GPU당 run 2개 + 분할 8개 + 타 사용자 학습 공존) | 110–224 samples/s | 2026-10-15 | RxRx3-core 1 epoch(2,350 step) ≈ 1.5–2.5 h, 최대 25 epoch |
| δ (비열등성 마진) | | | A1 seed 표준편차 × 2, E1 전에 고정 |
| E0 RxRx1 학습 처리량 (분할 거의 종료 후) | C 349–370, S 365–403 samples/s | 2026-10-15 06:30 | 분할 16개가 돌던 때(C 225, S 295)보다 1.4–1.6배. 1 epoch(약 1,680 step) ≈ 1.1 h였음 |
| 경합 상태 벤치마크 (GPU 1, batch 64, 합성 입력) | bf16 matmul 8192² 303 TFLOPS. 우리 모델 checkpointing 켬 212 / 끔 291 img/s(peak 1.8 / 6.1 GB). LoRA 없는 timm ViT-B 전체 파인튜닝 336 img/s | 2026-10-15 05:5x | 공칭(약 2,200 TFLOPS) 대비 약 14%만 받음 → 느린 주원인은 GPU 공유. 코드 측 손실은 checkpointing 약 27% + PEFT LoRA 약 13%. 경합 없는 처리량은 아직 미측정 |
| ViT-B/16 DINOv3 + LoRA(block 한정, 학습 파라미터 1,179,666), bf16 fwd+bwd | 0.105 s/step (batch 64, 약 609 img/s), peak 7.03 GB | 2026-10-14 | GPU 0, torch.compile 없음, 다른 작업과 GPU 공유 중에 잰 참고값 |
| bf16 matmul (4096²) | GPU0 16 / GPU1 180 / GPU2 79 / GPU3 371 TFLOPS | 2026-10-14 | 1차 측정(GPU0 7 / GPU1 52 / GPU2 470 / GPU3 238)과 크게 다르다. 사용량 채우기 작업이 도는 상태라 우리 작업이 실제로 받을 성능과 다를 수 있다 |
| Cellpose-SAM `cpsam` 추론 (256², 합성) | 0.55 s, 가중치 bf16, SHA-256 일치 | 2026-10-14 | 실데이터 분할 비용은 Phase 0 데이터 처리에서 따로 측정 |
| Cellpose-SAM 2회 + 매칭 (RxRx3-core 512²) | 0.62 s/FOV (plate 80 FOV ≈ 50 s) | 2026-10-14 | GPU 1장 1프로세스, 공유 상태 |
| Cellpose-SAM 2회 (JUMP 1080²) | 1.7 s/FOV (핵 0.74 + 세포 0.94) | 2026-10-14 | 매칭 제외, 공유 상태 |
| Stage A 실처리 (RxRx1 HepG2 512², 8 프로세스) | 2.2–2.3 s/FOV/프로세스, FOV당 세포 290–390 | 2026-10-14 | GPU 사용률 17–32%, CPU 후처리가 병목 → 16 프로세스로 재시작 |

## 이슈·블로커

1. **GPU 공유:** `nvidia-smi`에 다른 프로세스가 여럿 보인다. 그중 `/opt/conda/bin/python3`(multiprocessing spawn 4개, GPU당 40–143 GB, 2026-09 말부터 실행)은 다른 사람이 사용량을 채우려고 올린 작업으로, 우리가 GPU를 쓰면 자원이 유동적으로 우리에게 할당된다(사용자 확인, 2026-10-14). 따라서 점유 메모리·사용률만 보고 GPU가 부족하다고 판단하지 않는다. 그 밖의 학습 작업(다른 사용자의 train_val.py, run_brca.py 등)은 실제 경합 대상이다. T를 잴 때는 측정 시점의 GPU 점유 상태를 함께 기록한다
2. **CPU 스레드 경합(해결):** torch가 프로세스마다 CPU 스레드 144개를 잡아, 분할 프로세스 8–16개가 288코어에서 경합했다(같은 FOV가 2–18초). `OMP_NUM_THREADS` 등을 12로 제한해 재시작. 여러 프로세스를 띄울 때는 항상 스레드 수를 제한한다
3. **HF CDN 속도:** 단일 연결이 약 45 KB/s로 떨어지는 파일이 있다(cpsam_v2 1.23 GB). 64분할 병렬 range 다운로드 + SHA-256 확인으로 해결
4. CP-BG-Bench 아티팩트 미공개. 저자 문의는 하지 않기로 함 (2026-10-14)
5. JUMP-CP는 근사 부분집합(화합물 907 vs 원문 1,466). G0는 "근사" 표기
6. RxRx3-core well 수 불일치: 원문 104,166 wells / 735 perturbation, 메타데이터 CRISPR 126,900 / 736
7. RxRx1 HepG2 well 수 불일치: 원문 12,581 wells / 1,108 perturbation, 메타데이터 13,547 wells(처리군 12,179 + 양성대조군 1,318 + 음성대조군 50) / siRNA 라벨 1,139. 분할·QC 뒤 재비교
8. RxRx1 sphering: 대조군(EMPTY) well이 50개뿐인데 임베딩은 128차원이라, 대조군 공분산의 계수(rank)가 ≤ 49다. ZCA-cor(epsilon 1e-6)가 대조군에서 분산이 없는 방향을 크게 키울 수 있다. 원문 사양대로 진행하되, G0가 어긋나면 이것을 먼저 진단한다
9. **메모리 상한(cgroup) 641 GB:** 서버 RAM은 2.2 TB지만 컨테이너 상한은 641 GB다(`/sys/fs/cgroup/memory.max`). JUMP 분할 16개가 약 250 GB를 쓰는 동안에는 page cache로 약 333 GB만 남아, crop 캐시(RxRx3-core 227 + RxRx1 158 = 385 GB)가 다 올라가지 않는다. RxRx3-core run이 데이터 대기 70–80%(57 samples/s)로 떨어져 중단했고, 분할이 끝나면 자동으로 재시작한다. 앞으로 JUMP 캐시까지 더하면 동시 학습 데이터셋 수를 이 상한에 맞춰 정한다
