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

1. 사용자 확인 필요 항목(아래 이슈 1–3) 결정
2. 블로커 확인: CP-BG-Bench 아티팩트 공개 여부, JUMP-CP 부분집합 정의 (저자 문의, 사용자 진행)
3. RxRx3-core 다운로드와 메타데이터 확인 (배치 키 "plate"). HF CDN 속도 문제가 있으므로 큰 파일은 병렬 range 다운로드 사용

## 진행 중

- (없음)

## 완료

- 2026-10-14 환경 구축 (`docs/tasks/P0_env_setup.md`): 독립 repo + GitHub push, uv 환경, 패키지 골격, `configs/base_cpbg.yaml`, `scripts/check_env.py`, `tests/test_env.py`(15 passed). 점검 결과는 아래 측정값
- 2026-10-14 연구 계획 수립 (`docs/research_plan.md`)

## 실행 중인 작업

| tmux 세션 | 명령 | 로그 경로 | 시작 시각 |
| --- | --- | --- | --- |

## 측정값

| 항목 | 값 | 측정일 | 비고 |
| --- | --- | --- | --- |
| T (B200, A1 1회 학습) | | | 계획서 추정은 A100 기준 2–5 GPU-시간. E0에서 측정 |
| δ (비열등성 마진) | | | A1 seed 표준편차 × 2, E1 전에 고정 |
| ViT-B/16 DINOv3 + LoRA, bf16 fwd+bwd | 0.128 s/step (batch 64, 약 500 img/s), peak 7.1 GB | 2026-10-14 | GPU 0, torch.compile 없음, **다른 작업과 GPU 공유 중**이라 하한 참고값일 뿐 |
| bf16 matmul (4096²) | GPU0 7 / GPU1 52 / GPU2 470 / GPU3 238 TFLOPS | 2026-10-14 | GPU 점유율에 따라 66배 차이. 공유 상태의 지표 |

## 이슈·블로커

1. **[사용자 확인] Harmony max_iter:** 지시서는 "harmonypy 기본값 (θ=2, λ=1, σ=0.1, max_iter 100)"인데 harmonypy 2.1.0의 `max_iter_harmony` 기본값은 10이다. 원문 값을 확인할 때까지 config는 TODO
2. **[사용자 확인] Cellpose-SAM 가중치:** cellpose 4.2.1.1의 기본 모델은 `cpsam_v2`다. 원문의 "Cellpose-SAM 기본 파라미터"가 `cpsam`(v1)인지 `cpsam_v2`인지 미확인. config는 TODO
3. **[사용자 확인] scIB Batch 집계:** PCR 포함 5개 vs 원문 A.6 표의 4개
4. **GPU 공유:** B200 4장 모두 다른 사용자 작업이 GPU당 약 160 GB를 점유하고 있다. 학습 시간 T 측정과 장시간 학습 배치에 영향. 측정할 때 GPU 점유율을 같이 기록한다
5. **HF CDN 속도:** 단일 연결이 약 45 KB/s로 떨어지는 파일이 있다(cpsam_v2 1.23 GB). 64분할 병렬 range 다운로드 + SHA-256 확인으로 해결
6. CP-BG-Bench 아티팩트 미공개 (2026-10-14 기준)
7. JUMP-CP 부분집합의 정확한 plate·화합물 목록이 원문에 없음
