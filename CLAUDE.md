# PVIL — Paired-View Interventional Learning for Cell Painting

<!-- Claude Code가 매 세션 시작 시 자동으로 읽는 파일이다. 200줄 미만으로 유지하고, 세부 사양은 docs/에 둔다. -->
<!-- 전체 계획을 @import하지 않는다. import한 파일은 매 세션 통째로 로드되어 컨텍스트를 낭비한다. 필요할 때 읽게 한다. -->

## 프로젝트 요약

- 목표: CP-BG-Bench(arXiv 2609.12761)의 paired view(C/S/CD/SD)를 **학습 신호**로 써서, 배치 특이 shortcut은 억제하고 배치를 넘어 전이되는 맥락은 보존하는 Cell Painting 인코더 학습법 PVIL을 개발한다.
- 주 지표(사전 등록): held-out 배치의 cross-batch perturbation R@10 (post-Harmony).
- 비열등성 지표: replicate mAP, CellProfiler feature prediction(median Pearson r), scIB Bio.
- 데이터셋: JUMP-CP(cpg0016 부분집합, 화합물, U2OS), RxRx1(HepG2, siRNA), RxRx3-core(HUVEC, CRISPR-KO).
- 목표 학회: NeurIPS 2027 (백업 ICLR 2028).

## 현재 단계

- **Phase 0: CP-BG-Bench 재현** → 작업 지시서 `docs/tasks/P0_reproduction.md`
- 세션을 시작하면 먼저 `docs/progress.md`를 읽는다. 세션을 끝내기 전에 갱신한다.

## 문서 지도 (필요할 때 읽는다)

| 파일 | 내용 |
| --- | --- |
| `docs/research_plan.md` | 전체 계획: 가설, 손실 함수, 데이터, 베이스라인, 실험 매트릭스, 관문, 예산 |
| `docs/tasks/P0_reproduction.md` | 현재 단계 작업 지시서와 CP-BG-Bench 재현 사양·참조값 |
| `docs/progress.md` | 세션 인수인계: 다음 할 일, 진행 중, 실행 중인 작업, 측정값 |
| `docs/decisions.md` | 설계·프로토콜 결정 로그 (날짜, 결정, 근거) |
| `.claude/rules/evaluation.md` | 평가 코드 작업 시 자동 로드되는 평가 규칙 |

## 절대 규칙

1. **테스트 배치 봉인**: JUMP-CP `source_4`, RxRx1 `HEPG2-07`, RxRx3-core `plate 4`는 최종 평가 외에 학습, early stopping, 하이퍼파라미터 선택, 디버깅에 쓰지 않는다. 주 프로토콜은 P-strict(별도 validation 배치)다. 예외는 E0 재현의 P-orig뿐이며, 결과에 `protocol: P-orig`를 기록한다.
2. **후처리 고정**: 모든 방법에 L2 정규화 → well 평균 → 대조군 기준 ZCA-cor sphering → PCA 50 → Harmony를 적용하고, Harmony 전후 결과를 모두 저장한다.
3. **평가 누수 금지**: cp_measure 특징은 평가 타깃이다. 학습 목표, 보조 손실, 하이퍼파라미터 선택 기준으로 쓰지 않는다.
4. **반복과 기록**: 보고용 결과는 3 seeds 평균 ± 표준편차다. 최고 seed만 고르지 않는다. 모든 run은 config, seed, git commit hash, 데이터 버전을 결과와 함께 저장한다.
5. **공정성**: 조정 항이 있는 모든 방법에 같은 탐색 예산(데이터셋당 8 trials)을 준다.
6. **사전 등록 유지**: 판정 규칙과 비열등성 마진 δ는 E1 실행 전에 고정한다. 바꾸려면 `docs/decisions.md`에 날짜와 사유를 남기고 사용자 승인을 받는다.
7. **데이터 라이선스**: RxRx1은 CC BY-NC-SA 4.0이다. 원본·파생 이미지를 외부에 재배포하지 않는다. 공개 대상은 코드와 재구성 파이프라인뿐이다.
8. **사실 날조 금지**: 논문 수치나 설정을 쓸 때는 `docs/research_plan.md`와 `docs/tasks/P0_reproduction.md`의 출처 표기를 확인한다. 모르는 값은 추측하지 말고 TODO로 남긴 뒤 사용자에게 묻는다.

## 저장소 구조 (목표)

```
CLAUDE.md
.claude/rules/evaluation.md
docs/                 research_plan.md, progress.md, decisions.md, tasks/
configs/              실험 yaml (base_cpbg.yaml을 상속)
src/pvil/
  data/               다운로드, Cellpose-SAM 분할, crop·mask 저장, view 생성기
  models/             백본 + 채널 어댑터 + LoRA, projection head, PVIL 성분 head
  losses/             multi-positive contrastive, L_view, L_null, L_inv, L_dres
  eval/               집계·후처리, recall, copairs mAP, scIB, CP prediction, 진단 지표
  baselines/          B·C·D 그룹 구현
scripts/              make_views.py, train.py, embed.py, evaluate.py
tests/                단위 테스트
results/<exp>/<method>/<dataset>/seed<k>/   metrics.json, config.yaml, git_hash.txt
data/                 원본·중간 산출물 (git 제외)
```

## 작업 방식

- 사용자와의 대화, 진행 상황 설명, 문서는 한국어로 쓴다.
- 여러 파일에 걸친 변경은 plan mode로 계획을 먼저 보여주고, 승인 후 구현한다.
- 새 실험은 `configs/`에 yaml을 추가하고 스크립트 인자로 실행한다. 경로와 하이퍼파라미터를 코드에 하드코딩하지 않는다.
- view 생성, 손실, 평가 지표 코드를 바꾸면 `pytest -q tests/`를 실행한다.
- 장시간 학습은 `tmux`로 띄우고, 세션 이름과 로그 경로를 `docs/progress.md`의 "실행 중인 작업" 표에 적는다.
- 의미 있는 단위마다 git commit하고 `origin`(https://github.com/CallmeYiombi/pvil.git) `main`에 push한다. 결과 원본은 커밋하지 않고 `results/**/summary.md`만 커밋한다.
- 커밋 메시지와 PR에 Claude 관련 문구, Co-Authored-By, 워터마크를 넣지 않는다.
- `~/`(/home/jovyan)에 아무것도 저장하지 않는다. venv, 캐시, 모델 가중치, 데이터는 모두 `/home/genetics/yiombi/` 아래에 둔다. 작업 전에 `source scripts/env.sh`로 경로를 설정한다.
- 데이터는 150×150 crop과 instance mask만 저장하고, view는 학습 중에 생성한다. 모든 view를 저장하면 데이터셋당 약 1.8 TB가 된다.

## 환경 (B200 클라우드 서버)

- 세션 시작 시 GPU 확인:
  `nvidia-smi`
  `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"`
- 학습은 bf16 mixed precision을 쓴다(CP-BG-Bench와 동일).
- 1회 학습 비용 T(GPU-시간)를 E0에서 실측해 `docs/progress.md`에 기록한다. 계획서의 예산은 A100 기준 추정(T ≈ 2–5 GPU-시간)이므로 실측값으로 대체한다.
- 서버는 재생성될 수 있다. 작업 상태는 auto memory가 아니라 이 저장소의 `docs/progress.md`와 git에 남긴다(auto memory는 머신 로컬이다).
