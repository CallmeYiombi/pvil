#!/usr/bin/env bash
# E0 run 하나를 tmux 세션으로 띄운다: (GPU 여유 확인 → train.py, OOM이면 재시도) → evaluate.py.
# 사용: scripts/run_e0_tmux.sh SESSION GPU CONFIG [WORKERS]
#   예: scripts/run_e0_tmux.sh pvil_e0_rxrx3_c 1 configs/E0/dino_c_rxrx3_core.yaml
# 공유 서버: GPU는 1, 2만 쓴다(decisions.md 2026-10-14). CPU 스레드는 프로세스당 8로 제한한다.
# 환경 변수: NEED_GB(시작 전 필요한 여유, 기본 25), MAX_TRY(OOM 재시도 횟수, 기본 3), RETRY_SEC(기본 600)
set -eu
SESSION=$1; GPU=$2; CONFIG=$3; WORKERS=${4:-10}
case "$GPU" in 1|2) ;; *) echo "GPU는 1 또는 2만 허용" >&2; exit 1 ;; esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=/home/genetics/yiombi/pvil_data/logs/${SESSION}_$(date -u +%Y%m%d_%H%M).log
tmux new-session -d -s "$SESSION" "cd $REPO && scripts/run_e0_job.sh $GPU $CONFIG $WORKERS 2>&1 | grep --line-buffered -v -i warn | tee $LOG"
echo "session=$SESSION gpu=$GPU log=$LOG"
