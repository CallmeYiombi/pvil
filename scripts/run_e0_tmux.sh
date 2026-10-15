#!/usr/bin/env bash
# E0 run 하나를 tmux 세션으로 띄운다: train.py → evaluate.py.
# 사용: scripts/run_e0_tmux.sh SESSION GPU CONFIG [WORKERS]
#   예: scripts/run_e0_tmux.sh pvil_e0_rxrx3_c 1 configs/E0/dino_c_rxrx3_core.yaml
# 공유 서버: GPU는 1, 2만 쓴다(decisions.md 2026-10-14). CPU 스레드는 프로세스당 8로 제한한다.
set -eu
SESSION=$1; GPU=$2; CONFIG=$3; WORKERS=${4:-10}
case "$GPU" in 1|2) ;; *) echo "GPU는 1 또는 2만 허용" >&2; exit 1 ;; esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=/home/genetics/yiombi/pvil_data/logs/${SESSION}_$(date -u +%Y%m%d_%H%M).log
tmux new-session -d -s "$SESSION" \
  "cd $REPO && source scripts/env.sh && export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
   (python scripts/train.py $CONFIG --workers $WORKERS && python scripts/evaluate.py $CONFIG) 2>&1 | grep --line-buffered -v -i warn | tee $LOG"
echo "session=$SESSION gpu=$GPU log=$LOG"
