#!/usr/bin/env bash
# Stage A 분할을 tmux 세션 하나에 샤드별 창으로 띄운다.
# 사용: scripts/run_segment_tmux.sh SESSION NSHARD THREADS CONFIG [CONFIG ...]
#   예: scripts/run_segment_tmux.sh pvil_seg 16 12 configs/preprocess/rxrx1.yaml configs/preprocess/rxrx3_core.yaml
# 샤드 i는 GPU 목록(환경 변수 GPUS, 기본 "1 2")의 i % 개수 번째를 쓰고, CONFIG들을 순서대로 처리한다. 프로세스당 CPU 스레드를 THREADS로 제한한다
# (제한하지 않으면 torch가 프로세스마다 144 스레드를 잡아 서로 경합한다).
set -eu
SESSION=$1; N=$2; THREADS=$3; shift 3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR=/home/genetics/yiombi/pvil_data/logs
TS=$(date +%Y%m%d_%H%M)
read -r -a GPU_LIST <<< "${GPUS:-1 2}"   # 공유 서버: 기본은 GPU 1, 2만 (사용자 지시 2026-10-14)
NGPU=${#GPU_LIST[@]}
chain=""
for c in "$@"; do chain="${chain:+$chain && }python scripts/segment.py $c --shard \$I/$N"; done
tmux new-session -d -s "$SESSION" -n s0
for i in $(seq 0 $((N - 1))); do
  [ "$i" -gt 0 ] && tmux new-window -t "$SESSION" -n "s$i"
  tmux send-keys -t "$SESSION:s$i" "cd $REPO && source scripts/env.sh && export I=$i CUDA_VISIBLE_DEVICES=${GPU_LIST[$((i % NGPU))]} OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS && ($chain) 2>&1 | grep --line-buffered -v -i warn | tee $LOGDIR/segment_${TS}_shard$i.log" Enter
done
echo "session=$SESSION logs=$LOGDIR/segment_${TS}_shard*.log"
