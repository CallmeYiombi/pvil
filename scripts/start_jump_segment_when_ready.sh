#!/usr/bin/env bash
# JUMP-CP 다운로드(로그의 DONE)와 RxRx3-core 분할(1,584 plates)이 모두 끝나면 JUMP 분할을 16 샤드로 시작한다.
# 사용: tmux new-session -d -s pvil_wait_jump scripts/start_jump_segment_when_ready.sh DOWNLOAD_LOG
set -u
DL_LOG=$1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEG=/home/genetics/yiombi/pvil_data/seg/rxrx3_core
until grep -q "^DONE" "$DL_LOG" && [ "$(ls $SEG/*.cells.parquet 2>/dev/null | wc -l)" -ge 1584 ]; do sleep 60; done
echo "$(date -u) 조건 충족: JUMP 다운로드 완료, RxRx3-core 분할 완료"
tmux kill-session -t pvil_seg 2>/dev/null || true
cd "$REPO" && scripts/run_segment_tmux.sh pvil_seg_jump 16 12 configs/preprocess/jump_cp.yaml
echo "$(date -u) JUMP 분할 시작"
