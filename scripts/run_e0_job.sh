#!/usr/bin/env bash
# run_e0_tmux.sh가 tmux 안에서 실행한다. 사용: scripts/run_e0_job.sh GPU CONFIG WORKERS
set -u
GPU=$1; CONFIG=$2; WORKERS=$3
NEED_GB=${NEED_GB:-25}; MAX_TRY=${MAX_TRY:-3}; RETRY_SEC=${RETRY_SEC:-600}
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TMP=$(mktemp)
for try in $(seq 1 "$MAX_TRY"); do
  until [ "$(python3 scripts/gpu_free_gb.py "$GPU" | cut -d. -f1)" -ge "$NEED_GB" ]; do
    echo "$(date -u +%T) GPU $GPU 여유 $(python3 scripts/gpu_free_gb.py "$GPU") GB < ${NEED_GB} GB, 대기"; sleep 300
  done
  echo "$(date -u +%T) 시도 $try/$MAX_TRY: GPU $GPU 여유 $(python3 scripts/gpu_free_gb.py "$GPU") GB, train.py 시작"
  python scripts/train.py "$CONFIG" --workers "$WORKERS" 2>&1 | tee "$TMP"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    python scripts/evaluate.py "$CONFIG"; rm -f "$TMP"; exit $?
  fi
  if grep -q -E "OutOfMemoryError|CUDA error: out of memory" "$TMP"; then
    echo "$(date -u +%T) OOM. ${RETRY_SEC}s 뒤 처음부터 재시도"; sleep "$RETRY_SEC"
  else
    echo "$(date -u +%T) OOM이 아닌 오류로 종료. 재시도하지 않는다"; rm -f "$TMP"; exit 1
  fi
done
echo "$(date -u +%T) OOM 재시도 $MAX_TRY회 모두 실패"; rm -f "$TMP"; exit 1
