#!/usr/bin/env bash
# JUMP 분할(1,149 plate)이 끝나 메모리(cgroup 641 GB)가 풀리면 RxRx3-core E0 C/S를 GPU 1·2에 띄운다.
# 이유: 분할 16개 프로세스가 약 250 GB를 써서 crop 캐시가 page cache에 다 올라가지 않는다(progress.md 이슈 9)
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEG=/home/genetics/yiombi/pvil_data/seg/jump_cp
until [ "$(ls $SEG | grep -c 'cells.parquet$')" -ge 1149 ] && ! pgrep -f 'scripts/segment.py' >/dev/null; do sleep 60; done
echo "$(date -u) JUMP 분할 완료 확인, RxRx3-core E0 시작"
cd "$REPO"
scripts/run_e0_tmux.sh pvil_e0_rxrx3_c 1 configs/E0/dino_c_rxrx3_core.yaml
scripts/run_e0_tmux.sh pvil_e0_rxrx3_s 2 configs/E0/dino_s_rxrx3_core.yaml
