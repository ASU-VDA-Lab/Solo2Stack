#!/bin/bash
# Table VI transferability runs for the MB+HB dataset, three GPUs (GPU 3 is
# left to the other user). Eight runs: four settings x two arms. The "tech"
# setting (train hybrid -> test microbump) is NOT queued here: hybrid's gmap is
# spatially uniform, so restricting the gmap statistics to it alone gives
# std = 0 and the loader's (x-mean)/(std+1e-12) would blow the test-time
# microbump channel to ~1e12. That setting needs a normalisation decision first.
# repo bootstrap: configs/paths.yaml -> shell variables
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../solo2stack/env.sh"
PY=${ML_PYTHON}
D=${TRANSFER_DIR}
L=$D/logs
mkdir -p "$L" "$D/runs"

run_queue () {         # $1 = gpu index, rest = "setting:arm" pairs
  local gpu=$1; shift
  for job in "$@"; do
    local setting=${job%%:*}
    local arm=${job##*:}
    echo "[gpu$gpu] START $setting $arm  $(date -Is)" >> "$L/driver.log"
    CUDA_VISIBLE_DEVICES=$gpu $PY $D/train_transfer.py \
        --arm "$arm" --setting "$setting" --device cuda:0 \
        > "$L/${setting}_${arm}.log" 2>&1
    echo "[gpu$gpu] END   $setting $arm rc=$?  $(date -Is)" >> "$L/driver.log"
  done
}

echo "=== transfer_mbhb campaign started $(date -Is) ===" >> "$L/driver.log"
run_queue 0 h_interp:A1 depth:A3     tech_reverse:A1 &
run_queue 1 h_extrap:A1 depth:A1     tech_reverse:A3 &
run_queue 2 h_interp:A3 h_extrap:A3 &
wait
echo "=== transfer_mbhb campaign finished $(date -Is) ===" >> "$L/driver.log"
