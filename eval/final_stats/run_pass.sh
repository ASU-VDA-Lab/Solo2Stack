#!/bin/bash
# $1 = cuda index, rest = folds. All four arms are run per fold so the three
# later arms hit the page cache the first arm warmed.
# repo bootstrap: configs/paths.yaml -> shell variables
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../solo2stack/env.sh"
G=$1; shift
PY=${ML_PYTHON}
cd ${DATA_ROOT}
for k in "$@"; do
  $PY -I -u task45_model_pass.py --arms A0 A1 A2 A3 --folds $k \
      --device cuda:$G --workers 16 \
      --out-dir ${RESULTS}/pass \
      >> ${RESULTS}/logs/pass_gpu$G.log 2>&1
done
echo "GPU $G done: folds $*" >> ${RESULTS}/logs/pass_gpu$G.log
