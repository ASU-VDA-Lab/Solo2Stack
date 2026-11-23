#!/usr/bin/env bash
# Solve every chiplet in the library on its own, at all five cooling conditions.
#
# These single-chiplet ("solo") fields are the extra input that distinguishes
# the A2 and A3 arms: one solo solve per chiplet is reused by every stack that
# chiplet appears in.  Output lands in $SOLO_RUNS as
#     <chiplet>_<bc>.grid.steady.layer{0,1}
#
# The solver is SciPy's sparse LU, the only backend this repository ships.  The
# published solo labels were originally produced with an external SuperLU_MT
# driver, which has been removed; see pact/README.md for the comparison.
#
# Everything outside the repository comes from configs/paths.yaml.
# repo bootstrap: configs/paths.yaml -> shell variables
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../solo2stack/env.sh"
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

LCF_DIR="${CHIPLET_LIBRARY}/raw_lcf"
CONFIG_DIR="${REPO}/configs/pact"
MODEL_PARAMS="${CONFIG_DIR}/modelParam.config"

OUTDIR="${SOLO_RUNS}"
LOGDIR="${PACT_RUNS}/logs_solo"
# PACT resolves relative layer paths in an LCF against its cwd, which must
# therefore be the chiplet library's parent directory
RUN_CWD="$(dirname "${CHIPLET_LIBRARY%/}")"

MAX_JOBS="${MAX_JOBS:-5}"

mkdir -p "$OUTDIR" "$LOGDIR"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [ ! -f "${PACT_SRC}/PACT.py" ]; then
  echo "ERROR: no PACT at ${PACT_SRC}/PACT.py" >&2
  echo "run ./pact/fetch_pact.sh and set pact_src in configs/paths.yaml" >&2
  exit 1
fi

BC_FILES=()
for i in 0 1 2 3 4; do
  bc="${CONFIG_DIR}/bc_${i}.config"
  if [ ! -f "$bc" ]; then
    echo "ERROR: missing BC config: $bc" >&2
    exit 1
  fi
  BC_FILES+=("$bc")
done

run_one_case() {
  local lcf_file="$1" bc_file="$2" chiplet_id="$3" bc_name="$4"
  local grid_base="${OUTDIR}/${chiplet_id}_${bc_name}.grid.steady"
  local log_file="${LOGDIR}/${chiplet_id}_${bc_name}.log"

  if [ -f "${grid_base}.layer0" ]; then
    echo "=== ${chiplet_id} ${bc_name} already completed, skipping ==="
    return 0
  fi

  set +e
  ( cd "$RUN_CWD" && "$PACT_PYTHON" "${PACT_SRC}/PACT.py" \
      "$lcf_file" "$bc_file" "$MODEL_PARAMS" \
      --gridSteadyFile "$grid_base" ) > "$log_file" 2>&1
  local rc=$?
  set -e

  if [ $rc -ne 0 ]; then
    echo "!!! ${chiplet_id} ${bc_name} FAILED (rc=$rc) @ $(date)" \
      | tee -a "${LOGDIR}/failed.list"
  else
    echo "=== ${chiplet_id} ${bc_name} OK @ $(date) ===" >> "$log_file"
  fi
  return $rc
}

wait_for_slot() {
  while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "$1" ]; do
    set +e
    wait -n
    set -e
  done
}

shopt -s nullglob
lcf_files=("${LCF_DIR}"/chiplet*_lcf.csv)

if [ ${#lcf_files[@]} -eq 0 ]; then
  echo "ERROR: no LCF files found in ${LCF_DIR}" >&2
  echo "run data_gen/floorset/build_chiplet_library.py first" >&2
  exit 1
fi

echo "chiplets : ${#lcf_files[@]}"
echo "PACT     : ${PACT_SRC}/PACT.py"
echo "output   : ${OUTDIR}"
echo "cwd      : ${RUN_CWD}"

for lcf_file in "${lcf_files[@]}"; do
  base="$(basename "$lcf_file")"
  chiplet_id="${base%_lcf.csv}"
  echo "=== ${chiplet_id} @ $(date) ==="
  for bc_file in "${BC_FILES[@]}"; do
    bc_name="$(basename "$bc_file" .config)"
    wait_for_slot "$MAX_JOBS"
    run_one_case "$lcf_file" "$bc_file" "$chiplet_id" "$bc_name" &
  done
done

wait
echo "=== All jobs finished @ $(date) ==="
