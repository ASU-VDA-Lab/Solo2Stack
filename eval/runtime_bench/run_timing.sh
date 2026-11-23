#!/usr/bin/env bash
# Full-stack and solo PACT wall time at 2, 3 and 4 dies.
# One job at a time, single-threaded, identical solver (SciPy SuperLU) at every
# depth, so the depth comparison is not confounded by contention or by the
# SuperLU_MT/SciPy difference that separated the old 2-die data from the newer
# 3-/4-die data.
# repo bootstrap: configs/paths.yaml -> shell variables
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../solo2stack/env.sh"
set -u
B=${PACT_RUNS}/runtime_bench
PY=${PACT_PYTHON}
CFG="$B/Config_files/bc_4.config $B/modelParams_files/modelParam.config"
REPS=2
cd "$B"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "config,rep,wall_s,peak_rss_gb" > "$B/timing.csv"

run_one () {          # $1 = lcf basename
  local name=$1
  for r in $(seq 1 $REPS); do
    /usr/bin/time -f "%e %M" -o "$B/logs/${name}_$r.time" \
      $PY src_bench/PACT.py "lcf/${name}.lcf.csv" $CFG \
      --gridSteadyFile "out/${name}" > "$B/logs/${name}_$r.log" 2>&1
    read w m < "$B/logs/${name}_$r.time"
    printf "%s,%d,%.2f,%.3f\n" "$name" "$r" "$w" \
      "$(echo "scale=4; $m/1048576" | bc)" >> "$B/timing.csv"
    echo "  $name rep$r  ${w}s" >&2
  done
}

echo "warm-up..." >&2
$PY src_bench/PACT.py lcf/d2_microbump.lcf.csv $CFG --gridSteadyFile out/warm \
    > /dev/null 2>&1

for n in solo d2_microbump d2_hybrid \
         d3_microbump d3_hybrid \
         d4_microbump d4_hybrid; do
  run_one "$n"
done
echo "TIMING DONE" >&2
cat "$B/timing.csv"
