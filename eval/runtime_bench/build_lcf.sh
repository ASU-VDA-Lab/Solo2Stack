#!/usr/bin/env bash
# Build runtime-benchmark LCFs: the same chiplets composed into 2-, 3- and
# 4-die stacks, so the depth comparison is not confounded by chiplet choice.
# repo bootstrap: configs/paths.yaml -> shell variables
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../solo2stack/env.sh"
set -e
B=${PACT_RUNS}/runtime_bench
rm -f "$B"/lcf/*.lcf.csv
C=(chiplet041 chiplet478 chiplet227 chiplet352)
H='Layer,FloorplanFile,Thickness (m),PtraceFile,LateralHeatFlow'

die () { echo "$1,floorset_pipeline_out/raw_floorplans/$2_flp.csv,0.0005,floorset_pipeline_out/raw_ptraces/$2_ptrace.csv,True"; }
bond () { echo "$1,floorset_pipeline_out/interconnect_flp_grid/$2,$3,,True"; }

declare -A FLP TH
FLP[microbump]=microbump_40x40_30um_flp.csv;  TH[microbump]=2.5e-05
FLP[hybrid]=hybrid_100x100_5um_flp.csv;       TH[hybrid]=1.5e-06

for t in microbump hybrid; do
  f=${FLP[$t]}; th=${TH[$t]}
  { echo "$H"; die 0 "${C[0]}"; bond 1 "$f" "$th"; die 2 "${C[1]}"; } \
      > "$B/lcf/d2_$t.lcf.csv"
  { echo "$H"; die 0 "${C[0]}"; bond 1 "$f" "$th"; die 2 "${C[1]}";
    bond 3 "$f" "$th"; die 4 "${C[2]}"; } > "$B/lcf/d3_$t.lcf.csv"
  { echo "$H"; die 0 "${C[0]}"; bond 1 "$f" "$th"; die 2 "${C[1]}";
    bond 3 "$f" "$th"; die 4 "${C[2]}"; bond 5 "$f" "$th"; die 6 "${C[3]}"; } \
      > "$B/lcf/d4_$t.lcf.csv"
done
{ echo "$H"; die 0 "${C[0]}"; } > "$B/lcf/solo.lcf.csv"
ls "$B/lcf"
