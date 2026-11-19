#!/usr/bin/env bash
# Usage: bench.sh <rundir-with-gemm.loom> [repeats] [device-uri, e.g. xdna://0?columns=8]
# Runs loom-benchmark (dispatch_complete, device=xdna) N times and prints p50/min per run.
set -euo pipefail
d="$1"; n="${2:-3}"; dev="${3:-xdna}"
cd "$d"
for i in $(seq 1 "$n"); do
  loom-benchmark gemm.loom --benchmark=@gemm_bench --device="$dev" --measure=dispatch_complete \
    --output="bench_$i.json" >/dev/null 2>"bench_$i.stderr" || { echo "run $i failed"; tail -5 "bench_$i.stderr"; continue; }
  python3 - "bench_$i.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); w=d["work_items"][0]; m=w["measurement"]["operation_timing_ns"]
print(f'{sys.argv[1]}: state={w["state"]} correctness={w["correctness"]["state"]} n={m["count"]} min={m["min"]/1e6:.3f}ms p50={m["p50"]/1e6:.3f}ms max={m["max"]/1e6:.3f}ms')
PY
done
