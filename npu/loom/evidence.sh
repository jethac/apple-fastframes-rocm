#!/usr/bin/env bash
# Collect evidence that a Loom XDNA run actually executes on the NPU.
# Usage: evidence.sh <rundir-with-gemm.loom> <outdir>
# Runs loom-benchmark (device xdna://0?columns=8) repeatedly in the background
# and samples, while it runs: the process's open fds (/dev/accel/accel0) and
# xrt-smi's hardware-context table (PID, columns, submissions/completions).
set -u
d="$1"; out="$2"; mkdir -p "$out"
absd=$(cd "$d" && pwd)
# xrt-smi as the normal user fails here with an mmap EAGAIN (memlock limit 8 MiB
# < its 64 MiB mapping), so it is run read-only via sudo.
X="sudo -n /opt/xilinx/xrt/bin/xrt-smi"
amdf-info --device=xdna --details > "$out/amdf-info-details.txt" 2>&1
amdf-info --device=xdna --activate > "$out/amdf-info-activate.txt" 2>&1
loom-run --list_devices=xdna > "$out/loom-run-list_devices.txt" 2>&1
$X examine > "$out/xrt-smi_examine_idle.txt" 2>&1
$X examine -r aie-partitions > "$out/xrt-smi_aie-partitions_idle.txt" 2>&1

bench_loop() {
  cd "$absd" || exit 1
  for r in 1 2 3 4 5 6; do
    loom-benchmark gemm.loom --benchmark=@gemm_bench "--device=xdna://0?columns=8" \
      --measure=dispatch_complete "--output=evidence_bench_$r.json" > "evidence_bench_$r.log" 2>&1
    echo "run $r rc=$?" >> evidence_bench_rc.txt
  done
}
rm -f "$absd"/evidence_bench_*
bench_loop &
bp=$!
sleep 1
for i in 1 2 3 4 5; do
  pid=$(pgrep -n loom-benchmark || true)
  {
    echo "### sample $i $(date +%T.%N) loom-benchmark pid=$pid"
    if [ -n "$pid" ]; then
      tr '\0' ' ' < /proc/$pid/cmdline; echo
      ls -l /proc/$pid/fd 2>/dev/null | grep -E "accel|dri|kfd"
    fi
  } >> "$out/proc_during_run.txt"
  $X examine -r aie-partitions > "$out/xrt-smi_aie-partitions_during_$i.txt" 2>&1
  sleep 0.5
done
$X examine -r all > "$out/xrt-smi_all_during.txt" 2>&1
wait $bp
cp "$absd"/evidence_bench_* "$out/" 2>/dev/null
echo done
