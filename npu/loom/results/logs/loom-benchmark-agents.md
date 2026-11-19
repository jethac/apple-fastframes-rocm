## loom-benchmark

`loom-benchmark` executes `check.benchmark` records over authored
cases and scenarios. Use it after correctness is represented in the
module and the question requires physical timing.

### Ordinary loop

```shell
loom-benchmark kernel.loom --benchmark=@kernel_latency --dry-run
loom-benchmark kernel.loom --benchmark=@kernel_latency \
  --compile-only --target=amdgpu:gfx11-generic \
  --artifact-bundle-dir=compile-requests
loom-benchmark kernel.loom --benchmark=@kernel_latency \
  --device=amdgpu --measure=dispatch_complete --output=results.json
loom-benchmark kernel.loom --benchmark=@kernel_latency \
  --device=amdgpu --target=amdgpu:gfx11-generic \
  --measure=dispatch_complete --output=generic.json
loom-benchmark kernel.loom \
  --compare=@baseline_latency,@variant_latency \
  --device=amdgpu --measure=dispatch_complete \
  --interleave=ABABA --output-format=jsonl --output=comparison.jsonl
```

`--dry-run` resolves cases, samples, candidates, and deduplicated work
without compiling or executing them. A normal run writes a compact JSON
snapshot. `--output-format=jsonl` writes lifecycle rows and is useful for
long sweeps, live inspection, and interleaved comparisons.
`--compile-only` captures each unique selected concrete `check.case` sample as
canonical request bytecode under `compile_requests/`, compiles it without
a HAL device, and records roots, bindings, target, config, and source hashes
in `compile_requests/manifest.json`.
An explicit `--device=DRIVER` is validated before planning; unavailable
selections identify the missing driver.
`--target=family:selector` forces a compatible compiler target without changing the selected device. Empty target selection derives the best target from the device and authored kernel.

The default `--measure=auto` selects complete execution for a `check.case`
and prepared dispatch completion for a `check.scenario`. Scenario
comparison benchmarks execute only the target product after target-local
input staging; oracle execution and comparison remain correctness work.
Scenario timing defaults to unchecked: no oracle is prepared or run.
`--scenario-correctness=true` checks each selected trial once before
timing, including command programs and interleaved candidates. Failed
expectations or unhandled device errors then suppress its score.
Use this opt-in for occasional validation or release gates; realistic
model oracles can cost far more than the measured workload.
The flag does not change `check.case` preflight or the full
authored program timed by `case_end_to_end`.
JSONL sample rows carry scenario observations in `trial_result`; snapshots
retain failed trials in `failed_samples` and count all checks in `summary`.

`--measure=dispatch_complete` times device completion for the selected
launches. Keep it distinct from complete case timing, and compare candidates
with the same measure, inputs, launch shape, batching, and cache policy.
Profiling and compile diagnostics are collected outside measured windows.

### Read the result

```shell
jq '{summary, benchmarks, work_items}' results.json
jq '.work_items[] | {work_item_index,state,measure:.policy.measure,p50:(.measurement.operation_timing_ns.p50 // .measurement.timing_ns.p50)}' results.json
jq '(.work_items[] | select(.failure)), .failed_samples[]?, .failures[]?' results.json
jq 'select(.row=="benchmark") | .benchmark_result | {benchmark,state,p50:.measurement.operation_timing_ns.p50,stop_reason:.measurement.stop_reason}' comparison.jsonl
jq 'select(.row=="compile" and .state!="ok") | .diagnostics[]?' comparison.jsonl
```

Snapshot `work_items` own physical execution and timing; logical
`benchmarks` reference them by `work_item_index`. Missing failure arrays
mean there was nothing to report, not that correctness was checked.
`correctness.state` distinguishes `unchecked`, `passed`, and `failed`.
JSONL rows carry `row`, `run_id`, and
`candidate_id` fields for joining plan, compile, sample, benchmark,
comparison, profile, and summary evidence.

Use `--artifact-bundle-dir=DIR` when the result must travel with its
manifests, compile reports, target artifacts, and profile data. Use
`--compile-report=details` only after a summary identifies a compiler
question that needs row-level evidence.

Detailed measurement, comparison, output, and report workflows live in
`loom/docs/src/workflows/benchmark.md`,
`loom/docs/src/workflows/benchmark-results.md`, and
`loom/docs/src/workflows/compile-reports.md`.
