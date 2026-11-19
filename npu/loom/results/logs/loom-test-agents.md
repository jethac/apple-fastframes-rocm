## loom-test

`loom-test` executes `check.case` and `check.scenario` records from
ordinary Loom modules and writes a structured `loom.test.v0` JSON report.
Use it for correctness before promoting the same records to
`check.benchmark` rows.

### Common flows

```shell
loom-test module.loom
loom-test module.loom --case=@case_q8_block_unroll_wg64
loom-test module.loom --case=@sampled_choice --sample=1
loom-test test.loom --library=motifs.loombc
loom-test module.loom --file-output-root=artifacts
loom-test module.loom --max-samples-per-case=16
loom-test module.loom --pipeline=@hal_actual_pipeline
loom-test module.loom --device=amdgpu --target=amdgpu:gfx11-generic
loom-test module.loom --config=model.hidden_size=4096
loom-test module.loom --config-file=model_config.jsonc
loom-test module.loom --sanitizer=tsan
loom-test module.loom --sanitizer=asan --sanitizer-reporting=report-only
loom-test module.loom --xfail=@unsupported=TARGET/003
loom-test module.loom --allow-failure=@device_dependent=TARGET/003
```

`--case=@name` selects one checked case or scenario; empty selection runs
all records. `--sample=N` selects one planned sample for `check.case`
records. Scenario trial domains execute in bounded batches.
`--max-samples-per-case=N` bounds planning for generator-heavy cases.
`--xfail=@record=DOMAIN/NNN` requires that record to fail with the named
compile or expectation diagnostic. `--allow-failure` also accepts a pass.
Comma-separated diagnostics express target-dependent failure modes; any
other diagnostic fails the run.

### Fixture files

`check.file.read.npy` resolves relative paths beside the primary input
module, or in the working directory for stdin. Absolute read paths are
also accepted. `check.file.write.npy` requires an explicit
`--file-output-root` only when the write is performed; there is no
implicit output directory. Authored output paths are relative to that
directory, use '/' separators, and cannot contain '..' components.
Nested directories are created as needed. These lexical path rules do
not sandbox filesystem symlinks. '-' is not a file-output directory.

### Kernel launches

A case can mix reference/oracle checks with HAL kernel launches. Kernel
launches use the selected HAL artifact provider, target provider, and HAL device linked into this binary. An explicit `--device=DRIVER` is validated even when the selected cases contain no kernel launches; unavailable selections identify the missing driver. `--target=family:selector` forces the compiler target for every HAL kernel without changing the device selected by `--device`. `--pipeline=default|none|@symbol|pass,list` controls the HAL kernel compile pipeline. `none` disables all compiler transformations and requires emission-ready input. `--config=key=value` and `--config-file=path` materialize config declarations in the private compile copy before lowering runs. Case sample values remain runtime invocation inputs. `--sanitizer=...` and `--sanitizer-reporting=...` enable the same target-pipeline instrumentation accepted by `loom-benchmark`.

### Report shape

```shell
loom-test module.loom --case=@smoke | jq '.failed_sample_count'
loom-test module.loom | jq '.samples[] | {case, sample_ordinal, passed}'
loom-test module.loom | jq '.samples[] | select(.issues) | {case, issue: .issues[]}'
loom-test module.loom | jq '.trials[] | {scenario, configuration_ordinal, trial_index, trial_ordinal, passed}'
loom-test module.loom | jq '.skipped_cases[]? | {case, provider, op, code, provider_code}'
loom-test module.loom | jq '.planning_issues[]? | {case, kind, op, source_location, fix_hint}'
```

The report carries case/sample and scenario/trial counts, including
`failed_sample_count` and `failed_trial_count`, plus
`skipped_case_count`, `skipped_scenario_count`, `planning_issue_count`,
`samples`, `trials`, `skipped_cases`, `skipped_scenarios`,
and `planning_issues`. Skipped records use
stable `op` and `code` fields; `provider_code` is provider-defined and
`display_message` is human-facing only. Failed samples may include `issues`
when execution could not reach expectations, such as a compile rejection
with stable `category`, `provider`, `stage`, and `kind` fields. Planning
issues carry stable
`kind`, `case`, `op`, `source_location`, and optional `fix_hint` fields.
A nonzero failed sample, failed trial, or planning issue count makes the
process fail after the JSON report is written.
