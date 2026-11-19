"""Shared harness: run an ONNX model through the Ryzen AI 1.9 VitisAI EP (VAIML, BF16) and
prove where it ran.

Evidence collected for every run:
  * VAIML partitions from the compiler cache context.json (device == "VAIML"), with the
    number of ONNX nodes each partition covers vs the total graph node count;
  * ORT profiling events grouped by execution provider (VitisAIExecutionProvider vs CPU),
    with per-provider summed kernel time;
  * the loaded XRT / VAIML shared libraries (from /proc/self/maps).

A run with no executed VAIML partition is reported as CPU fallback, never as NPU success.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def compare(actual: np.ndarray, ref: np.ndarray, col_block: int = 64) -> dict:
    """Numerics vs a reference, with a specific check for silently zeroed column blocks.

    The last axis is treated as "columns" (the Gemm N axis of the stamp bug).
    """
    a = np.asarray(actual, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    assert a.shape == r.shape, (a.shape, r.shape)
    diff = np.abs(a - r)
    af, rf = a.ravel(), r.ravel()
    cos = float(af @ rf / (np.linalg.norm(af) * np.linalg.norm(rf) + 1e-30))
    a2 = a.reshape(-1, a.shape[-1])
    r2 = r.reshape(-1, r.shape[-1])
    zero_cols = np.all(a2 == 0, axis=0) & ~np.all(r2 == 0, axis=0)
    n = a2.shape[1]
    blocks = []
    for s in range(0, n, col_block):
        e = min(n, s + col_block)
        blk_err = float(np.linalg.norm(a2[:, s:e] - r2[:, s:e]) / (np.linalg.norm(r2[:, s:e]) + 1e-30))
        blocks.append(blk_err)
    return dict(
        shape=list(a.shape),
        max_abs_err=float(diff.max()),
        mean_abs_err=float(diff.mean()),
        ref_abs_max=float(np.abs(r).max()),
        rel_l2=float(np.linalg.norm(a - r) / (np.linalg.norm(r) + 1e-30)),
        cosine=cos,
        finite=bool(np.isfinite(a).all()),
        zero_columns_unexpected=int(zero_cols.sum()),
        zero_column_ranges=_ranges(np.flatnonzero(zero_cols)),
        worst_col_block_rel_l2=float(max(blocks)),
        col_block=col_block,
    )


def _ranges(idx):
    out = []
    for i in idx.tolist():
        if out and out[-1][1] == i - 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def vaiml_config(root: Path, tp_size: int | None, optimize_level: int = 1, extra: dict | None = None) -> Path:
    settings = dict(optimize_level=optimize_level, preferred_data_storage="auto",
                    keep_outputs=True, enable_temp_file_io_in_mem=False)
    if tp_size is not None:
        settings["tp_size"] = tp_size
    if extra:
        settings.update(extra)
    config = {"passes": [{"name": "init", "plugin": "vaip-pass_init"},
                         {"name": "vaiml_partition", "plugin": "vaip-pass_vaiml_partition",
                          "vaiml_config": settings}],
              "target": "VAIML",
              "targets": [{"name": "VAIML", "pass": ["init", "vaiml_partition"]}]}
    path = root / "config.json"
    path.write_text(json.dumps(config, indent=2))
    return path


def loaded_libs() -> list[str]:
    keys = ("libxrt", "libvaiml2", "libflexmlrt", "libonnxruntime_vitisai_ep", "libxrt_driver_xdna")
    return sorted({l.split()[-1] for l in Path("/proc/self/maps").read_text().splitlines()
                   if any(k in l for k in keys)})


def placement(root: Path, model_path: Path) -> dict:
    total = len(onnx.load(str(model_path), load_external_data=False).graph.node)
    parts = []
    for ctx in (root / "cache").glob("*/context.json"):
        for meta in json.loads(ctx.read_text()).get("metaDef", []):
            parts.append(dict(id=meta.get("id"), device=meta.get("device"), nodes=len(meta.get("nodes", []))))
    on_npu = sum(p["nodes"] for p in parts if p["device"] == "VAIML")
    return dict(total_onnx_nodes=total, vaiml_partitions=parts,
                nodes_in_vaiml_partitions=on_npu)


def profile_summary(profile: Path) -> dict:
    counts, dur = {}, {}
    for event in json.loads(profile.read_text()):
        if event.get("cat") != "Node":
            continue
        provider = event.get("args", {}).get("provider")
        if provider and event.get("name", "").endswith("_kernel_time"):
            counts[provider] = counts.get(provider, 0) + 1
            dur[provider] = dur.get(provider, 0) + event.get("dur", 0)
    return dict(kernel_events=counts, kernel_us_total=dur)


def run_vitisai(model_path: Path, feeds: dict, root: Path, tp_size: int | None = 1, warmup: int = 2,
                iters: int = 10, optimize_level: int = 1, extra_cfg: dict | None = None) -> dict:
    """Compile+run on VitisAI EP. Returns outputs, timing, placement evidence."""
    root.mkdir(parents=True, exist_ok=False)  # fresh dir == fresh compiler cache
    cfg = vaiml_config(root, tp_size, optimize_level, extra_cfg)
    opts = ort.SessionOptions()
    opts.log_severity_level = 2
    opts.enable_profiling = True
    opts.profile_file_prefix = str(root / "profile")
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["VitisAIExecutionProvider"],
                                provider_options=[{"config_file": str(cfg), "cache_dir": str(root / "cache"),
                                                   "enable_cache_file_io_in_mem": "0"}])
    compile_s = time.perf_counter() - t0
    outs = sess.run(None, feeds)
    for _ in range(warmup):
        sess.run(None, feeds)
    times = []
    c0 = time.process_time()
    for _ in range(iters):
        t = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t) * 1e3)
    host_cpu_ms_per_iter = (time.process_time() - c0) * 1e3 / iters  # all host threads, summed
    prof = Path(sess.end_profiling())
    rep = dict(compile_s=compile_s, host_cpu_ms_per_iter=host_cpu_ms_per_iter, wall_ms_min=min(times), wall_ms_median=statistics.median(times),
               wall_ms_all=times, profile=profile_summary(prof), placement=placement(root, model_path),
               loaded_libs=loaded_libs(), tp_size=tp_size, optimize_level=optimize_level)
    rep["npu_executed"] = bool(rep["placement"]["nodes_in_vaiml_partitions"] > 0 and
                               rep["profile"]["kernel_events"].get("VitisAIExecutionProvider", 0) > 0)
    names = [o.name for o in sess.get_outputs()]
    return dict(outputs=dict(zip(names, outs)), report=rep, session=sess)


def run_cpu(model_path: Path, feeds: dict, iters: int = 5) -> tuple[dict, dict]:
    opts = ort.SessionOptions()
    sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    outs = sess.run(None, feeds)
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t) * 1e3)
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, outs)), dict(wall_ms_min=min(times), wall_ms_median=statistics.median(times))
