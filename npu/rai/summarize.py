"""Collect every report_*.json under an artifact root into one compact table (stdout + summary.json)."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "art")
rows = []
for p in sorted(root.rglob("report_*.json")):
    r = json.loads(p.read_text())
    n = r.get("npu", {})
    cmp = (r.get("npu_vs_upstream_fp32") or {}) or {}
    if "npu_vs_torch_tokens" in r:
        cmp = {"tokens_f0": r["npu_vs_torch_tokens"][0]}
    if "cmp" in r:
        cmp = {"y": r["cmp"]}
    floor = r.get("torch_bf16_vs_fp32") or r.get("torch_bf16_release_vs_fp32") or {}
    rows.append(dict(
        report=str(p.relative_to(root)),
        component=r.get("component", "gemm"), frame=r.get("frame"), tp_size=r.get("tp_size"),
        opt_level=n.get("optimize_level"),
        npu_executed=n.get("npu_executed"),
        nodes=f'{n.get("placement", {}).get("nodes_in_vaiml_partitions")}/{n.get("placement", {}).get("total_onnx_nodes")}',
        npu_ms_min=round(n.get("wall_ms_min", float("nan")), 1),
        host_cpu_ms=round(n["host_cpu_ms_per_iter"], 1) if "host_cpu_ms_per_iter" in n else None,
        cpu_ort_fp32_ms=round(r["onnx_cpu_fp32_ms"]["wall_ms_min"], 1) if "onnx_cpu_fp32_ms" in r else None,
        npu_vs_ref={k: dict(rel_l2=round(c["rel_l2"], 4), cos=round(c["cosine"], 5),
                            zero_cols=c["zero_column_ranges"]) for k, c in cmp.items()},
        bf16_floor_rel_l2={k: round(c["rel_l2"], 4) for k, c in floor.items()},
        uint8_mean_abs=r.get("npu_pixel_uint8_mean_abs_diff"),
    ))
(root / "summary.json").write_text(json.dumps(rows, indent=1))
for row in rows:
    print(json.dumps(row))
