"""Re-open an already compiled VAIML cache (no recompilation) and run it in a timed loop.
Used for the GPU+NPU concurrency test. Usage: python npu_loop.py RUN_DIR SECONDS
RUN_DIR is a vai_* directory produced by run_vitisai (holds config.json and cache/)."""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

run, secs = Path(sys.argv[1]), float(sys.argv[2])
model = next(run.parent.glob("*.onnx")) if not (run / "model.onnx").exists() else run / "model.onnx"
sess = ort.InferenceSession(str(model), providers=["VitisAIExecutionProvider"], provider_options=[{
    "config_file": str(run / "config.json"), "cache_dir": str(run / "cache"), "enable_cache_file_io_in_mem": "0"}])
feeds = {i.name: np.random.default_rng(0).standard_normal([int(d) for d in i.shape]).astype(np.float32)
         for i in sess.get_inputs()}
for _ in range(3):
    sess.run(None, feeds)
times, t_end = [], time.time() + secs
while time.time() < t_end:
    t = time.perf_counter()
    sess.run(None, feeds)
    times.append((time.perf_counter() - t) * 1e3)
print(json.dumps(dict(model=model.name, n=len(times), ms_min=min(times), ms_median=statistics.median(times))))
