"""GPU side of the concurrency test: one FastFrames-S DiT block (frame 20, 32760-token KV) in a timed loop,
BF16 eager on ROCm. Usage (in a ROCm torch container, ~/fastframes-npu at /w): python gpu_loop.py SECONDS"""
import json
import statistics
import sys
import time
import types

import torch

sys.modules["npu_common"] = types.SimpleNamespace(compare=None, run_cpu=None, run_vitisai=None)
sys.path[:0] = ["/w/rai", "/w/src"]
import dit_block_npu as D  # noqa: E402
import fastframes.wan.causal_model as cm  # noqa: E402
from fastframes.wan.model import WAN_CONFIGS  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

secs, dev, bf = float(sys.argv[1]), "cuda", torch.bfloat16
cfg = WAN_CONFIGS["Wan2.1-T2V-1.3B"]
blk = cm.CausalWanAttentionBlock(cfg.dim, cfg.ffn_dim, cfg.num_heads, cfg.qk_norm, cfg.cross_attn_norm, cfg.eps)
sd = load_file("/w/rai/art/dit/dit_layer0_weights.safetensors")
blk.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith("blocks.0.")})
blk = blk.to(dev, bf).eval()
f = 20
g = D.BlockGraph(blk, torch.ones(1560, 1536), torch.zeros(1560, 1536), True).to(dev, bf)
a = [torch.randn(1560, 1536, device=dev, dtype=bf), torch.randn(6, 1536, device=dev, dtype=bf),
     torch.randn(512, 1536, device=dev, dtype=bf), torch.randn(512, 1536, device=dev, dtype=bf)]
a += [torch.randn(f * 1560, 1536, device=dev, dtype=bf)] * 2
times = []
with torch.no_grad():
    for _ in range(3):
        g(*a)
    torch.cuda.synchronize()
    t_end = time.time() + secs
    while time.time() < t_end:
        t = time.perf_counter()
        g(*a)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
print(json.dumps(dict(gpu="dit_block_f20", n=len(times), ms_min=min(times), ms_median=statistics.median(times))))
