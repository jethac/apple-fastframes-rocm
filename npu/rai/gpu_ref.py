"""GPU (Radeon 8060S, ROCm torch, BF16) latency of the *same* per-frame graphs that were run on the NPU.

Rough reference only: eager torch, no CUDA graphs / compile, so it is an upper bound on GPU time.
Run inside a ROCm torch container with ~/fastframes-npu mounted at /w and checkpoints at /ckpt.
"""
import sys
import time
import types

import torch

sys.modules["npu_common"] = types.SimpleNamespace(compare=None, run_cpu=None, run_vitisai=None)
sys.path[:0] = ["/w/rai", "/w/src"]
import dit_block_npu as D  # noqa: E402
import kvref_npu as K  # noqa: E402
import pixdec_npu as P  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

dev, bf = "cuda", torch.bfloat16


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


res = {}
with torch.no_grad():
    from fastframes.models.pixel_decoder import PixelDecoder
    dec = PixelDecoder(1536)
    dec.load_state_dict({k.removeprefix("pixel_head."): v for k, v in
                         load_file("/ckpt/fastframes-s/pixel_decoder.safetensors").items()})
    dec = dec.to(dev, bf).eval()
    fr = P.PixelDecoderFrame(dec, 3, True)
    ins = [torch.randn(1560, 1536, device=dev, dtype=bf), torch.randn(1, 1536, device=dev, dtype=bf),
           torch.randn(1, 1536, device=dev, dtype=bf)] + [torch.randn(1, 1536, 30, 52, device=dev, dtype=bf)
                                                            for _ in range(6)]
    res["pixel_decoder_frame_ms"] = bench(lambda: fr(*ins))

    from fastframes.models.kv_refinement import KVRefinementNetwork
    net = KVRefinementNetwork(d_a=768)
    net.load_state_dict(load_file("/ckpt/fastframes-s/kv_refiner.safetensors"))
    net = net.to(dev, bf).eval()
    e = net.embed_timestep(K.REFINE_T, 1, dev, bf)
    stem = K.StemGraph(net, e)
    x = torch.randn(1560, 1536, device=dev, dtype=bf)
    res["kvref_stem_ms"] = bench(lambda: stem(x))
    head = K.HeadGraph(net, e, 0)
    nk, nv, phi = (torch.randn(1560, d, device=dev, dtype=bf) for d in (1536, 1536, 768))
    res["kvref_head_one_layer_ms"] = bench(lambda: head(nk, nv, phi))
    gm = net.heads_grouped
    nK = torch.randn(30, 1560, 1536, device=dev, dtype=bf)
    res["kvref_heads_all30_upstream_batched_ms"] = bench(lambda: gm(nK, nK, phi[None], e))

    from fastframes.wan.model import WAN_CONFIGS
    import fastframes.wan.causal_model as cm
    cfg = WAN_CONFIGS["Wan2.1-T2V-1.3B"]
    blk = cm.CausalWanAttentionBlock(cfg.dim, cfg.ffn_dim, cfg.num_heads, cfg.qk_norm, cfg.cross_attn_norm, cfg.eps)
    sd = load_file("/w/rai/art/dit/dit_layer0_weights.safetensors")
    blk.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith("blocks.0.")})
    blk = blk.to(dev, bf).eval()
    for f in (0, 4, 20):
        g = D.BlockGraph(blk, torch.ones(1560, 1536, device=dev, dtype=bf), torch.zeros(1560, 1536, device=dev, dtype=bf),
                         f > 0).to(dev, bf)
        a = [torch.randn(1560, 1536, device=dev, dtype=bf), torch.randn(6, 1536, device=dev, dtype=bf),
             torch.randn(512, 1536, device=dev, dtype=bf), torch.randn(512, 1536, device=dev, dtype=bf)]
        if f:
            a += [torch.randn(f * 1560, 1536, device=dev, dtype=bf)] * 2
        res[f"dit_block_frame{f}_ms_eager_math_attention"] = bench(lambda: g(*a), n=10)
print(torch.cuda.get_device_name(0), torch.__version__)
for k, v in res.items():
    print(f"{k}: {v:.2f}")
