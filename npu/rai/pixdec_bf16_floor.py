"""BF16 precision floor for the pixel decoder frame graph: torch CPU BF16 vs the FP32 tokens saved by
pixdec_npu.py (same seeds, frame 0, zero history). Usage: python pixdec_bf16_floor.py WEIGHTS ART_DIR"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path.home() / "fastframes-npu/src"))
import pixdec_npu as P  # noqa: E402
from fastframes.models.pixel_decoder import PixelDecoder  # noqa: E402
from npu_common import compare  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

weights, art = sys.argv[1], Path(sys.argv[2])
dec = PixelDecoder(P.DIM)
dec.load_state_dict({k.removeprefix("pixel_head."): v.float() for k, v in load_file(weights).items()})
dec = dec.eval()
g = torch.Generator().manual_seed(1234)
feats = torch.randn(1, 3 * P.TOK, P.DIM, generator=g)
mods = 0.2 * torch.randn(1, 3, 1, P.DIM, generator=g)
with torch.no_grad():
    shift, scale = (dec.modulation.unsqueeze(1) + mods[:, :1]).chunk(2, dim=2)
    decb = dec.to(torch.bfloat16)
    fr = P.PixelDecoderFrame(decb, 3, True)
    hist = [torch.zeros(1, P.DIM, P.H, P.W, dtype=torch.bfloat16) for _ in range(6)]
    out = fr(feats[0, :P.TOK].bfloat16(), shift.reshape(1, P.DIM).bfloat16(), scale.reshape(1, P.DIM).bfloat16(), *hist)
ref = np.load(art / "torch_tokens.npy")[0]
bf = out[0].float().numpy()
u = lambda a: np.round((np.clip(a, -1, 1) + 1) * 127.5)
res = dict(bf16_vs_fp32_tokens=compare(bf, ref),
           bf16_uint8_mean_abs=float(np.abs(u(P.unpatchify(bf)) - u(P.unpatchify(ref))).mean()),
           bf16_uint8_max_abs=float(np.abs(u(P.unpatchify(bf)) - u(P.unpatchify(ref))).max()))
(art / "bf16_floor.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
