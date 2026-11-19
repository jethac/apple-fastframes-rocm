set -x
./run.sh pixdec_npu.py --weights /home/jethac/models/fastframes/checkpoints/fastframes-s/pixel_decoder.safetensors --out art/pixdec --tp-size 1 --opt-level 2 > art/pixdec_tp1_o2.log 2>&1
./run.sh kvref_npu.py --weights /home/jethac/models/fastframes/checkpoints/fastframes-s/kv_refiner.safetensors --part head --layer 0 --out art/kvref --tp-size 1 --opt-level 2 > art/kvref_head_tp1_o2.log 2>&1
./run.sh kvref_npu.py --weights /home/jethac/models/fastframes/checkpoints/fastframes-s/kv_refiner.safetensors --part stem --out art/kvref --tp-size 1 --opt-level 2 > art/kvref_stem_tp1_o2.log 2>&1
./run.sh dit_block_npu.py --ckpt /home/jethac/models/fastframes/checkpoints/causal-forcing++/framewise-2step.pt --frame 0 --out art/dit --tp-size 1 --opt-level 2 > art/dit_f0_tp1_o2.log 2>&1
./run.sh dit_block_npu.py --ckpt /home/jethac/models/fastframes/checkpoints/causal-forcing++/framewise-2step.pt --frame 20 --out art/dit --tp-size 1 --opt-level 2 > art/dit_f20_tp1_o2.log 2>&1
./run.sh gemm_probe.py --out art/gemm --tp-size 0 --opt-level 2 > art/gemm_tp0_o2.log 2>&1
echo QUEUE_DONE
