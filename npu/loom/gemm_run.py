#!/usr/bin/env python3
"""Generate, run (loom-test --device=xdna) and verify a BF16 GEMM on the XDNA2 NPU.

Usage:
  python gemm_run.py M K N workdir [--mode identity|random] [--kc 32] [--grid G]
                     [--device xdna] [--bench N]

identity mode (stamp-bug probe): A[m,k] = (m*K + k) % 16 + 1, B = I (K==N).
Expect C == A exactly.  random mode: A,B ~ U(-1,1) rounded to bf16.

Reports max abs err, cosine similarity vs numpy fp32/fp64 reference, all-zero
column blocks (any 8-wide / 64-wide / 384-wide column block of C that is all
zero while the reference is not), and wall time of the loom-test invocation.
With --bench N also runs loom-benchmark (dispatch_complete) for device timing.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def bf16_round(x):
    x = np.ascontiguousarray(np.asarray(x, np.float32))
    b = x.view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.astype(np.uint32).view(np.float32)


def bf16_bits(x):
    return (bf16_round(x).view(np.uint32) >> 16).astype(np.uint16).view(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("M", type=int); ap.add_argument("K", type=int); ap.add_argument("N", type=int)
    ap.add_argument("wd")
    ap.add_argument("--mode", default="random")
    ap.add_argument("--kc", type=int, default=32)
    ap.add_argument("--grid", type=int, default=1)
    ap.add_argument("--device", default="xdna")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--gen", default="pipe", choices=["pipe", "kernel"],
                    help="pipe: gemm_pipe_gen.py (DMA-blocked pipeline); kernel: gemm_gen.py (kernel.def, <64KB)")
    a, extra = ap.parse_known_args()
    M, K, N = a.M, a.K, a.N
    wd = os.path.abspath(a.wd); os.makedirs(wd, exist_ok=True)

    if a.mode == "identity":
        assert K == N
        A = ((np.arange(M * K).reshape(M, K) % 16) + 1).astype(np.float32)
        B = np.eye(K, dtype=np.float32)
    else:
        rng = np.random.default_rng(1234)
        A = bf16_round(rng.uniform(-1, 1, (M, K))).reshape(M, K)
        B = bf16_round(rng.uniform(-1, 1, (K, N))).reshape(K, N)
    # pack into 8-wide panels: Ap[i][k][r] = A[8i+r,k], Bp[j][k][c] = B[k,8j+c]
    Ap = A.reshape(M // 8, 8, K).transpose(0, 2, 1).reshape(-1)
    Bp = B.reshape(K, N // 8, 8).transpose(1, 0, 2).reshape(-1)
    np.save(f"{wd}/ap.npy", bf16_bits(Ap)); np.save(f"{wd}/bp.npy", bf16_bits(Bp))
    ref = A.astype(np.float64) @ B.astype(np.float64)

    src = f"{wd}/gemm.loom"
    if a.gen == "kernel":
        gargs = [f"{HERE}/gemm_gen.py", str(M), str(K), str(N), src, "--kc", str(a.kc), "--grid", str(a.grid)]
    else:
        gargs = [f"{HERE}/gemm_pipe_gen.py", str(M), str(K), str(N), src] + extra
    subprocess.check_call([sys.executable] + gargs + ["--npy", wd])
    if os.path.exists(f"{wd}/c.npy"):
        os.remove(f"{wd}/c.npy")
    cmd = ["loom-test", src, f"--device={a.device}", f"--file-output-root={wd}", "--case=@gemm_npy"]
    print("$", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
    wall = time.perf_counter() - t0
    open(f"{wd}/loom-test.stdout.json", "w").write(p.stdout)
    open(f"{wd}/loom-test.stderr.txt", "w").write(p.stderr)
    print(f"loom-test rc={p.returncode} wall={wall:.3f}s (includes JIT compile + staging + readback)")
    if p.returncode != 0 or not os.path.exists(f"{wd}/c.npy"):
        print(p.stdout[-4000:]); print(p.stderr[-4000:])
        sys.exit(1)
    C = np.load(f"{wd}/c.npy").reshape(M, N).astype(np.float64)

    err = np.abs(C - ref)
    cos = float((C * ref).sum() / (np.linalg.norm(C) * np.linalg.norm(ref) + 1e-300))
    res = {"M": M, "K": K, "N": N, "mode": a.mode, "device": a.device, "gen": a.gen, "gen_args": extra,
           "max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
           "max_abs_ref": float(np.abs(ref).max()), "cosine": cos,
           "exact_match": bool(np.array_equal(C, ref)),
           "untouched_sentinel_count": int((C == -777.0).sum()),
           "loom_test_wall_s": wall}
    for blk in (8, 64, 384):
        if N % blk == 0:
            zc = [j for j in range(N // blk)
                  if np.all(C[:, j*blk:(j+1)*blk] == 0) and not np.all(ref[:, j*blk:(j+1)*blk] == 0)]
            res[f"zero_col_blocks_{blk}"] = zc
    res["zero_columns"] = int(sum(1 for j in range(N) if np.all(C[:, j] == 0) and not np.all(ref[:, j] == 0)))
    res["zero_rows"] = int(sum(1 for i in range(M) if np.all(C[i] == 0) and not np.all(ref[i] == 0)))

    if a.bench:
        out = f"{wd}/bench.json"
        bcmd = ["loom-benchmark", src, f"--device={a.device}", "--benchmark=@gemm_bench",
                "--measure=dispatch_complete", f"--output={out}"]
        print("$", " ".join(bcmd), flush=True)
        bp = subprocess.run(bcmd, capture_output=True, text=True, timeout=a.timeout)
        open(f"{wd}/bench.stderr.txt", "w").write(bp.stderr + bp.stdout)
        print(f"loom-benchmark rc={bp.returncode}")
        res["bench_rc"] = bp.returncode
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{wd}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
