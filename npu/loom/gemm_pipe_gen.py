#!/usr/bin/env python3
"""Generate a blocked BF16 GEMM as a resident XDNA2 (AIE2P) Loom pipeline.

C[M,N] (f32, row-major) = A[M,K] (bf16) @ B[K,N] (bf16)

Host pre-packs operands into 8-wide panels (same as gemm_gen.py):
  Ap[M/8][K][8]  (Ap[i][k][r] = A[8i+r, k])
  Bp[N/8][K][8]  (Bp[j][k][c] = B[k, 8j+c])

Each worker (one AIE2P compute tile) owns MBxNB output blocks (round-robin).
Per worker there are four strands on that tile:
  * A ingress: per block, per KB-deep k-chunk, DMA an A block [MB/8][KB][8]
    (2-D strided view of Ap) into a channel record in tile memory.
  * B ingress: same for B [NB/8][KB][8].
  * compute: reserves a C record (f32, tile-major [MB/8][NB/8][8][8]),
    accumulates 8x8 subtiles with vector.mma (AIE2P mma.bf16bf16.m8n8k1),
    publishes it when K is exhausted.
  * drain: DMA the C record to DDR with a 4-D strided view (row-major C).
Channels have --capa slots (A/B) and --capc slots (C), so DMA overlaps compute.

Workers: --cols C --rows R places C*R workers at columns [0,C), rows [2,2+R).

Usage: python gemm_pipe_gen.py M K N out.loom [--mb 32 --nb 64 --kb 64 --u 8]
                               [--cols 1 --rows 1] [--capa 2 --capc 1] [--npy dir]
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("M", type=int); ap.add_argument("K", type=int); ap.add_argument("N", type=int)
ap.add_argument("out")
ap.add_argument("--mb", type=int, default=32)
ap.add_argument("--nb", type=int, default=64)
ap.add_argument("--kb", type=int, default=64)
ap.add_argument("--u", type=int, default=8, help="k unroll inside a chunk")
ap.add_argument("--cols", type=int, default=1)
ap.add_argument("--rows", type=int, default=1)
ap.add_argument("--row0", type=int, default=2)
ap.add_argument("--capa", type=int, default=2, help="A/B channel slots")
ap.add_argument("--capc", type=int, default=1, help="C channel slots")
ap.add_argument("--fused", action="store_true",
                help="one strand per worker: compute strand issues its own A/B/C DMAs (no overlap)")
ap.add_argument("--wide", action="store_true", help="load U*8 bf16 per operand at once and slice")
ap.add_argument("--npy", default=None)
a = ap.parse_args()
M, K, N, MB, NB, KB, U = a.M, a.K, a.N, a.mb, a.nb, a.kb, a.u
assert M % MB == 0 and N % NB == 0 and K % KB == 0 and MB % 8 == 0 and NB % 8 == 0 and KB % U == 0
PM, PN = MB // 8, NB // 8           # panels per block
BM, BN = M // MB, N // NB           # blocks
NBLK = BM * BN
W = a.cols * a.rows
assert NBLK >= W, "more workers than blocks"
A_EL, B_EL, C_EL = PM * KB * 8, PN * KB * 8, MB * NB
CAPA, CAPC = a.capa, a.capc
bytes_local = CAPA * (2 * A_EL + 2 * B_EL) + CAPC * 4 * C_EL
assert bytes_local <= 62 * 1024, f"tile memory {bytes_local} B too large"

AT = f"tile<{A_EL}xbf16>"; BT = f"tile<{B_EL}xbf16>"; CT = f"tile<{C_EL}xf32>"
ASV = f"view<{PM}x{KB*8}xbf16, #encoding.layout.strided<strides=[{K*8}, 1]>>"
BSV = f"view<{PN}x{KB*8}xbf16, #encoding.layout.strided<strides=[{K*8}, 1]>>"
# local C is subtile-major with q (column panel) outermost: [PN][PM][8][8] == [PN][MB][8]
CDV = f"view<{PN}x{MB}x8xf32, #encoding.layout.strided<strides=[8, {N}, 1]>>"


def common(wid):
    return f"""
      %origin = index.constant 0 : offset
      %c0 = index.constant 0 : index
      %c1 = index.constant 1 : index
      %c8 = index.constant 8 : index
      %c64 = index.constant 64 : index
      %cKB = index.constant {KB} : index
      %cK = index.constant {K} : index
      %cBN = index.constant {BN} : index
      %wid = index.constant {wid} : index
      %cW = index.constant {W} : index
      %cNBLK = index.constant {NBLK} : index"""


def ingress(col, row, wid, name, ch, T, SV, EL, blk_bytes, idx):
    return f"""
  pipeline.strand workers([{col}, {row}], [1, 1], [1, 1]) {{{common(wid)}
      %blk_bytes = index.constant {blk_bytes} : index
      %kb_bytes = index.constant {KB*8*2} : index
      scf.for %blk = [%wid to %cNBLK step %cW] {{
        %bi = index.div %blk, %cBN : index
        %bj = index.rem %blk, %cBN : index
        %boff = index.mul {idx}, %blk_bytes : index
        scf.for %k0 = [%c0 to %cK step %cKB] {{
          %kchunk = index.div %k0, %cKB : index
          %koff = index.mul %kchunk, %kb_bytes : index
          %off_i = index.add %boff, %koff : index
          %off = index.cast %off_i : index to offset
          %src = buffer.view %{name}[%off] : buffer -> {SV}
          %w, %pay = channel.reserve %{ch} : channel<{T}> -> (write<{T}>, view<{EL}xbf16>)
          %cp = kernel.async.copy %src to %pay {{cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup}} : {SV} to view<{EL}xbf16> -> kernel.async.token
          %g = kernel.async.group %cp : kernel.async.token -> kernel.async.group
          kernel.async.wait %g {{newer_groups = 0}} : kernel.async.group
          channel.publish %w : write<{T}>
        }}
      }}
  }}"""


def compute(col, row, wid):
    steps = []
    if a.wide:
        # one 16*U-byte load per operand, then static lane slices
        steps.append(f"""
                %wa = vector.load %Al[%abase_k] : view<{A_EL}xbf16> -> vector<{U*8}xbf16>
                %wb = vector.load %Bl[%bbase_k] : view<{B_EL}xbf16> -> vector<{U*8}xbf16>""")
    for s in range(U):
        if a.wide:
            ld = f"""
                %va{s} = vector.slice %wa[{8*s}] : vector<{U*8}xbf16> -> vector<8xbf16>
                %vb{s} = vector.slice %wb[{8*s}] : vector<{U*8}xbf16> -> vector<8xbf16>"""
        else:
            ld = f"""
                %ka{s} = index.add %abase_k, %c{s*8}e : index
                %kb{s} = index.add %bbase_k, %c{s*8}e : index
                %va{s} = vector.load %Al[%ka{s}] : view<{A_EL}xbf16> -> vector<8xbf16>
                %vb{s} = vector.load %Bl[%kb{s}] : view<{B_EL}xbf16> -> vector<8xbf16>"""
        steps.append(ld + f"""
                %fa{s} = vector.fragment<lhs> %va{s} shape [%c8, %c1] : vector<8xbf16>
                %fb{s} = vector.fragment<rhs> %vb{s} shape [%c1, %c8] : vector<8xbf16>
                %acc{s+1} = vector.mma %fa{s}, %fb{s}, %acc{s} : vector<8xbf16>, vector<8xbf16>, vector<64xf32>""")
    consts = "".join(f"""
      %c{s*8}e = index.constant {s*8} : index""" for s in range(U))
    return f"""
  pipeline.strand workers([{col}, {row}], [1, 1], [1, 1]) {{{common(wid)}
      %cPM = index.constant {PM} : index
      %cPN = index.constant {PN} : index
      %cCEL = index.constant {C_EL} : index
      %cKB8 = index.constant {KB*8} : index
      %cU8 = index.constant {U*8} : index{consts}
      %zero = vector.splatf 0.0 : vector<64xf32>
      scf.for %blk = [%wid to %cNBLK step %cW] {{
        %wc, %Cl = channel.reserve %cch_{wid} : channel<{CT}> -> (write<{CT}>, view<{C_EL}xf32>)
        scf.for %z = [%c0 to %cCEL step %c64] {{
          vector.store %zero, %Cl[%z] : vector<64xf32>, view<{C_EL}xf32>
        }}
        scf.for %k0 = [%c0 to %cK step %cKB] {{{fused_fill(wid) if a.fused else ""}
          %ra, %Al = channel.acquire %ach_{wid} : channel<{AT}> -> (read<{AT}>, view<{A_EL}xbf16>)
          %rb, %Bl = channel.acquire %bch_{wid} : channel<{BT}> -> (read<{BT}>, view<{B_EL}xbf16>)
          scf.for %p = [%c0 to %cPM step %c1] {{
            %abase = index.mul %p, %cKB8 : index
            scf.for %q = [%c0 to %cPN step %c1] {{
              %bbase = index.mul %q, %cKB8 : index
              %tile = index.mul %q, %cPM : index
              %tile2 = index.add %tile, %p : index
              %coff = index.mul %tile2, %c64 : index
              %cv = vector.load %Cl[%coff] : view<{C_EL}xf32> -> vector<64xf32>
              %init = vector.fragment<init> %cv shape [%c8, %c8] : vector<64xf32>
              %res = scf.for %kk = [%c0 to %cKB8 step %cU8](%acc0 = %init : vector<64xf32>) -> (vector<64xf32>) {{
                %abase_k = index.add %abase, %kk : index
                %bbase_k = index.add %bbase, %kk : index{"".join(steps)}
                scf.yield %acc{U} : vector<64xf32>
              }}
              vector.store %res, %Cl[%coff] : vector<64xf32>, view<{C_EL}xf32>
            }}
          }}
          channel.release %ra : read<{AT}>
          channel.release %rb : read<{BT}>
        }}
        channel.publish %wc : write<{CT}>{fused_drain(wid) if a.fused else ""}
      }}
  }}"""


def fused_fill(wid):
    """In-strand DMA of the A and B k-chunk (no separate ingress strands)."""
    return f"""
          %bi_f = index.div %blk, %cBN : index
          %bj_f = index.rem %blk, %cBN : index
          %a_blk_bytes = index.constant {PM*K*8*2} : index
          %b_blk_bytes = index.constant {PN*K*8*2} : index
          %kb_bytes = index.constant {KB*8*2} : index
          %kchunk = index.div %k0, %cKB : index
          %koff = index.mul %kchunk, %kb_bytes : index
          %aoff0 = index.mul %bi_f, %a_blk_bytes : index
          %aoff1 = index.add %aoff0, %koff : index
          %aoff = index.cast %aoff1 : index to offset
          %boff0 = index.mul %bj_f, %b_blk_bytes : index
          %boff1 = index.add %boff0, %koff : index
          %boff = index.cast %boff1 : index to offset
          %asrc = buffer.view %ap[%aoff] : buffer -> {ASV}
          %bsrc = buffer.view %bp[%boff] : buffer -> {BSV}
          %wa, %pa = channel.reserve %ach_{wid} : channel<{AT}> -> (write<{AT}>, view<{A_EL}xbf16>)
          %wb, %pb = channel.reserve %bch_{wid} : channel<{BT}> -> (write<{BT}>, view<{B_EL}xbf16>)
          %cpa = kernel.async.copy %asrc to %pa {{cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup}} : {ASV} to view<{A_EL}xbf16> -> kernel.async.token
          %cpb = kernel.async.copy %bsrc to %pb {{cache_scope = workgroup, cache_temporal = regular, direction = global_to_workgroup}} : {BSV} to view<{B_EL}xbf16> -> kernel.async.token
          %gab = kernel.async.group %cpa, %cpb : kernel.async.token, kernel.async.token -> kernel.async.group
          kernel.async.wait %gab {{newer_groups = 0}} : kernel.async.group
          channel.publish %wa : write<{AT}>
          channel.publish %wb : write<{BT}>"""


def fused_drain(wid):
    """In-strand DMA of the finished C block to DDR."""
    return f"""
        %bi_d = index.div %blk, %cBN : index
        %bj_d = index.rem %blk, %cBN : index
        %c_row_block_bytes = index.constant {MB*N*4} : index
        %c_col_block_bytes = index.constant {NB*4} : index
        %rc, %Cr = channel.acquire %cch_{wid} : channel<{CT}> -> (read<{CT}>, view<{C_EL}xf32>)
        %c_off_r = index.mul %bi_d, %c_row_block_bytes : index
        %c_off_c = index.mul %bj_d, %c_col_block_bytes : index
        %c_off_i = index.add %c_off_r, %c_off_c : index
        %c_off = index.cast %c_off_i : index to offset
        %cdst = buffer.view %cb[%c_off] : buffer -> {CDV}
        %cpc = kernel.async.copy %Cr to %cdst {{cache_scope = workgroup, cache_temporal = regular, direction = workgroup_to_global}} : view<{C_EL}xf32> to {CDV} -> kernel.async.token
        %gc = kernel.async.group %cpc : kernel.async.token -> kernel.async.group
        kernel.async.wait %gc {{newer_groups = 0}} : kernel.async.group
        channel.release %rc : read<{CT}>"""


def drain(col, row, wid):
    return f"""
  pipeline.strand workers([{col}, {row}], [1, 1], [1, 1]) {{{common(wid)}
      %c_row_block_bytes = index.constant {MB*N*4} : index
      %c_col_block_bytes = index.constant {NB*4} : index
      scf.for %blk = [%wid to %cNBLK step %cW] {{
        %bi = index.div %blk, %cBN : index
        %bj = index.rem %blk, %cBN : index
        %rc, %Cl = channel.acquire %cch_{wid} : channel<{CT}> -> (read<{CT}>, view<{C_EL}xf32>)
        %c_off_r = index.mul %bi, %c_row_block_bytes : index
        %c_off_c = index.mul %bj, %c_col_block_bytes : index
        %c_off_i = index.add %c_off_r, %c_off_c : index
        %c_off = index.cast %c_off_i : index to offset
        %cdst = buffer.view %cb[%c_off] : buffer -> {CDV}
        %cp = kernel.async.copy %Cl to %cdst {{cache_scope = workgroup, cache_temporal = regular, direction = workgroup_to_global}} : view<{C_EL}xf32> to {CDV} -> kernel.async.token
        %g = kernel.async.group %cp : kernel.async.token -> kernel.async.group
        kernel.async.wait %g {{newer_groups = 0}} : kernel.async.group
        channel.release %rc : read<{CT}>
      }}
  }}"""


allocs, strands = [], []
wid = 0
for col in range(a.cols):
    for r in range(a.rows):
        row = a.row0 + r
        allocs.append(f"""
  %mem_{wid} = pipeline.memory<workgroup>[{col}, {row}] : pool
  %abuf_{wid} = buffer.alloca<workgroup> pool(%mem_{wid}) align(64) %a_bytes : buffer
  %bbuf_{wid} = buffer.alloca<workgroup> pool(%mem_{wid}) align(64) %b_bytes : buffer
  %cbuf_{wid} = buffer.alloca<workgroup> pool(%mem_{wid}) align(64) %c_bytes : buffer
  %aslots_{wid} = buffer.view %abuf_{wid}[%org] : buffer -> view<{CAPA}x{A_EL}xbf16>
  %ach_{wid} = channel.bind<fifo> %aslots_{wid}, %capa : view<{CAPA}x{A_EL}xbf16>, index -> channel<{AT}>
  %bslots_{wid} = buffer.view %bbuf_{wid}[%org] : buffer -> view<{CAPA}x{B_EL}xbf16>
  %bch_{wid} = channel.bind<fifo> %bslots_{wid}, %capa : view<{CAPA}x{B_EL}xbf16>, index -> channel<{BT}>
  %cslots_{wid} = buffer.view %cbuf_{wid}[%org] : buffer -> view<{CAPC}x{C_EL}xf32>
  %cch_{wid} = channel.bind<fifo> %cslots_{wid}, %capc : view<{CAPC}x{C_EL}xf32>, index -> channel<{CT}>""")
        if not a.fused:
            strands.append(ingress(col, row, wid, "ap", f"ach_{wid}", AT, ASV, A_EL, PM*K*8*2, "%bi"))
            strands.append(ingress(col, row, wid, "bp", f"bch_{wid}", BT, BSV, B_EL, PN*K*8*2, "%bj"))
        strands.append(compute(col, row, wid))
        if not a.fused:
            strands.append(drain(col, row, wid))
        wid += 1

src = f"""// Generated by gemm_pipe_gen.py: fused={a.fused} wide={a.wide} M={M} K={K} N={N} MB={MB} NB={NB} KB={KB} U={U} cols={a.cols} rows={a.rows} capa={CAPA} capc={CAPC}
aie2p.target<array> @npu

pipeline.def<kernel> public retain target(@npu) @gemm_bf16() run(%ap_arg: buffer, %bp_arg: buffer, %c_arg: buffer) {{
  %ap_u, %bp_u, %c_u = buffer.assume.noalias %ap_arg, %bp_arg, %c_arg : buffer, buffer, buffer
  %ap, %bp, %cb = buffer.assume.alignment %ap_u, %bp_u, %c_u {{minimum_alignment = 64}} : buffer, buffer, buffer
  %org = index.constant 0 : offset
  %capa = index.constant {CAPA} : index
  %capc = index.constant {CAPC} : index
  %a_bytes = index.constant {CAPA*2*A_EL} : offset
  %b_bytes = index.constant {CAPA*2*B_EL} : offset
  %c_bytes = index.constant {CAPC*4*C_EL} : offset{"".join(allocs)}
{"".join(strands)}
}}
"""
if a.npy:
    TM, TN = M // 8, N // 8
    src += f"""
check.case public @gemm_npy {{
  %ap = check.file.read.npy path("{a.npy}/ap.npy") : tensor<{TM*K*8}xi16>
  %bp = check.file.read.npy path("{a.npy}/bp.npy") : tensor<{TN*K*8}xi16>
  %c = check.generate.fill value(-777.0) : tensor<{M*N}xf32>
  kernel.launch @gemm_bf16(%ap, %bp, %c) : (tensor<{TM*K*8}xi16>, tensor<{TN*K*8}xi16>, tensor<{M*N}xf32>)
  check.file.write.npy value(%c) path("c.npy") mode(always) : tensor<{M*N}xf32>
  check.return
}}

check.benchmark<@gemm_npy> @gemm_bench
"""
open(a.out, "w").write(src)
