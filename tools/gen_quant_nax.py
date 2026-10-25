#!/usr/bin/env python3
"""Generates the embedded MSL for the MoE grouped Q4 GEMM from MLX 0.32.3's
NAX quantized-matmul headers in third_party/.

The engine compiles MSL from one embedded string at run time, so this flattens
`quantized_nax.h` (and the steel NAX body it uses) into one source, then adds
this engine's block-mapped entry points. The block map entry gives the output
row tile [m0, m_end), the expert's weight-row base (expert*N + n0), and the
output column n0, exactly as `lfm_moe_blocks_parallel` writes it. The kernel
body is MLX's `qmm_t_nax_tgp_impl`, whose padded threadgroup B staging
(BK + 16/sizeof(T)) and SK=32 `tile_matmad_nax` loop the reference runs.

The generated output is committed; rerun this script after changing the pinned
header set or the entry points.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLX = os.path.join(ROOT, "third_party", "mlx-0.32.3")
OUT = os.path.join(ROOT, "src", "kernels", "metal", "quant_nax.metal")

INCLUDE_RE = re.compile(r'^\s*#include\s+([<"])([^>"]+)[>"]\s*$')


def flatten(entry, prelude):
    system_includes, body, visited = [], [], set()

    def emit(path):
        if path in visited:
            return
        visited.add(path)
        with open(os.path.join(MLX, path)) as f:
            lines = f.readlines()
        body.append("// ---- %s\n" % path)
        for line in lines:
            m = INCLUDE_RE.match(line)
            if m and m.group(1) == '<':
                norm = line.strip()
                if norm not in [x.strip() for x in system_includes]:
                    system_includes.append(norm + "\n")
            elif m:
                emit(m.group(2))
            elif line.strip() == '#pragma once':
                continue
            else:
                body.append(line)
        body.append("\n")

    for path in prelude:
        emit(path)
    emit(entry)
    return system_includes, "".join(body)


def mlx_qmm_t_body():
    """MLX's qmm_t_nax_tgp_impl with K/N/M taken by value.

    The original takes them as `const constant int&`, which a kernel can only
    bind from its own constant buffer; the grouped entry point knows M from the
    block map, so the body is copied verbatim with those three parameters by
    value.
    """
    raw = open(os.path.join(MLX, "mlx/backend/metal/kernels/quantized_nax.h")).read()
    start = raw.index("METAL_FUNC void qmm_t_nax_tgp_impl(")
    start = raw.rindex("template <", 0, start)
    sig_end = raw.index(")", raw.index("uint simd_lid", start))
    brace = raw.index("{", sig_end)
    depth = 0
    i = brace
    while True:
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = raw[start : i + 1]
    body = body.replace("qmm_t_nax_tgp_impl(", "lfm_qmm_t_body(")
    body = body.replace("const constant int& K,", "const int K,")
    body = body.replace("const constant int& N,", "const int N,")
    body = body.replace("const constant int& M,", "const int M,")
    return body



def mlx_qmm_t_shared_body():
    """MLX's qmm_t NAX body with the per-sub-tile MMA and store guarded.

    A 128-row block-map tile is still one threadgroup, but the four simdgroups
    that own rows beyond the expert segment (sgp_sm <= 0) skip only their own
    tensor-op work and store; the weight staging and its barriers stay
    threadgroup-uniform. Live sub-tiles therefore run MLX's body verbatim.
    """
    body = mlx_qmm_t_body().replace(
        "lfm_qmm_t_body(", "lfm_qmm_t_shared_body("
    )
    loop_head = (
        "        STEEL_PRAGMA_NO_UNROLL\n"
        "        for (int kk1 = 0; kk1 < BK; kk1 += SK) {\n"
    )
    start = body.index(loop_head)
    brace = body.index("{", start)
    depth = 0
    i = brace
    while True:
        if body[i] == "{":
            depth += 1
        elif body[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    loop = body[start:i + 1]
    guarded = "        if (sgp_sm > 0) {\n" + loop + "\n        }\n"
    body = body[:start] + guarded + body[i + 1:]

    store_head = "      if constexpr (kAlignedM.value && kAlignedN.value) {\n"
    start = body.index(store_head)
    brace = body.index("{", start)
    depth = 0
    i = brace
    while True:
        if body[i] == "{":
            depth += 1
        elif body[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    # include the else-if / else chain that follows the first closing brace
    rest = body[i + 1:]
    tail_prefix = " else if (kAlignedM.value && sgp_sn == SN) {"
    if rest.startswith(tail_prefix):
        j = body.index("{", i + 1)
        depth = 0
        k = j
        while True:
            if body[k] == "{":
                depth += 1
            elif body[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        i = k
        rest = body[i + 1:]
        tail_prefix = " else {"
        if rest.startswith(tail_prefix):
            j = body.index("{", i + 1)
            depth = 0
            k = j
            while True:
                if body[k] == "{":
                    depth += 1
                elif body[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            i = k
    store = body[start:i + 1]
    guarded = "      if (sgp_sm > 0) {\n" + store + "\n      }\n"
    body = body[:start] + guarded + body[i + 1:]
    return body


HOST = r'''

// ---------------------------------------------------------------------------
// Entry points for this engine's grouped block map. BM matches the block map's
// row tile (32 for chunks of at most 512 rows; larger chunks use the 128-row
// shared-B entry below); BN=BK=64 and WM=WN=2 are MLX's qmm_t NAX tile. Each
// block entry is one (row tile, column tile) threadgroup, so the arithmetic per
// output element is MLX's and the staged B tile is exactly the dequantized
// BF16 weight tile.
template <int BM>
static inline void lfm_grouped_mlx_body(
    const device uint32_t* w,
    const device bfloat16_t* scales,
    const device bfloat16_t* biases,
    const device bfloat16_t* a,
    device bfloat16_t* c,
    const device uint4* blocks,
    threadgroup bfloat16_t* Ws,
    uint K, uint N, uint bid,
    uint lid, uint simd_gid, uint simd_lid) {
  constexpr int BK = 64;
  constexpr int BN = 64;
  uint4 blk = blocks[bid];
  if (blk.x >= blk.w) {
    return;
  }
  const uint m0 = blk.x;
  const uint b_row0 = blk.y;
  const uint n0 = blk.z;
  const uint m_end = blk.w;
  const int K_w = int(K) / 2;  // packed 4-bit bytes per weight row
  const int K_g = int(K) / 64;
  // b_row0 already includes n0; the body adds y_col = tid.x*BN = n0.
  const ulong expert_row = (ulong)(b_row0 - n0);
  const device uint32_t* wb =
      (const device uint32_t*)((const device uint8_t*)w + expert_row * K_w);
  const device bfloat16_t* sb = scales + expert_row * K_g;
  const device bfloat16_t* bb = biases + expert_row * K_g;
  lfm_qmm_t_body<
      bfloat16_t, 64, 4, true, BM, BK, BN, 2, 2>(
      wb, sb, bb,
      a + (ulong)m0 * K,
      c + (ulong)m0 * N,
      Ws, int(K), int(N), int(m_end - m0),
      uint3(n0 / BN, 0, 0), lid, simd_gid, simd_lid);
}

// ---------------------------------------------------------------------------
// Shared-B entry: one threadgroup per 128-row block-map tile (WM=4), so the
// staged 64x64 weight tile feeds every row of the tile. The column-stripe
// count is a parameter: WN=1 gives one 64-column stripe per threadgroup (128
// threads), so the activation tile is loaded once per tile instead of once per
// stripe, at twice the tensor-op tiles per simdgroup of a two-stripe tile.
template <int BM, int BN_, int WM, int WN>
static inline void lfm_grouped_mlx_shared_entry(
    const device uint32_t* w,
    const device bfloat16_t* scales,
    const device bfloat16_t* biases,
    const device bfloat16_t* a,
    device bfloat16_t* c,
    const device uint4* blocks,
    threadgroup bfloat16_t* Ws,
    uint K, uint N, uint bid,
    uint lid, uint simd_gid, uint simd_lid) {
  constexpr int BK = 64;
  constexpr int BN = BN_;
  uint4 blk = blocks[bid];
  if (blk.x >= blk.w) {
    return;
  }
  const uint m0 = blk.x;
  const uint b_row0 = blk.y;
  const uint n0 = blk.z;
  const uint m_end = blk.w;
  const int K_w = int(K) / 2;
  const int K_g = int(K) / 64;
  const ulong expert_row = (ulong)(b_row0 - n0);
  const device uint32_t* wb =
      (const device uint32_t*)((const device uint8_t*)w + expert_row * K_w);
  const device bfloat16_t* sb = scales + expert_row * K_g;
  const device bfloat16_t* bb = biases + expert_row * K_g;
  lfm_qmm_t_shared_body<
      bfloat16_t, 64, 4, true, BM, BK, BN, WM, WN>(
      wb, sb, bb,
      a + (ulong)m0 * K,
      c + (ulong)m0 * N,
      Ws, int(K), int(N), int(m_end - m0),
      uint3(n0 / BN, 0, 0), lid, simd_gid, simd_lid);
}

#define LFM_Q4_GROUPED_MLX_SHARED(NAME, BM, BN, WM, WN)                       \
    kernel void NAME(const device uint32_t* w        [[buffer(0)]],           \
                     const device bfloat16_t* scales [[buffer(1)]],           \
                     const device bfloat16_t* biases [[buffer(2)]],           \
                     const device bfloat16_t* a      [[buffer(3)]],           \
                     device bfloat16_t* c            [[buffer(4)]],           \
                     const device uint4* blocks      [[buffer(5)]],           \
                     constant uint& K                [[buffer(6)]],           \
                     constant uint& N                [[buffer(7)]],           \
                     uint bid [[threadgroup_position_in_grid]],               \
                     uint lid [[thread_index_in_threadgroup]],                \
                     uint simd_gid [[simdgroup_index_in_threadgroup]],        \
                     uint simd_lid [[thread_index_in_simdgroup]]) {           \
        threadgroup bfloat16_t Ws[(BN) * (64 + 16 / sizeof(bfloat16_t))];     \
        lfm_grouped_mlx_shared_entry<(BM), (BN), (WM), (WN)>(                 \
            w, scales, biases, a, c, blocks, Ws, K, N,                        \
            bid, lid, simd_gid, simd_lid);                                    \
    }

// One column stripe per threadgroup: the activation tile is staged once per
// 128-row tile instead of once per stripe, and each simdgroup owns twice the
// tensor-op tiles.
LFM_Q4_GROUPED_MLX_SHARED(lfm_q4_grouped_mlx_s128w1, 128, 64, 4, 1)

#define LFM_Q4_GROUPED_MLX(NAME, BM)                                           \
    kernel void NAME(const device uint32_t* w        [[buffer(0)]],            \
                     const device bfloat16_t* scales [[buffer(1)]],            \
                     const device bfloat16_t* biases [[buffer(2)]],            \
                     const device bfloat16_t* a      [[buffer(3)]],            \
                     device bfloat16_t* c            [[buffer(4)]],            \
                     const device uint4* blocks      [[buffer(5)]],            \
                     constant uint& K                [[buffer(6)]],            \
                     constant uint& N                [[buffer(7)]],            \
                     uint bid [[threadgroup_position_in_grid]],                \
                     uint lid [[thread_index_in_threadgroup]],                 \
                     uint simd_gid [[simdgroup_index_in_threadgroup]],         \
                     uint simd_lid [[thread_index_in_simdgroup]]) {            \
        threadgroup bfloat16_t Ws[64 * (64 + 16 / sizeof(bfloat16_t))];        \
        lfm_grouped_mlx_body<(BM)>(w, scales, biases, a, c, blocks, Ws, K, N,  \
                                   bid, lid, simd_gid, simd_lid);              \
    }


// ---------------------------------------------------------------------------
// Dual-weight entry: computes the gate and up expert GEMMs for the same block
// map tile in one threadgroup. The two weight streams are staged side by side
// into separate threadgroup buffers, but the activation tile is loaded once
// per K slice and fed to both tensor-op accumulators. Each accumulator runs
// MLX's qmm_t body's per-slice MMA, so every gate/up value is bitwise the
// separate-GEMM result; only the A loading is shared.
template <int BM, int WM, int WN>
static inline void lfm_grouped_mlx_dual_body(
    const device uint32_t* wg,
    const device bfloat16_t* sg,
    const device bfloat16_t* bg,
    const device uint32_t* wu,
    const device bfloat16_t* su,
    const device bfloat16_t* bu,
    const device bfloat16_t* a,
    device bfloat16_t* yg,
    device bfloat16_t* yu,
    const device uint4* blocks,
    threadgroup bfloat16_t* Ws_g,
    threadgroup bfloat16_t* Ws_u,
    uint K, uint N, uint bid,
    uint lid, uint simd_gid, uint simd_lid) {
  constexpr int BK = 64;
  constexpr int BN = 64;
  uint4 blk = blocks[bid];
  if (blk.x >= blk.w) {
    return;
  }
  const uint m0 = blk.x;
  const uint b_row0 = blk.y;
  const uint n0 = blk.z;
  const uint m_end = blk.w;
  const int K_w = int(K) / 2;
  const int K_g = int(K) / 64;
  const ulong expert_row = (ulong)(b_row0 - n0);
  const device uint32_t* wbg =
      (const device uint32_t*)((const device uint8_t*)wg + expert_row * K_w);
  const device bfloat16_t* sbg = sg + expert_row * K_g;
  const device bfloat16_t* bbg = bg + expert_row * K_g;
  const device uint32_t* wbu =
      (const device uint32_t*)((const device uint8_t*)wu + expert_row * K_w);
  const device bfloat16_t* sbu = su + expert_row * K_g;
  const device bfloat16_t* bbu = bu + expert_row * K_g;

  constexpr int BK_padded = (BK + 16 / sizeof(bfloat16_t));
  using loader_t = QuantizedBlockLoader<
      bfloat16_t, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, 64, 4>;

  const int y_row = 0;
  const int y_col = int(n0);
  a += (ulong)m0 * K;
  const device uint8_t* wg0 = (const device uint8_t*)wbg + y_col * K_w;
  const device uint8_t* wu0 = (const device uint8_t*)wbu + y_col * K_w;
  const device bfloat16_t* sg0 = sbg + y_col * K_g;
  const device bfloat16_t* bg0 = bbg + y_col * K_g;
  const device bfloat16_t* su0 = sbu + y_col * K_g;
  const device bfloat16_t* bu0 = bbu + y_col * K_g;
  device bfloat16_t* yg0 = yg + (ulong)m0 * N + y_col;
  device bfloat16_t* yu0 = yu + (ulong)m0 * N + y_col;

  loader_t loader_g(wg0, sg0, bg0, K, Ws_g, simd_gid, simd_lid);
  loader_t loader_u(wu0, su0, bu0, K, Ws_u, simd_gid, simd_lid);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;

  constexpr bool aligned_N = true;
  const short sgp_sm = min(int(SM), int(m_end - m0) - (y_row + tm));
  const bool is_unaligned_sm = (sgp_sm != SM);

  const short sgp_sn = SN;

  const short tgp_bn = BN;
  const bool is_unaligned_bn = false;

  NAXTile<float, TM, TN> Dg;
  NAXTile<float, TM, TN> Du;
  Dg.clear();
  Du.clear();

  a += tm * K;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) {
          loader_g.load_unsafe();
          loader_u.load_unsafe();
        } else {
          loader_g.load_safe(short2(BK, tgp_bn));
          loader_u.load_safe(short2(BK, tgp_bn));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgp_sm > 0) {
        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          NAXTile<bfloat16_t, TM, TK> Atile;
          NAXTile<bfloat16_t, TN, TK> Btg;
          NAXTile<bfloat16_t, TN, TK> Btu;

          volatile int compiler_barrier;

          if constexpr (kAlignedM.value) {
            Atile.load(a + kk1, K);
          } else {
            Atile.load_safe(a + kk1, K, short2(SK, sgp_sm));
          }

          Btg.template load<bfloat16_t, BK_padded, 1>(Ws_g + tn * BK_padded + kk1);
          Btu.template load<bfloat16_t, BK_padded, 1>(Ws_u + tn * BK_padded + kk1);

          tile_matmad_nax(
              Dg, Atile, metal::bool_constant<transpose_a>{}, Btg,
              metal::bool_constant<transpose_b>{});
          tile_matmad_nax(
              Du, Atile, metal::bool_constant<transpose_a>{}, Btu,
              metal::bool_constant<transpose_b>{});

          (void)compiler_barrier;
        }
        }

        a += BK;
        loader_g.next();
        loader_u.next();
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sgp_sm > 0) {
      if constexpr (kAlignedM.value && kAlignedN.value) {
        Dg.store(yg0 + tm * N + tn, N);
        Du.store(yu0 + tm * N + tn, N);
      } else if (kAlignedM.value && sgp_sn == SN) {
        Dg.store(yg0 + tm * N + tn, N);
        Du.store(yu0 + tm * N + tn, N);
      } else {
        Dg.store_safe(yg0 + tm * N + tn, N, short2(sgp_sn, sgp_sm));
        Du.store_safe(yu0 + tm * N + tn, N, short2(sgp_sn, sgp_sm));
      }
      }

    });
  });
}

#define LFM_Q4_GROUPED_MLX_DUAL(NAME, BM, WM, WN)                              \
    kernel void NAME(const device uint32_t* wg        [[buffer(0)]],           \
                     const device bfloat16_t* sg      [[buffer(1)]],           \
                     const device bfloat16_t* bg      [[buffer(2)]],           \
                     const device uint32_t* wu        [[buffer(3)]],           \
                     const device bfloat16_t* su      [[buffer(4)]],           \
                     const device bfloat16_t* bu      [[buffer(5)]],           \
                     const device bfloat16_t* a       [[buffer(6)]],           \
                     device bfloat16_t* cg            [[buffer(7)]],           \
                     device bfloat16_t* cu            [[buffer(8)]],           \
                     const device uint4* blocks       [[buffer(9)]],           \
                     constant uint& K                 [[buffer(10)]],          \
                     constant uint& N                 [[buffer(11)]],          \
                     uint bid [[threadgroup_position_in_grid]],                \
                     uint lid [[thread_index_in_threadgroup]],                 \
                     uint simd_gid [[simdgroup_index_in_threadgroup]],         \
                     uint simd_lid [[thread_index_in_simdgroup]]) {            \
        threadgroup bfloat16_t Wsg[64 * (64 + 16 / sizeof(bfloat16_t))];      \
        threadgroup bfloat16_t Wsu[64 * (64 + 16 / sizeof(bfloat16_t))];      \
        lfm_grouped_mlx_dual_body<(BM), (WM), (WN)>(                          \
            wg, sg, bg, wu, su, bu, a, cg, cu, blocks, Wsg, Wsu, K, N,        \
            bid, lid, simd_gid, simd_lid);                                    \
    }

LFM_Q4_GROUPED_MLX_DUAL(lfm_q4_grouped_mlx_gu_t32, 32, 2, 2)

LFM_Q4_GROUPED_MLX(lfm_q4_grouped_mlx_t32, 32)
'''


def main():
    system_includes, body = flatten(
        "mlx/backend/metal/kernels/quantized_nax.h",
        [
            "mlx/backend/metal/kernels/utils.h",
            # elem_to_loc_broadcast used by the header's offset helpers.
            "mlx/backend/metal/kernels/steel/utils.h",
            "mlx/backend/metal/kernels/steel/gemm/nax.h",
        ],
    )
    out = (
        "// Generated by tools/gen_quant_nax.py from MLX 0.32.3 (MIT).\n"
        "// Do not edit by hand; edit the generator or the pinned headers.\n"
        + "".join(system_includes)
        + "using namespace metal;\n"
        + body
        + mlx_qmm_t_body()
        + mlx_qmm_t_shared_body()
        + HOST
    )
    with open(OUT, "w") as f:
        f.write(out)
    print("wrote", OUT, len(out), "bytes")


if __name__ == "__main__":
    main()
