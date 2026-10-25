// Modified from Lily (Perplexity AI, Apache-2.0): see VENDORED.md for the changes to this file.
// SwiGLU and embedding-row gather kernels.
#include <metal_stdlib>
using namespace metal;

// SwiGLU: out = silu(gate) * up.
kernel void silu_mul_bf16(device const bfloat* gate [[buffer(0)]],
                          device const bfloat* up   [[buffer(1)]],
                          device bfloat*       out  [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    float g = float(gate[gid]);
    // MLX rounds the SiLU result to the activation dtype before the product.
    float s = float(bfloat(g / (1.0f + exp(-g))));
    out[gid] = bfloat(s * float(up[gid]));
}

// Batched embedding lookup: out[i, :] = table[ids[i], :].
kernel void gather_rows_bf16(device const bfloat* table [[buffer(0)]],
                             device const uint*   ids   [[buffer(1)]],
                             device bfloat*       out   [[buffer(2)]],
                             constant uint&       H     [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    uint i = gid / H;
    uint d = gid % H;
    out[gid] = table[(ulong)ids[i] * H + d];
}

// Vector gather: one 16-byte (eight bf16) chunk per thread, 2D grid.
kernel void gather_rows_bf16_v8(device const uint4* table [[buffer(0)]],
                                device const uint*  ids   [[buffer(1)]],
                                device uint4*       out   [[buffer(2)]],
                                constant uint&      H8    [[buffer(3)]],
                                uint2 gid [[thread_position_in_grid]]) {
    const uint r = gid.y;
    out[(ulong)r * H8 + gid.x] = table[(ulong)ids[r] * H8 + gid.x];
}

// 32-byte copy per thread: the same row copy as the v8 kernel as two 16-byte
// moves, which halves the number of threadgroups and stores for a chunk's
// gathered expert rows.
kernel void gather_rows_bf16_v16(device const uint4* table [[buffer(0)]],
                                 device const uint*  ids   [[buffer(1)]],
                                 device uint4*       out   [[buffer(2)]],
                                 constant uint&      H16   [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    const uint r = gid.y;
    const ulong d = (ulong)gid.x * 2;
    const ulong src = (ulong)ids[r] * H16 * 2 + d;
    const ulong dst = (ulong)r * H16 * 2 + d;
    out[dst] = table[src];
    out[dst + 1] = table[src + 1];
}
