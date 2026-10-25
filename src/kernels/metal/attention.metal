// Modified from Lily (Perplexity AI, Apache-2.0): see VENDORED.md for the changes to this file.
// RoPE and KV-cache append.
#include <metal_stdlib>
using namespace metal;

// In-place partial NeoX RoPE over [M, H, D]; grid is (H * rot/2, M).
kernel void rope_neox_bf16(device bfloat*  x        [[buffer(0)]],
                           constant uint&  D        [[buffer(1)]],
                           constant uint&  rot      [[buffer(2)]],
                           constant uint&  base_pos [[buffer(3)]],
                           constant float& theta    [[buffer(4)]],
                           constant uint&  row_elems [[buffer(5)]],  // H * D
                           uint2 gid [[thread_position_in_grid]]) {
    uint half_rot = rot / 2;
    uint h = gid.x / half_rot;
    uint j = gid.x % half_rot;
    float inv_freq = pow(theta, -2.0f * float(j) / float(rot));
    float ang = float(base_pos + gid.y) * inv_freq;
    float c = cos(ang);
    float s = sin(ang);
    ulong base = (ulong)gid.y * row_elems + (ulong)h * D;
    float lo = float(x[base + j]);
    float hi = float(x[base + half_rot + j]);
    x[base + j] = bfloat(lo * c - hi * s);
    x[base + half_rot + j] = bfloat(hi * c + lo * s);
}

// Appends [M, KVH, D] rows to cache [KVH, MAX, D] at base_pos.
kernel void scatter_kv_bf16(device bfloat*       cache [[buffer(0)]],
                            device const bfloat* rows  [[buffer(1)]],
                            constant uint&       D     [[buffer(2)]],
                            constant uint&       max_seq [[buffer(3)]],
                            constant uint&       base_pos [[buffer(4)]],
                            constant uint&       kvh   [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
    uint d = gid % D;
    uint h = (gid / D) % kvh;
    uint m = gid / (D * kvh);
    cache[((ulong)h * max_seq + base_pos + m) * D + d] = rows[gid];
}
