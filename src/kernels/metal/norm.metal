// Modified from Lily (Perplexity AI, Apache-2.0): see VENDORED.md for the changes to this file.
// Row-wise weighted RMSNorm with fp32 reduction.
#include <metal_stdlib>
using namespace metal;

#define TG 256

// H=2048 fused residual add + RMSNorm; rounded residuals are written to x.
constant constexpr uint RESIDUAL_H = 2048;

kernel void add_rmsnorm_bf16(
                             device bfloat*       x   [[buffer(0)]],
                             device const bfloat* b   [[buffer(1)]],
                             device const bfloat* w   [[buffer(2)]],
                             device bfloat*       out [[buffer(3)]],
                             constant float&      eps [[buffer(4)]],
                             constant float&      w_bias [[buffer(5)]],
                             uint row  [[threadgroup_position_in_grid]],
                             uint tid  [[thread_index_in_threadgroup]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float inv_rms;
    bfloat cached[8];

    float acc = 0.0f;
    for (uint j = 0; j < 8; ++j) {
        const uint i = tid + j * TG;
        const bfloat y = bfloat(float(x[row * RESIDUAL_H + i]) +
                                float(b[row * RESIDUAL_H + i]));
        cached[j] = y;
        x[row * RESIDUAL_H + i] = y;
        const float v = float(y);
        acc += v * v;
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < TG / 32; ++i) {
            total += partial[i];
        }
        inv_rms = rsqrt(total / float(RESIDUAL_H) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 8; ++j) {
        const uint i = tid + j * TG;
        const float gain = w_bias + float(w[i]);
        out[row * RESIDUAL_H + i] = bfloat(float(cached[j]) * inv_rms * gain);
    }
}

// Vectorized H=2048 fused residual add + RMSNorm. The global streams (the
// residual read/write and the addend read) and the normalized store move eight
// contiguous elements per thread as 16-byte bfloat4 accesses; the residuals are
// staged in a 4 KiB threadgroup row copy and the reduction and the normalized
// values are still computed in the scalar kernel's own stride-256 order with
// the same expression, so every stored value and the row's RMS are bitwise
// unchanged.
kernel void add_rmsnorm_bf16_v8(
                             device bfloat*       x   [[buffer(0)]],
                             device const bfloat* b   [[buffer(1)]],
                             device const bfloat* w   [[buffer(2)]],
                             device bfloat*       out [[buffer(3)]],
                             constant float&      eps [[buffer(4)]],
                             constant float&      w_bias [[buffer(5)]],
                             uint row  [[threadgroup_position_in_grid]],
                             uint tid  [[thread_index_in_threadgroup]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float inv_rms;
    threadgroup bfloat srow[RESIDUAL_H];
    threadgroup bfloat orow[RESIDUAL_H];

    const ulong base = (ulong)row * RESIDUAL_H;
    device bfloat4* xv = (device bfloat4*)(x + base);
    device const bfloat4* bv = (device const bfloat4*)(b + base);
    device bfloat4* ov = (device bfloat4*)(out + base);
    threadgroup bfloat4* sv = (threadgroup bfloat4*)srow;
    threadgroup bfloat4* rv = (threadgroup bfloat4*)orow;
    const uint i = 2 * tid;

    const bfloat4 x0 = xv[i];
    const bfloat4 x1 = xv[i + 1];
    const bfloat4 b0 = bv[i];
    const bfloat4 b1 = bv[i + 1];
    bfloat4 y0;
    bfloat4 y1;
    y0.x = bfloat(float(x0.x) + float(b0.x));
    y0.y = bfloat(float(x0.y) + float(b0.y));
    y0.z = bfloat(float(x0.z) + float(b0.z));
    y0.w = bfloat(float(x0.w) + float(b0.w));
    y1.x = bfloat(float(x1.x) + float(b1.x));
    y1.y = bfloat(float(x1.y) + float(b1.y));
    y1.z = bfloat(float(x1.z) + float(b1.z));
    y1.w = bfloat(float(x1.w) + float(b1.w));
    xv[i] = y0;
    xv[i + 1] = y1;
    sv[i] = y0;
    sv[i + 1] = y1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint j = 0; j < 8; ++j) {
        const uint e = tid + j * TG;
        const float v = float(srow[e]);
        acc += v * v;
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint j = 0; j < TG / 32; ++j) {
            total += partial[j];
        }
        inv_rms = rsqrt(total / float(RESIDUAL_H) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < 8; ++j) {
        const uint e = tid + j * TG;
        const float gain = w_bias + float(w[e]);
        orow[e] = bfloat(float(srow[e]) * inv_rms * gain);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ov[i] = rv[i];
    ov[i + 1] = rv[i + 1];
}

// Vectorized H=2048 RMSNorm (gain w_bias + w), same staging and order.
kernel void rmsnorm_bf16_v8(device const bfloat* x   [[buffer(0)]],
                            device const bfloat* w   [[buffer(1)]],
                            device bfloat*       out [[buffer(2)]],
                            constant uint&       H   [[buffer(3)]],
                            constant float&      eps [[buffer(4)]],
                            constant float&      w_bias [[buffer(5)]],
                            uint row  [[threadgroup_position_in_grid]],
                            uint tid  [[thread_index_in_threadgroup]],
                            uint sg   [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float inv_rms;
    threadgroup bfloat srow[RESIDUAL_H];
    threadgroup bfloat orow[RESIDUAL_H];

    const ulong base = (ulong)row * H;
    device const bfloat4* xv = (device const bfloat4*)(x + base);
    device bfloat4* ov = (device bfloat4*)(out + base);
    threadgroup bfloat4* sv = (threadgroup bfloat4*)srow;
    threadgroup bfloat4* rv = (threadgroup bfloat4*)orow;
    const uint i = 2 * tid;

    sv[i] = xv[i];
    sv[i + 1] = xv[i + 1];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint j = 0; j < 8; ++j) {
        const uint e = tid + j * TG;
        const float v = float(srow[e]);
        acc += v * v;
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint j = 0; j < TG / 32; ++j) {
            total += partial[j];
        }
        inv_rms = rsqrt(total / float(H) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < 8; ++j) {
        const uint e = tid + j * TG;
        const float gain = w_bias + float(w[e]);
        orow[e] = bfloat(float(srow[e]) * inv_rms * gain);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ov[i] = rv[i];
    ov[i + 1] = rv[i + 1];
}

// RMSNorm with gain w_bias + w and FP32 accumulation.
kernel void rmsnorm_bf16(device const bfloat* x   [[buffer(0)]],
                         device const bfloat* w   [[buffer(1)]],
                         device bfloat*       out [[buffer(2)]],
                         constant uint&       H   [[buffer(3)]],
                         constant float&      eps [[buffer(4)]],
                         constant float&      w_bias [[buffer(5)]],
                         uint row  [[threadgroup_position_in_grid]],
                         uint tid  [[thread_index_in_threadgroup]],
                         uint sg   [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[TG / 32];
    threadgroup float inv_rms;

    float acc = 0.0f;
    for (uint i = tid; i < H; i += TG) {
        float v = float(x[row * H + i]);
        acc += v * v;
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < TG / 32; ++i) {
            total += partial[i];
        }
        inv_rms = rsqrt(total / float(H) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < H; i += TG) {
        float gain = w_bias + float(w[i]);
        out[row * H + i] = bfloat(float(x[row * H + i]) * inv_rms * gain);
    }
}
