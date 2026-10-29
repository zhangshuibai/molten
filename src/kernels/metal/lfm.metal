// LFM2.5-24B-A2B kernels: affine Q4/Q6 GEMV, 6-bit gathers, routing,
// short convolution, MoE combine, argmax and single-token attention.
// Portions adapted from MLX 0.32.3 affine and sdpa_vector kernels and from
// Lily's Q4 GEMV and argmax kernels; see NOTICE and VENDORED.md.
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- RMSNorm 64

// One simdgroup per row of 64 values (four rows per 256-thread threadgroup).
kernel void rmsnorm64_bf16(device const bfloat* x   [[buffer(0)]],
                          device const bfloat* w   [[buffer(1)]],
                          device bfloat*       out [[buffer(2)]],
                          constant uint&       rows [[buffer(3)]],
                          constant float&      eps [[buffer(4)]],
                          uint group [[threadgroup_position_in_grid]],
                          uint sg    [[simdgroup_index_in_threadgroup]],
                          uint lane  [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + sg;
    if (row >= rows) {
        return;
    }
    const ulong base = (ulong)row * 64;
    const float v0 = float(x[base + lane]);
    const float v1 = float(x[base + lane + 32]);
    float acc = v0 * v0 + v1 * v1;
    acc = simd_sum(acc);
    const float inv = rsqrt(acc / 64.0f + eps);
    out[base + lane] = bfloat(v0 * inv * float(w[lane]));
    out[base + lane + 32] = bfloat(v1 * inv * float(w[lane + 32]));
}

// Decode-only per-head RMSNorm of the query and key rows (rmsnorm64_bf16's
// arithmetic) fused with NeoX RoPE and the key-cache scatter. One simdgroup
// owns one head row; lane j holds x[j] and x[j+32], which is exactly the NeoX
// pair RoPE rotates, so the normalized values and the RoPE arithmetic
// (including each bfloat rounding) are those of a separate RMSNorm and RoPE.
kernel void lfm_norm_rope_qk_scatter_bf16(device const bfloat* xq      [[buffer(0)]],
                                         device const bfloat* wq      [[buffer(1)]],
                                         device bfloat*       oq      [[buffer(2)]],
                                         device const bfloat* xk      [[buffer(3)]],
                                         device const bfloat* wk      [[buffer(4)]],
                                         device bfloat*       k_cache [[buffer(5)]],
                                         constant uint&       rows_q  [[buffer(6)]],
                                         constant uint&       rows_k  [[buffer(7)]],
                                         constant uint&       rot     [[buffer(8)]],
                                         constant uint&       base_pos[[buffer(9)]],
                                         constant float&      theta   [[buffer(10)]],
                                         constant float&      eps     [[buffer(11)]],
                                         constant uint&       max_seq [[buffer(12)]],
                                         uint group [[threadgroup_position_in_grid]],
                                         uint sg    [[simdgroup_index_in_threadgroup]],
                                         uint lane  [[thread_index_in_simdgroup]]) {
    constexpr uint D = 64;
    const uint row = group * 8 + sg;
    const uint rows = rows_q + rows_k;
    if (row >= rows) {
        return;
    }
    device const bfloat* x;
    device const bfloat* w;
    uint h;
    bool is_q;
    if (row < rows_q) {
        x = xq; w = wq; h = row; is_q = true;
    } else {
        x = xk; w = wk; h = row - rows_q; is_q = false;
    }
    const ulong base = (ulong)h * D;
    const float v0 = float(x[base + lane]);
    const float v1 = float(x[base + lane + 32]);
    float acc = v0 * v0 + v1 * v1;
    acc = simd_sum(acc);
    const float inv = rsqrt(acc / 64.0f + eps);
    // Keep the normalized bfloat rounding before the rotation, exactly as the
    // separate kernels store and reload it.
    const float n0 = float(bfloat(v0 * inv * float(w[lane])));
    const float n1 = float(bfloat(v1 * inv * float(w[lane + 32])));
    float inv_freq = pow(theta, -2.0f * float(lane) / float(rot));
    float ang = float(base_pos) * inv_freq;
    float c = cos(ang);
    float s = sin(ang);
    // The separate rope kernel's `lo*c - hi*s` / `hi*c + lo*s` compile to a
    // multiply plus a fused multiply-add on this toolchain; writing the fma
    // explicitly keeps this fused kernel's bfloat results bitwise identical.
    const float t1 = n1 * s;
    const float t3 = n0 * s;
    const bfloat r0 = bfloat(fma(n0, c, -t1));
    const bfloat r1 = bfloat(fma(n1, c, t3));
    if (is_q) {
        oq[base + lane] = r0;
        oq[base + lane + 32] = r1;
    } else {
        const ulong dst = ((ulong)h * max_seq + base_pos) * D;
        k_cache[dst + lane] = r0;
        k_cache[dst + lane + 32] = r1;
    }
}

// ------------------------------------------------------------- Q4/Q6 GEMV

static inline float dot_word_q4(uint word, float s, float b,
                                device const bfloat4* xv, uint w) {
    float4 xlo = float4(xv[2 * w]);
    float4 xhi = float4(xv[2 * w + 1]);
    float4 qlo = float4(float((word >> 0) & 0xF), float((word >> 4) & 0xF),
                        float((word >> 8) & 0xF), float((word >> 12) & 0xF));
    float4 qhi = float4(float((word >> 16) & 0xF), float((word >> 20) & 0xF),
                        float((word >> 24) & 0xF), float((word >> 28) & 0xF));
    // The reference's L=1 route is MLX's qmv/qdot: accumulate the integer
    // codes against the BF16 activation in F32 and apply
    // `scale * sum(q*x) + bias * sum(x)` once per group. Prefill keeps the
    // dequantize-to-BF16 + BF16 tensor-op GEMM path instead, which is the
    // reference's qmm_t rounding.
    float qx = dot(qlo, xlo) + dot(qhi, xhi);
    float xs = dot(xlo, float4(1.0f)) + dot(xhi, float4(1.0f));
    return s * qx + b * xs;
}

#define GEMV_Q4_BODY(OUT_STORE)                                                \
    const uint words = K / 8;                                                  \
    const uint blocks = words / 4;                                             \
    const uint bpg = GS / 32;                                                  \
    const uint groups = K / GS;                                                \
    device const uint4* wrow = (device const uint4*)(codes + (ulong)row * words); \
    device const bfloat4* xv = (device const bfloat4*)x;                       \
    float sum = 0.0f;                                                          \
    for (uint i = lane; i < blocks; i += 32) {                                 \
        uint g = i / bpg;                                                      \
        float s = float(scales[row * groups + g]);                             \
        float b = float(biases[row * groups + g]);                             \
        uint4 w4 = wrow[i];                                                    \
        sum += dot_word_q4(w4.x, s, b, xv, 4 * i)                              \
            + dot_word_q4(w4.y, s, b, xv, 4 * i + 1)                           \
            + dot_word_q4(w4.z, s, b, xv, 4 * i + 2)                           \
            + dot_word_q4(w4.w, s, b, xv, 4 * i + 3);                          \
    }                                                                          \
    sum = simd_sum(sum);                                                       \
    if (lane == 0) { OUT_STORE; }

kernel void lfm_gemv_q4_bf16(device const uint*   codes  [[buffer(0)]],
                             device const bfloat* scales [[buffer(1)]],
                             device const bfloat* biases [[buffer(2)]],
                             device const bfloat* x      [[buffer(3)]],
                             device bfloat*       y      [[buffer(4)]],
                             constant uint&       K      [[buffer(5)]],
                             constant uint&       GS     [[buffer(6)]],
                             uint row  [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_threadgroup]]) {
    GEMV_Q4_BODY(y[row] = bfloat(sum))
}

// The model's decode GEMV shapes, fixed at compile time (the Rust wrappers
// check them). With K and the group size known, the block count, the
// per-block group index and the row strides are constants instead of runtime
// integer divisions ahead of the first weight load. Each lane still walks the
// same blocks in the same order with the same expressions and simd_sum, so
// every output is bitwise the runtime kernel's. Longer fixed trip counts are
// not used: fixing the 6-bit head's 16-packet loop or the four-way MoE combine
// lets fast-math re-associate the unrolled sums (measured bit differences).
constant constexpr uint LFM_GS = 64;
constant constexpr uint LFM_HIDDEN = 2048;
constant constexpr uint LFM_MOE_INTER = 1536;

template <uint K>
kernel void lfm_gemv_q4_bf16_k(device const uint*   codes  [[buffer(0)]],
                               device const bfloat* scales [[buffer(1)]],
                               device const bfloat* biases [[buffer(2)]],
                               device const bfloat* x      [[buffer(3)]],
                               device bfloat*       y      [[buffer(4)]],
                               uint row  [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]]) {
    constexpr uint GS = LFM_GS;
    GEMV_Q4_BODY(y[row] = bfloat(sum))
}

#define LFM_GEMV_Q4_K(K)                                                       \
template [[host_name("lfm_gemv_q4_bf16_k" #K)]] kernel void                    \
lfm_gemv_q4_bf16_k<K>(device const uint*, device const bfloat*,                \
                      device const bfloat*, device const bfloat*,              \
                      device bfloat*, uint, uint);
LFM_GEMV_Q4_K(2048)   // hidden-size inputs: mixer out_proj, dense gate/up
LFM_GEMV_Q4_K(11776)  // dense FFN down

// Per-value affine dequantization for 6-bit codes (MLX 4-values-in-3-bytes).
static inline void decode_q6(device const uchar* p, float s, float b,
                             device const bfloat* x, uint base,
                             thread float& acc) {
    const uint b0 = p[0], b1 = p[1], b2 = p[2];
    const uint q0 = b0 & 0x3f;
    const uint q1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0f) << 2);
    const uint q2 = ((b1 >> 4) & 0x0f) | ((b2 & 0x03) << 4);
    const uint q3 = (b2 >> 2) & 0x3f;
    const float qx = float(q0) * float(x[base]) + float(q1) * float(x[base + 1])
        + float(q2) * float(x[base + 2]) + float(q3) * float(x[base + 3]);
    const float xs = float(x[base]) + float(x[base + 1])
        + float(x[base + 2]) + float(x[base + 3]);
    acc += s * qx + b * xs;
}

kernel void lfm_gemv_q6_bf16(device const uint*   codes  [[buffer(0)]],
                             device const bfloat* scales [[buffer(1)]],
                             device const bfloat* biases [[buffer(2)]],
                             device const bfloat* x      [[buffer(3)]],
                             device bfloat*       y      [[buffer(4)]],
                             constant uint&       K      [[buffer(5)]],
                             constant uint&       GS     [[buffer(6)]],
                             uint row  [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_threadgroup]]) {
    const uint groups = K / GS;
    const uint bytes_per_group = GS * 6 / 8;
    device const uchar* wrow = (device const uchar*)codes + (ulong)row * (K * 6 / 8);
    float sum = 0.0f;
    for (uint g = lane; g < groups; g += 32) {
        const float s = float(scales[row * groups + g]);
        const float b = float(biases[row * groups + g]);
        device const uchar* gp = wrow + g * bytes_per_group;
        for (uint j = 0; j < GS / 4; ++j) {
            decode_q6(gp + 3 * j, s, b, x, g * GS + 4 * j, sum);
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = bfloat(sum);
    }
}

static inline void decode_q6_reg(device const uchar* p, float s, float b,
                                 float4 xv, thread float& acc) {
    const uint b0 = p[0], b1 = p[1], b2 = p[2];
    const uint q0 = b0 & 0x3f;
    const uint q1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0f) << 2);
    const uint q2 = ((b1 >> 4) & 0x0f) | ((b2 & 0x03) << 4);
    const uint q3 = (b2 >> 2) & 0x3f;
    const float qx = float(q0) * xv.x + float(q1) * xv.y
        + float(q2) * xv.z + float(q3) * xv.w;
    const float xs = xv.x + xv.y + xv.z + xv.w;
    acc += s * qx + b * xs;
}

// Six-bit output-head GEMV: sixteen rows per 128-thread threadgroup, four per
// simdgroup. Each group of four rows shares one 4-value x packet. Every row
// keeps the original group striding and simd_sum reduction, so its value is
// unchanged.
kernel void lfm_gemv_q6_bf16_x4(device const uint*   codes  [[buffer(0)]],
                                device const bfloat* scales [[buffer(1)]],
                                device const bfloat* biases [[buffer(2)]],
                                device const bfloat* x      [[buffer(3)]],
                                device bfloat*       y      [[buffer(4)]],
                                constant uint&       K      [[buffer(5)]],
                                constant uint&       GS     [[buffer(6)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint sgi   [[simdgroup_index_in_threadgroup]],
                                uint lane  [[thread_index_in_simdgroup]]) {
    const uint row0 = group * 16u + sgi * 4u;
    const uint groups = K / GS;
    const uint bytes_per_group = GS * 6 / 8;
    const uint bytes_per_row = K * 6 / 8;
    device const bfloat4* xv4 = (device const bfloat4*)x;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint g = lane; g < groups; g += 32) {
        device const uchar* w0 = (device const uchar*)codes
            + (ulong)(row0 + 0) * bytes_per_row + g * bytes_per_group;
        device const uchar* w1 = (device const uchar*)codes
            + (ulong)(row0 + 1) * bytes_per_row + g * bytes_per_group;
        device const uchar* w2 = (device const uchar*)codes
            + (ulong)(row0 + 2) * bytes_per_row + g * bytes_per_group;
        device const uchar* w3 = (device const uchar*)codes
            + (ulong)(row0 + 3) * bytes_per_row + g * bytes_per_group;
        const float s0 = float(scales[(row0 + 0) * groups + g]);
        const float b0 = float(biases[(row0 + 0) * groups + g]);
        const float s1 = float(scales[(row0 + 1) * groups + g]);
        const float b1 = float(biases[(row0 + 1) * groups + g]);
        const float s2 = float(scales[(row0 + 2) * groups + g]);
        const float b2 = float(biases[(row0 + 2) * groups + g]);
        const float s3 = float(scales[(row0 + 3) * groups + g]);
        const float b3 = float(biases[(row0 + 3) * groups + g]);
        const uint packet_base = g * (GS / 4);
        for (uint j = 0; j < GS / 4; ++j) {
            const float4 xp = float4(xv4[packet_base + j]);
            decode_q6_reg(w0 + 3 * j, s0, b0, xp, a0);
            decode_q6_reg(w1 + 3 * j, s1, b1, xp, a1);
            decode_q6_reg(w2 + 3 * j, s2, b2, xp, a2);
            decode_q6_reg(w3 + 3 * j, s3, b3, xp, a3);
        }
    }
    a0 = simd_sum(a0);
    a1 = simd_sum(a1);
    a2 = simd_sum(a2);
    a3 = simd_sum(a3);
    if (lane == 0) {
        y[row0 + 0] = bfloat(a0);
        y[row0 + 1] = bfloat(a1);
        y[row0 + 2] = bfloat(a2);
        y[row0 + 3] = bfloat(a3);
    }
}


// Exact H=2048 residual add + RMSNorm, staged in threadgroup memory: the same
// bfloat4 loads/stores, stride-256 reduction order, simd_sum, eight-partial
// combine and bfloat rounding as norm.metal's add_rmsnorm_bf16_v8. The updated
// residual and the normalized row are left in threadgroup memory; the caller
// writes the buffers it owns, which lets a multi-threadgroup kernel refresh
// them without a cross-threadgroup write race.
static inline void lfm_norm_row_v8_tg(device const bfloat* x,
                                      device const bfloat* b,
                                      device const bfloat* w,
                                      threadgroup bfloat* srow,
                                      threadgroup bfloat* orow,
                                      threadgroup float* partial,
                                      threadgroup float* inv_rms,
                                      float eps, float w_bias,
                                      uint tid, uint sg, uint lane) {
    constexpr uint H = 2048;
    constexpr uint TG = 256;
    threadgroup bfloat4* sv = (threadgroup bfloat4*)srow;
    threadgroup bfloat4* rv = (threadgroup bfloat4*)orow;
    const device bfloat4* xv = (const device bfloat4*)x;
    const device bfloat4* bv = (const device bfloat4*)b;
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
        inv_rms[0] = rsqrt(total / float(H) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < 8; ++j) {
        const uint e = tid + j * TG;
        const float gain = w_bias + float(w[e]);
        orow[e] = bfloat(float(srow[e]) * inv_rms[0] * gain);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The per-lane k-series of the router GEMV, staged eight elements at a time.
// The additions still happen in increasing k order, so every lane's partial
// sum is bitwise the same as the scalar loop above; only the memory issue
// order changes, which removes the dependent-load stall.
static inline float lfm_router_lane_sum(device const float* wrow,
                                        device const bfloat* x,
                                        uint K, uint lane) {
    float sum = 0.0f;
    uint k = lane;
    for (; k + 480u < K; k += 512u) {
        const float x0 = float(x[k]), x1 = float(x[k + 32]);
        const float x2 = float(x[k + 64]), x3 = float(x[k + 96]);
        const float x4 = float(x[k + 128]), x5 = float(x[k + 160]);
        const float x6 = float(x[k + 192]), x7 = float(x[k + 224]);
        const float x8 = float(x[k + 256]), x9 = float(x[k + 288]);
        const float xa = float(x[k + 320]), xb = float(x[k + 352]);
        const float xc = float(x[k + 384]), xd = float(x[k + 416]);
        const float xe = float(x[k + 448]), xf = float(x[k + 480]);
        const float w0 = wrow[k], w1 = wrow[k + 32];
        const float w2 = wrow[k + 64], w3 = wrow[k + 96];
        const float w4 = wrow[k + 128], w5 = wrow[k + 160];
        const float w6 = wrow[k + 192], w7 = wrow[k + 224];
        const float w8 = wrow[k + 256], w9 = wrow[k + 288];
        const float wa = wrow[k + 320], wb = wrow[k + 352];
        const float wc = wrow[k + 384], wd = wrow[k + 416];
        const float we = wrow[k + 448], wf = wrow[k + 480];
        sum += w0 * x0;
        sum += w1 * x1;
        sum += w2 * x2;
        sum += w3 * x3;
        sum += w4 * x4;
        sum += w5 * x5;
        sum += w6 * x6;
        sum += w7 * x7;
        sum += w8 * x8;
        sum += w9 * x9;
        sum += wa * xa;
        sum += wb * xb;
        sum += wc * xc;
        sum += wd * xd;
        sum += we * xe;
        sum += wf * xf;
    }
    for (; k < K; k += 32) {
        sum += wrow[k] * float(x[k]);
    }
    return sum;
}

kernel void lfm_gemv_f32_bf16x_f32out_u8(device const float*  w   [[buffer(0)]],
                                         device const bfloat* x   [[buffer(1)]],
                                         device float*        y   [[buffer(2)]],
                                         constant uint&       K   [[buffer(3)]],
                                         uint row  [[threadgroup_position_in_grid]],
                                         uint lane [[thread_index_in_threadgroup]]) {
    float sum = lfm_router_lane_sum(w + (ulong)row * K, x, K, lane);
    sum = simd_sum(sum);
    if (lane == 0) {
        y[row] = sum;
    }
}

// ------------------------------------------------------------------- Q6 gather

kernel void lfm_gather_rows_q6_bf16(device const uint*   codes  [[buffer(0)]],
                                    device const bfloat* scales [[buffer(1)]],
                                    device const bfloat* biases [[buffer(2)]],
                                    device const uint*   ids    [[buffer(3)]],
                                    device bfloat*       out    [[buffer(4)]],
                                    constant uint&       K      [[buffer(5)]],
                                    constant uint&       GS     [[buffer(6)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    const uint pack = gid.x;
    const uint token = gid.y;
    const uint per_group = GS / 4;
    const uint g = pack / per_group;
    const uint j = pack % per_group;
    const uint row = ids[token];
    const uint groups = K / GS;
    const uint bytes_per_group = GS * 6 / 8;
    device const uchar* wrow = (device const uchar*)codes + (ulong)row * (K * 6 / 8);
    device const uchar* p = wrow + g * bytes_per_group + 3 * j;
    const uint b0 = p[0], b1 = p[1], b2 = p[2];
    const uint q0 = b0 & 0x3f;
    const uint q1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0f) << 2);
    const uint q2 = ((b1 >> 4) & 0x0f) | ((b2 & 0x03) << 4);
    const uint q3 = (b2 >> 2) & 0x3f;
    const float s = float(scales[row * groups + g]);
    const float b = float(biases[row * groups + g]);
    device bfloat* dst = out + (ulong)token * K + pack * 4;
    dst[0] = bfloat(s * float(q0) + b);
    dst[1] = bfloat(s * float(q1) + b);
    dst[2] = bfloat(s * float(q2) + b);
    dst[3] = bfloat(s * float(q3) + b);
}

// ------------------------------------------------------------ router (prefill)

// Eight prompt tokens per threadgroup with four experts per pass: each
// router weight row is read once for eight tokens. Every (token, expert)
// output keeps the 256-thread k striding, the simdgroup tree and the
// eight-way serial combine, so the logits are bitwise unchanged.
kernel void lfm_router_prefill_f32_batch8(device const bfloat* x   [[buffer(0)]],
                                          device const float*  w   [[buffer(1)]],
                                          device float*        out [[buffer(2)]],
                                          constant uint&       M   [[buffer(3)]],
                                          constant uint&       K   [[buffer(4)]],
                                          constant uint&       E   [[buffer(5)]],
                                          uint g   [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint sg  [[simdgroup_index_in_threadgroup]],
                                          uint lane[[thread_index_in_simdgroup]]) {
    constexpr uint TT = 8;
    constexpr uint EB = 4;
    threadgroup float partial[TT][EB][8];
    const uint t0 = g * TT;
    if (t0 >= M) {
        return;
    }
    const uint nt = min(TT, M - t0);
    device const bfloat* xrow[TT];
    for (uint t = 0; t < TT; ++t) {
        xrow[t] = x + (ulong)(t0 + (t < nt ? t : 0)) * K;
    }
    for (uint base = 0; base < E; base += EB) {
        float acc[TT][EB];
        for (uint t = 0; t < TT; ++t) {
            for (uint e = 0; e < EB; ++e) {
                acc[t][e] = 0.0f;
            }
        }
        for (uint k = tid; k < K; k += 256) {
            float xv[TT];
            for (uint t = 0; t < TT; ++t) {
                xv[t] = float(xrow[t][k]);
            }
            for (uint e = 0; e < EB; ++e) {
                const float wv = w[(ulong)(base + e) * K + k];
                for (uint t = 0; t < TT; ++t) {
                    acc[t][e] += wv * xv[t];
                }
            }
        }
        for (uint t = 0; t < TT; ++t) {
            for (uint e = 0; e < EB; ++e) {
                acc[t][e] = simd_sum(acc[t][e]);
            }
        }
        if (lane == 0) {
            for (uint t = 0; t < TT; ++t) {
                for (uint e = 0; e < EB; ++e) {
                    partial[t][e][sg] = acc[t][e];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            for (uint t = 0; t < nt; ++t) {
                for (uint e = 0; e < EB; ++e) {
                    float total = 0.0f;
                    for (uint i = 0; i < 8; ++i) {
                        total += partial[t][e][i];
                    }
                    out[(ulong)(t0 + t) * E + base + e] = total;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// --------------------------------------------------------- router (single token)

kernel void lfm_route_f32(device const float*  logits [[buffer(0)]],
                          device const float*  bias   [[buffer(1)]],
                          device uint*         ids    [[buffer(2)]],
                          device bfloat*       weights[[buffer(3)]],
                          constant uint&       E      [[buffer(4)]],
                          constant uint&       TOPK   [[buffer(5)]],
                          constant float&      eps    [[buffer(6)]],
                          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float s[128];
    // Parallel top-k. Each pass rebuilds a reduction tree over the untouched
    // candidate list from threadgroup memory, so the result is exactly that of
    // a serial scan over all 64 experts per selected slot: greatest
    // (sigmoid + bias) first, an exact tie to the lower expert id.
    threadgroup float cand[128];
    threadgroup uint cand_id[128];
    threadgroup uchar used[128];
    threadgroup float red_v[128];
    threadgroup uint red_e[128];
    if (tid < E) {
        // sigmoid in F32
        const float z = logits[tid];
        s[tid] = 1.0f / (1.0f + exp(-z));
        cand[tid] = s[tid] + bias[tid];
        cand_id[tid] = tid;
        used[tid] = 0;
    } else {
        // Sentinels never beat a real candidate.
        cand[tid] = -INFINITY;
        cand_id[tid] = 0xFFFFFFFFu;
        used[tid] = 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint chosen[8];
    for (uint r = 0; r < TOPK; ++r) {
        red_v[tid] = used[tid] ? -INFINITY : cand[tid];
        red_e[tid] = cand_id[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 64; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float a = red_v[tid];
                const float b = red_v[tid + stride];
                const uint ia = red_e[tid];
                const uint ib = red_e[tid + stride];
                if (b > a || (b == a && ib < ia)) {
                    red_v[tid] = b;
                    red_e[tid] = ib;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            const uint pick = red_e[0];
            chosen[r] = pick;
            used[pick] = 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float denom = eps;
        for (uint r = 0; r < TOPK; ++r) {
            denom += s[chosen[r]];
        }
        for (uint r = 0; r < TOPK; ++r) {
            ids[r] = chosen[r];
            weights[r] = bfloat(s[chosen[r]] / denom);
        }
    }
}

// Single-warp top-k. Each lane owns experts lane and lane+32, so the four
// selection passes are warp shuffles with no threadgroup barrier. The order is
// lfm_route_f32's: greater (sigmoid + bias), an exact tie to the lower id, with
// the same sigmoid, denominator accumulation and bfloat weights.
// Shared by lfm_route_warp_f32 and lfm_router_addnorm_route_f32, so both select
// with the same total order and the same normalized unbiased weights. Only
// lane 0 holds valid results; callers broadcast the entry they need.
// The lane's two selection-bias values are passed in (the caller loads
// bias[lane] and bias[lane + 32]).
static inline void lfm_topk_warp_b(device const float* logits,
                                   float bias_lo,
                                   float bias_hi,
                                   uint lane,
                                   thread uint* chosen,
                                   thread float* wt,
                                   float eps) {
    constexpr uint TOPK = 4;
    const float z0 = logits[lane];
    const float z1 = logits[lane + 32];
    const float s0 = 1.0f / (1.0f + exp(-z0));
    const float s1 = 1.0f / (1.0f + exp(-z1));
    float c0 = s0 + bias_lo;
    float c1 = s1 + bias_hi;
    bool used0 = false, used1 = false;
    for (uint r = 0; r < TOPK; ++r) {
        chosen[r] = 0u;
        wt[r] = 0.0f;
    }
    float raw[8];
    for (uint r = 0; r < TOPK; ++r) {
        float bv = -INFINITY;
        uint bi = 0xFFFFFFFFu;
        if (!used0) {
            if (c0 > bv || (c0 == bv && lane < bi)) { bv = c0; bi = lane; }
        }
        if (!used1) {
            const uint e = lane + 32;
            if (c1 > bv || (c1 == bv && e < bi)) { bv = c1; bi = e; }
        }
        for (uint off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(bv, off);
            uint oi = simd_shuffle_down(bi, off);
            if (lane + off < 32) {
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
        }
        const uint pick = simd_shuffle(bi, 0u);
        if (lane == pick) { used0 = true; }
        if (lane + 32 == pick) { used1 = true; }
        const float ps0 = simd_shuffle(s0, pick & 31u);
        const float ps1 = simd_shuffle(s1, pick & 31u);
        if (lane == 0) {
            chosen[r] = pick;
            raw[r] = (pick < 32u) ? ps0 : ps1;
        }
    }
    if (lane == 0) {
        float denom = eps;
        for (uint r = 0; r < TOPK; ++r) {
            denom += raw[r];
        }
        for (uint r = 0; r < TOPK; ++r) {
            wt[r] = float(bfloat(raw[r] / denom));
        }
    }
}

static inline void lfm_topk_warp(device const float* logits,
                                 device const float* bias,
                                 uint lane,
                                 thread uint* chosen,
                                 thread float* wt,
                                 float eps) {
    lfm_topk_warp_b(logits, bias[lane], bias[lane + 32], lane, chosen, wt, eps);
}

kernel void lfm_route_warp_f32(device const float*  logits [[buffer(0)]],
                               device const float*  bias   [[buffer(1)]],
                               device uint*         ids    [[buffer(2)]],
                               device bfloat*       weights[[buffer(3)]],
                               constant uint&       E      [[buffer(4)]],
                               constant uint&       TOPK   [[buffer(5)]],
                               constant float&      eps    [[buffer(6)]],
                               uint lane [[thread_index_in_simdgroup]]) {
    uint chosen[8];
    float wt[8];
    lfm_topk_warp(logits, bias, lane, chosen, wt, eps);
    if (lane == 0) {
        for (uint r = 0; r < TOPK; ++r) {
            ids[r] = chosen[r];
            weights[r] = bfloat(wt[r]);
        }
    }
}

#if __METAL_VERSION__ >= 320
// Decode-only: the post-mixer residual add + RMSNorm, the F32 router GEMV and
// lfm_route_warp_f32's top-k in one dispatch. Each threadgroup computes the row
// norm with the exact add_rmsnorm_bf16_v8 arithmetic and its own router expert;
// group 0 also writes the updated residual and the normalized row for the
// following gather dispatches. Every threadgroup publishes its expert's F32
// logit, then
// issues a device-scope fence and a relaxed increment of `counter`. The one
// threadgroup whose increment observes E - 1 (so every other logit store is
// already ordered before it) fences again, runs the identical warp top-k on
// the same F32 logits and resets the counter for the next dispatch. No
// threadgroup ever waits on another, so forward progress never depends on
// co-residency. `atomic_thread_fence` needs MSL 3.2; this file is also
// compiled at 3.1 (the other kernels), hence the guard.
constant constexpr uint LFM_ROUTE_E = 64;
constant constexpr uint LFM_ROUTE_TOPK = 4;
constant constexpr uint LFM_ROUTE_K = 2048;

// lfm_router_lane_sum_tg with K fixed at compile time: the K/512 blocks
// unroll, so all the weight and activation loads can issue ahead of the
// adds. The adds are the same statements in the same increasing-k order (for
// K = 2048 the runtime loop runs exactly these four blocks and no tail), so
// every lane's partial sum is bitwise unchanged.
template <uint K, typename WRow>
static inline float lfm_router_lane_sum_tg_fixed(WRow wrow,
                                                 threadgroup const bfloat* x,
                                                 uint lane) {
    static_assert(K % 512u == 0u, "the router lane sum unrolls 512-element blocks");
    float sum = 0.0f;
#pragma unroll
    for (uint blk = 0; blk < K / 512u; ++blk) {
        const uint k = lane + blk * 512u;
        const float x0 = float(x[k]), x1 = float(x[k + 32]);
        const float x2 = float(x[k + 64]), x3 = float(x[k + 96]);
        const float x4 = float(x[k + 128]), x5 = float(x[k + 160]);
        const float x6 = float(x[k + 192]), x7 = float(x[k + 224]);
        const float x8 = float(x[k + 256]), x9 = float(x[k + 288]);
        const float xa = float(x[k + 320]), xb = float(x[k + 352]);
        const float xc = float(x[k + 384]), xd = float(x[k + 416]);
        const float xe = float(x[k + 448]), xf = float(x[k + 480]);
        const float w0 = wrow[k], w1 = wrow[k + 32];
        const float w2 = wrow[k + 64], w3 = wrow[k + 96];
        const float w4 = wrow[k + 128], w5 = wrow[k + 160];
        const float w6 = wrow[k + 192], w7 = wrow[k + 224];
        const float w8 = wrow[k + 256], w9 = wrow[k + 288];
        const float wa = wrow[k + 320], wb = wrow[k + 352];
        const float wc = wrow[k + 384], wd = wrow[k + 416];
        const float we = wrow[k + 448], wf = wrow[k + 480];
        sum += w0 * x0;
        sum += w1 * x1;
        sum += w2 * x2;
        sum += w3 * x3;
        sum += w4 * x4;
        sum += w5 * x5;
        sum += w6 * x6;
        sum += w7 * x7;
        sum += w8 * x8;
        sum += w9 * x9;
        sum += wa * xa;
        sum += wb * xb;
        sum += wc * xc;
        sum += wd * xd;
        sum += we * xe;
        sum += wf * xf;
    }
    return sum;
}

kernel void lfm_router_addnorm_route_f32(device const float*  w       [[buffer(0)]],
                                         device const bfloat* h       [[buffer(1)]],
                                         device const bfloat* b       [[buffer(2)]],
                                         device const bfloat* nw      [[buffer(3)]],
                                         device bfloat*       hout    [[buffer(4)]],
                                         device bfloat*       normed  [[buffer(5)]],
                                         device float*        y       [[buffer(6)]],
                                         device const float*  bias    [[buffer(7)]],
                                         device uint*         ids     [[buffer(8)]],
                                         device bfloat*       weights [[buffer(9)]],
                                         device atomic_uint*  counter [[buffer(10)]],
                                         constant float&      eps     [[buffer(11)]],
                                         constant float&      w_bias  [[buffer(12)]],
                                         constant float&      route_eps [[buffer(13)]],
                                         uint expert [[threadgroup_position_in_grid]],
                                         uint tid    [[thread_index_in_threadgroup]],
                                         uint sg     [[simdgroup_index_in_threadgroup]],
                                         uint lane   [[thread_index_in_simdgroup]]) {
    threadgroup bfloat srow[2048];
    threadgroup bfloat orow[2048];
    threadgroup float wrow[LFM_ROUTE_K];
    threadgroup float partial[8];
    threadgroup float inv_rms[1];
    // Issue this expert's 8 KB F32 router row and the top-k bias at kernel
    // entry, with all 256 threads, so their DRAM latency overlaps the
    // add_rmsnorm prologue (whose barriers also publish the staged row). Pure
    // copies: simdgroup 0 then sums the same values in the same lane order.
    {
        const device float4* wv = (const device float4*)(w + (ulong)expert * LFM_ROUTE_K);
        threadgroup float4* tv = (threadgroup float4*)wrow;
        const uint i = 2 * tid;
        tv[i] = wv[i];
        tv[i + 1] = wv[i + 1];
    }
    float bias_lo = 0.0f;
    float bias_hi = 0.0f;
    if (sg == 0) {
        bias_lo = bias[lane];
        bias_hi = bias[lane + 32];
    }
    lfm_norm_row_v8_tg(h, b, nw, srow, orow, partial, inv_rms,
                       eps, w_bias, tid, sg, lane);
    if (expert == 0) {
        threadgroup bfloat4* sv = (threadgroup bfloat4*)srow;
        threadgroup bfloat4* rv = (threadgroup bfloat4*)orow;
        device bfloat4* hv = (device bfloat4*)hout;
        device bfloat4* nv = (device bfloat4*)normed;
        const uint i = 2 * tid;
        hv[i] = sv[i];
        hv[i + 1] = sv[i + 1];
        nv[i] = rv[i];
        nv[i + 1] = rv[i + 1];
    }
    if (sg == 0) {
        float sum = lfm_router_lane_sum_tg_fixed<LFM_ROUTE_K>(
            (threadgroup const float*)wrow, orow, lane);
        sum = simd_sum(sum);
        uint last = 0u;
        if (lane == 0) {
            y[expert] = sum;
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                                thread_scope::thread_scope_device);
            const uint done = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
            last = (done == LFM_ROUTE_E - 1u) ? 1u : 0u;
        }
        last = simd_shuffle(last, 0u);
        if (last) {
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                                thread_scope::thread_scope_device);
            uint chosen[8];
            float wt[8];
            lfm_topk_warp_b(y, bias_lo, bias_hi, lane, chosen, wt, route_eps);
            if (lane == 0) {
                for (uint r = 0; r < LFM_ROUTE_TOPK; ++r) {
                    ids[r] = chosen[r];
                    weights[r] = bfloat(wt[r]);
                }
                atomic_store_explicit(counter, 0u, memory_order_relaxed);
            }
        }
    }
}
#endif

// ------------------------------------------------ MoE combine (single token)

kernel void lfm_moe_combine_rows_bf16(device const bfloat* sorted  [[buffer(0)]],
                                      device const uint*   slotpos [[buffer(1)]],
                                      device const bfloat* weights [[buffer(2)]],
                                      device bfloat*       out     [[buffer(3)]],
                                      constant uint&       H       [[buffer(4)]],
                                      constant uint&       TOPK    [[buffer(5)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    const uint h = gid.x;
    const uint t = gid.y;
    float sum = 0.0f;
    for (uint r = 0; r < TOPK; ++r) {
        const uint p = slotpos[t * TOPK + r];
        const float prod = float(weights[t * TOPK + r]) * float(sorted[(ulong)p * H + h]);
        sum += float(bfloat(prod));
    }
    out[(ulong)t * H + h] = bfloat(sum);
}

// Vector combine: eight bf16 outputs per thread, same per-product BF16
// rounding and the same rank order as the scalar kernel.
kernel void lfm_moe_combine_rows_bf16_v8(device const bfloat* sorted  [[buffer(0)]],
                                         device const uint*   slotpos [[buffer(1)]],
                                         device const bfloat* weights [[buffer(2)]],
                                         device bfloat*       out     [[buffer(3)]],
                                         constant uint&       H8      [[buffer(4)]],
                                         constant uint&       TOPK    [[buffer(5)]],
                                         uint2 gid [[thread_position_in_grid]]) {
    const uint t = gid.y;
    const ulong o = (ulong)t * H8 * 8 + (ulong)gid.x * 8;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
    for (uint r = 0; r < TOPK; ++r) {
        const uint p = slotpos[t * TOPK + r];
        const float w = float(weights[t * TOPK + r]);
        const ulong src = (ulong)p * H8 * 8 + (ulong)gid.x * 8;
        s0 += float(bfloat(w * float(sorted[src + 0])));
        s1 += float(bfloat(w * float(sorted[src + 1])));
        s2 += float(bfloat(w * float(sorted[src + 2])));
        s3 += float(bfloat(w * float(sorted[src + 3])));
        s4 += float(bfloat(w * float(sorted[src + 4])));
        s5 += float(bfloat(w * float(sorted[src + 5])));
        s6 += float(bfloat(w * float(sorted[src + 6])));
        s7 += float(bfloat(w * float(sorted[src + 7])));
    }
    out[o + 0] = bfloat(s0);
    out[o + 1] = bfloat(s1);
    out[o + 2] = bfloat(s2);
    out[o + 3] = bfloat(s3);
    out[o + 4] = bfloat(s4);
    out[o + 5] = bfloat(s5);
    out[o + 6] = bfloat(s6);
    out[o + 7] = bfloat(s7);
}

// 16-value combine: the v8 kernel's per-product BF16 rounding, rank order and
// final BF16 store, with two 16-byte stores per thread.
kernel void lfm_moe_combine_rows_bf16_v16(device const bfloat* sorted  [[buffer(0)]],
                                          device const uint*   slotpos [[buffer(1)]],
                                          device const bfloat* weights [[buffer(2)]],
                                          device bfloat*       out     [[buffer(3)]],
                                          constant uint&       H16     [[buffer(4)]],
                                          constant uint&       TOPK    [[buffer(5)]],
                                          uint2 gid [[thread_position_in_grid]]) {
    const uint t = gid.y;
    const ulong h = (ulong)gid.x * 16;
    const ulong o = (ulong)t * H16 * 16 + h;
    float s[16];
    for (uint j = 0; j < 16; ++j) {
        s[j] = 0.0f;
    }
    for (uint r = 0; r < TOPK; ++r) {
        const uint p = slotpos[t * TOPK + r];
        const float w = float(weights[t * TOPK + r]);
        const ulong src = (ulong)p * H16 * 16 + h;
        for (uint j = 0; j < 16; ++j) {
            s[j] += float(bfloat(w * float(sorted[src + j])));
        }
    }
    for (uint j = 0; j < 16; ++j) {
        out[o + j] = bfloat(s[j]);
    }
}

// --------------------------------------------------------- short convolution

// Short convolution over the packed BCx projection output: B, C and X are
// the three contiguous H-wide segments of each row, so the separate split
// copy is not needed.
kernel void lfm_conv3_gate_bcx_bf16(device const bfloat* bcx      [[buffer(0)]],
                                    device const bfloat* w        [[buffer(1)]],  // [H, 3]
                                    device const bfloat* state    [[buffer(2)]],  // [2, H]
                                    device bfloat*       out      [[buffer(3)]],
                                    device bfloat*       new_state[[buffer(4)]],
                                    constant uint&       H        [[buffer(5)]],
                                    constant uint&       M        [[buffer(6)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    const uint channel = gid.x;
    const uint t = gid.y;
    if (channel >= H || t >= M) {
        return;
    }
    const ulong stride = 3ul * H;
    const ulong row = (ulong)t * stride + channel;
    float v[3];
    for (int k = 0; k < 3; ++k) {
        const int idx = int(t) + k - 2;
        if (idx >= 0) {
            const ulong p = (ulong)idx * stride + channel;
            v[k] = float(bfloat(float(bcx[p]) * float(bcx[p + 2ul * H])));
        } else {
            v[k] = float(state[(idx == -2 ? 0 : 1) * H + channel]);
        }
    }
    float z = float(w[channel * 3 + 0]) * v[0]
            + float(w[channel * 3 + 1]) * v[1]
            + float(w[channel * 3 + 2]) * v[2];
    const float zb = float(bfloat(z));
    out[(ulong)t * H + channel] = bfloat(float(bcx[row + H]) * zb);
    if (t == M - 1) {
        new_state[H + channel] = bfloat(v[2]);
        const float prev = (M >= 2)
            ? float(bfloat(float(bcx[row - stride]) * float(bcx[row - stride + 2ul * H])))
            : float(state[H + channel]);
        new_state[channel] = bfloat(prev);
    }
}


// ------------------------------------------------- decode-path fusions
// The decode step is a long chain of tiny serialized dispatches; each launch
// costs more than several of these kernels' arithmetic. These fused kernels
// keep every per-row dot product and every rounding boundary of the unfused
// kernels (same lane striding, same simd_sum, same bfloat() rounding) and
// only merge dispatches that would otherwise be adjacent and dependent.

// Exactly the row reduction of GEMV_Q4_BODY, shared by the fused kernels.
static inline float lfm_q4_row_sum(device const uint*   codes,
                                   device const bfloat* scales,
                                   device const bfloat* biases,
                                   device const bfloat* x,
                                   uint row, uint K, uint GS, uint lane) {
    const uint words = K / 8;
    const uint blocks = words / 4;
    const uint bpg = GS / 32;
    const uint groups = K / GS;
    device const uint4* wrow = (device const uint4*)(codes + (ulong)row * words);
    device const bfloat4* xv = (device const bfloat4*)x;
    float sum = 0.0f;
    for (uint i = lane; i < blocks; i += 32) {
        uint g = i / bpg;
        float s = float(scales[row * groups + g]);
        float b = float(biases[row * groups + g]);
        uint4 w4 = wrow[i];
        sum += dot_word_q4(w4.x, s, b, xv, 4 * i)
            + dot_word_q4(w4.y, s, b, xv, 4 * i + 1)
            + dot_word_q4(w4.z, s, b, xv, 4 * i + 2)
            + dot_word_q4(w4.w, s, b, xv, 4 * i + 3);
    }
    return simd_sum(sum);
}

// Q, K and V projections in one dispatch; V is written straight into the KV
// cache at the current position (a separate scatter would be a pure copy). K
// and the group size are the compile-time decode shape.
kernel void lfm_proj3_gemv_scatter_bf16(
        device const uint*   qc  [[buffer(0)]],
        device const bfloat* qs  [[buffer(1)]],
        device const bfloat* qb  [[buffer(2)]],
        device const uint*   kc  [[buffer(3)]],
        device const bfloat* ks  [[buffer(4)]],
        device const bfloat* kb  [[buffer(5)]],
        device const uint*   vc  [[buffer(6)]],
        device const bfloat* vs  [[buffer(7)]],
        device const bfloat* vb  [[buffer(8)]],
        device const bfloat* x   [[buffer(9)]],
        device bfloat*       q   [[buffer(10)]],
        device bfloat*       k   [[buffer(11)]],
        device bfloat*       v_cache [[buffer(12)]],
        constant uint&       NQ  [[buffer(13)]],
        constant uint&       NK  [[buffer(14)]],
        constant uint&       NV  [[buffer(15)]],
        constant uint&       max_seq [[buffer(16)]],
        constant uint&       pos [[buffer(17)]],
        uint row  [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]]) {
    device const uint*   codes;
    device const bfloat* scales;
    device const bfloat* biases;
    uint local;
    if (row < NQ) {
        codes = qc; scales = qs; biases = qb; local = row;
    } else if (row < NQ + NK) {
        codes = kc; scales = ks; biases = kb; local = row - NQ;
    } else {
        codes = vc; scales = vs; biases = vb; local = row - NQ - NK;
    }
    const float sum = lfm_q4_row_sum(codes, scales, biases, x, local, LFM_HIDDEN, LFM_GS, lane);
    if (lane == 0) {
        if (row < NQ) {
            q[local] = bfloat(sum);
        } else if (row < NQ + NK) {
            k[local] = bfloat(sum);
        } else {
            const uint h = local / 64u;
            const uint d = local % 64u;
            v_cache[((ulong)h * max_seq + pos) * 64ul + d] = bfloat(sum);
        }
    }
}

// Decode-only short-conv input projection (B, C, X for one channel) plus the
// conv itself. One simdgroup owns each of the three rows with lfm_q4_row_sum's
// lane/block walk and simd_sum; the B*X product, bfloat roundings, tap order
// and state update are lfm_conv3_gate_bcx_bf16's arithmetic. K and the group
// size are the compile-time decode shape.
kernel void lfm_conv3_gemv_split_bf16(device const uint*   codes  [[buffer(0)]],
                                      device const bfloat* scales [[buffer(1)]],
                                      device const bfloat* biases [[buffer(2)]],
                                      device const bfloat* x      [[buffer(3)]],
                                      device const bfloat* w      [[buffer(4)]],
                                      device const bfloat* state  [[buffer(5)]],
                                      device bfloat*       out    [[buffer(6)]],
                                      device bfloat*       new_state [[buffer(7)]],
                                      constant uint&       H      [[buffer(8)]],
                                      uint channel [[threadgroup_position_in_grid]],
                                      uint tid  [[thread_index_in_threadgroup]],
                                      uint sg   [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    const float sum = lfm_q4_row_sum(codes, scales, biases, x, channel + sg * H,
                                     LFM_HIDDEN, LFM_GS, lane);
    threadgroup float parts[3];
    if (lane == 0) {
        parts[sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const float b = float(bfloat(parts[0]));
        const float c = float(bfloat(parts[1]));
        const float xv = float(bfloat(parts[2]));
        const float v0 = float(state[channel]);
        const float v1 = float(state[H + channel]);
        const float v2 = float(bfloat(b * xv));
        const float z = float(w[channel * 3 + 0]) * v0
                      + float(w[channel * 3 + 1]) * v1
                      + float(w[channel * 3 + 2]) * v2;
        const float zb = float(bfloat(z));
        out[channel] = bfloat(c * zb);
        new_state[H + channel] = bfloat(v2);
        new_state[channel] = bfloat(state[H + channel]);
    }
}

// ---------------------------------------------------------------- argmax BF16

#define ARGMAX_TG 256
struct LfmArgMaxPair {
    float v;
    uint  i;
};

static inline void lfm_argmax_merge(thread float& bv, thread uint& bi, float v, uint i) {
    if (v > bv || (v == bv && i < bi)) {
        bv = v;
        bi = i;
    }
}

kernel void lfm_argmax_bf16_partial(device const bfloat* x        [[buffer(0)]],
                                    device LfmArgMaxPair* partials [[buffer(1)]],
                                    constant uint&        n        [[buffer(2)]],
                                    constant uint&        chunk    [[buffer(3)]],
                                    uint g    [[threadgroup_position_in_grid]],
                                    uint tid  [[thread_index_in_threadgroup]],
                                    uint sg   [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part_v[ARGMAX_TG / 32];
    threadgroup uint  part_i[ARGMAX_TG / 32];

    float bv = -INFINITY;
    uint  bi = 0;
    uint end = min((g + 1) * chunk, n);
    for (uint i = g * chunk + tid; i < end; i += ARGMAX_TG) {
        lfm_argmax_merge(bv, bi, float(x[i]), i);
    }
    for (uint off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(bv, off);
        uint  oi = simd_shuffle_down(bi, off);
        lfm_argmax_merge(bv, bi, ov, oi);
    }
    if (lane == 0) {
        part_v[sg] = bv;
        part_i[sg] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint s = 1; s < ARGMAX_TG / 32; ++s) {
            lfm_argmax_merge(bv, bi, part_v[s], part_i[s]);
        }
        partials[g].v = bv;
        partials[g].i = bi;
    }
}

kernel void lfm_argmax_final(device const LfmArgMaxPair* partials [[buffer(0)]],
                             device uint*             out      [[buffer(1)]],
                             constant uint&           groups   [[buffer(2)]],
                             uint tid  [[thread_index_in_threadgroup]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float part_v[ARGMAX_TG / 32];
    threadgroup uint  part_i[ARGMAX_TG / 32];

    float bv = -INFINITY;
    uint  bi = 0;
    for (uint g = tid; g < groups; g += ARGMAX_TG) {
        lfm_argmax_merge(bv, bi, partials[g].v, partials[g].i);
    }
    for (uint off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(bv, off);
        uint  oi = simd_shuffle_down(bi, off);
        lfm_argmax_merge(bv, bi, ov, oi);
    }
    if (lane == 0) {
        part_v[sg] = bv;
        part_i[sg] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint s = 1; s < ARGMAX_TG / 32; ++s) {
            lfm_argmax_merge(bv, bi, part_v[s], part_i[s]);
        }
        out[0] = bi;
    }
}

// ------------------------------------------------- decode token tail

// Orderable 64-bit key of an argmax candidate: a larger key is a larger value,
// or an equal value at a lower index, so atomic_max over keys selects what
// lfm_argmax_merge's "value desc, index asc" rule selects (-0 ties +0, as the
// merge's float == does). Candidates are never NaN (the merge never takes one).
static inline ulong lfm_argmax_key(float v, uint i) {
    uint u = as_type<uint>(v);
    if ((u << 1) == 0u) {
        u = 0u;
    }
    const uint o = (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
    return (ulong(o) << 32) | ulong(~i);
}

// Decode-step output head with an argmax epilogue: lfm_gemv_q6_bf16_x4's rows
// and arithmetic, but instead of storing the BF16 logits each threadgroup
// merges its sixteen BF16-rounded logits with lfm_argmax_merge and folds the
// winner into one device-wide atomic_max key. "Value desc, index asc" is a
// total order on non-NaN candidates, so the winner is the two-stage argmax's
// for any partition (including its index-0 result when no logit beats -inf).
kernel void lfm_gemv_q6_bf16_x4_argmax(device const uint*   codes  [[buffer(0)]],
                                       device const bfloat* scales [[buffer(1)]],
                                       device const bfloat* biases [[buffer(2)]],
                                       device const bfloat* x      [[buffer(3)]],
                                       device atomic_ulong* best   [[buffer(4)]],
                                       constant uint&       K      [[buffer(5)]],
                                       constant uint&       GS     [[buffer(6)]],
                                       uint group [[threadgroup_position_in_grid]],
                                       uint sgi   [[simdgroup_index_in_threadgroup]],
                                       uint lane  [[thread_index_in_simdgroup]]) {
    threadgroup float part_v[4];
    threadgroup uint  part_i[4];
    const uint row0 = group * 16u + sgi * 4u;
    const uint groups = K / GS;
    const uint bytes_per_group = GS * 6 / 8;
    const uint bytes_per_row = K * 6 / 8;
    device const bfloat4* xv4 = (device const bfloat4*)x;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint g = lane; g < groups; g += 32) {
        device const uchar* w0 = (device const uchar*)codes
            + (ulong)(row0 + 0) * bytes_per_row + g * bytes_per_group;
        device const uchar* w1 = (device const uchar*)codes
            + (ulong)(row0 + 1) * bytes_per_row + g * bytes_per_group;
        device const uchar* w2 = (device const uchar*)codes
            + (ulong)(row0 + 2) * bytes_per_row + g * bytes_per_group;
        device const uchar* w3 = (device const uchar*)codes
            + (ulong)(row0 + 3) * bytes_per_row + g * bytes_per_group;
        const float s0 = float(scales[(row0 + 0) * groups + g]);
        const float b0 = float(biases[(row0 + 0) * groups + g]);
        const float s1 = float(scales[(row0 + 1) * groups + g]);
        const float b1 = float(biases[(row0 + 1) * groups + g]);
        const float s2 = float(scales[(row0 + 2) * groups + g]);
        const float b2 = float(biases[(row0 + 2) * groups + g]);
        const float s3 = float(scales[(row0 + 3) * groups + g]);
        const float b3 = float(biases[(row0 + 3) * groups + g]);
        const uint packet_base = g * (GS / 4);
        for (uint j = 0; j < GS / 4; ++j) {
            const float4 xp = float4(xv4[packet_base + j]);
            decode_q6_reg(w0 + 3 * j, s0, b0, xp, a0);
            decode_q6_reg(w1 + 3 * j, s1, b1, xp, a1);
            decode_q6_reg(w2 + 3 * j, s2, b2, xp, a2);
            decode_q6_reg(w3 + 3 * j, s3, b3, xp, a3);
        }
    }
    a0 = simd_sum(a0);
    a1 = simd_sum(a1);
    a2 = simd_sum(a2);
    a3 = simd_sum(a3);
    if (lane == 0) {
        float bv = -INFINITY;
        uint  bi = 0;
        lfm_argmax_merge(bv, bi, float(bfloat(a0)), row0 + 0);
        lfm_argmax_merge(bv, bi, float(bfloat(a1)), row0 + 1);
        lfm_argmax_merge(bv, bi, float(bfloat(a2)), row0 + 2);
        lfm_argmax_merge(bv, bi, float(bfloat(a3)), row0 + 3);
        part_v[sgi] = bv;
        part_i[sgi] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgi == 0 && lane == 0) {
        float bv = part_v[0];
        uint  bi = part_i[0];
        for (uint s = 1; s < 4; ++s) {
            lfm_argmax_merge(bv, bi, part_v[s], part_i[s]);
        }
        atomic_max_explicit(best, lfm_argmax_key(bv, bi), memory_order_relaxed);
    }
}

// Decode token tail, one 256-thread threadgroup: reads the head's argmax key
// (and clears it for the next step), writes the token, and when `embed` is set
// also produces the next step's input: lfm_gather_rows_q6_bf16's dequantized
// embedding row into `x` and rmsnorm_bf16_v8's normalized row into `out`.
// Thread t dequantizes packs 2t and 2t+1, the eight elements rmsnorm_bf16_v8's
// thread t loads, with the gather's expression; the norm is rmsnorm_bf16_v8's
// body (same staging, stride-256 order, simd_sum, combine and rounding).
kernel void lfm_token_tail(device ulong*        best    [[buffer(0)]],
                           device uint*         out_tok [[buffer(1)]],
                           device const uint*   codes   [[buffer(2)]],
                           device const bfloat* scales  [[buffer(3)]],
                           device const bfloat* biases  [[buffer(4)]],
                           device const bfloat* w       [[buffer(5)]],
                           device bfloat*       x       [[buffer(6)]],
                           device bfloat*       out     [[buffer(7)]],
                           constant uint&       K       [[buffer(8)]],
                           constant uint&       GS      [[buffer(9)]],
                           constant uint&       H       [[buffer(10)]],
                           constant float&      eps     [[buffer(11)]],
                           constant float&      w_bias  [[buffer(12)]],
                           constant uint&       embed   [[buffer(13)]],
                           uint tid  [[thread_index_in_threadgroup]],
                           uint sg   [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint tok_shared;
    threadgroup float partial[ARGMAX_TG / 32];
    threadgroup float inv_rms;
    threadgroup bfloat srow[2048];
    threadgroup bfloat orow[2048];

    if (tid == 0) {
        const ulong key = best[0];
        const uint tok = ~uint(key & 0xFFFFFFFFul);
        best[0] = 0ul;
        out_tok[0] = tok;
        tok_shared = tok;
    }
    if (embed == 0u) {
        return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = tok_shared;

    const uint per_group = GS / 4;
    const uint groups = K / GS;
    const uint bytes_per_group = GS * 6 / 8;
    device const uchar* wrow = (device const uchar*)codes + (ulong)row * (K * 6 / 8);
    threadgroup bfloat4* sv = (threadgroup bfloat4*)srow;
    threadgroup bfloat4* rv = (threadgroup bfloat4*)orow;
    device bfloat4* xv = (device bfloat4*)x;
    device bfloat4* ov = (device bfloat4*)out;
    const uint i = 2 * tid;
    for (uint h = 0; h < 2; ++h) {
        const uint pack = i + h;
        const uint g = pack / per_group;
        const uint j = pack % per_group;
        device const uchar* p = wrow + g * bytes_per_group + 3 * j;
        const uint b0 = p[0], b1 = p[1], b2 = p[2];
        const uint q0 = b0 & 0x3f;
        const uint q1 = ((b0 >> 6) & 0x03) | ((b1 & 0x0f) << 2);
        const uint q2 = ((b1 >> 4) & 0x0f) | ((b2 & 0x03) << 4);
        const uint q3 = (b2 >> 2) & 0x3f;
        const float s = float(scales[row * groups + g]);
        const float b = float(biases[row * groups + g]);
        bfloat4 e;
        e.x = bfloat(s * float(q0) + b);
        e.y = bfloat(s * float(q1) + b);
        e.z = bfloat(s * float(q2) + b);
        e.w = bfloat(s * float(q3) + b);
        xv[pack] = e;
        sv[pack] = e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint j = 0; j < 8; ++j) {
        const uint e = tid + j * ARGMAX_TG;
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
        for (uint j = 0; j < ARGMAX_TG / 32; ++j) {
            total += partial[j];
        }
        inv_rms = rsqrt(total / float(H) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < 8; ++j) {
        const uint e = tid + j * ARGMAX_TG;
        const float gain = w_bias + float(w[e]);
        orow[e] = bfloat(float(srow[e]) * inv_rms * gain);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ov[i] = rv[i];
    ov[i + 1] = rv[i + 1];
}

// --------------------------------------------- expert gather GEMV (single token)

// Gate/up expert gather GEMV fused with the SwiGLU activation, at the
// compile-time decode shape (K, group size, expert rows).
kernel void lfm_gemv_q4_gather_act_bf16(device const uint*   cg  [[buffer(0)]],
                                        device const bfloat* sg  [[buffer(1)]],
                                        device const bfloat* bg  [[buffer(2)]],
                                        device const uint*   cu  [[buffer(3)]],
                                        device const bfloat* su  [[buffer(4)]],
                                        device const bfloat* bu  [[buffer(5)]],
                                        device const bfloat* x   [[buffer(6)]],
                                        device const uint*   ids [[buffer(7)]],
                                        device bfloat*       act [[buffer(8)]],
                                        uint group [[threadgroup_position_in_grid]],
                                        uint lane  [[thread_index_in_threadgroup]]) {
    constexpr uint K = LFM_HIDDEN;
    constexpr uint GS = LFM_GS;
    constexpr uint N = LFM_MOE_INTER;
    const uint words = K / 8;
    const uint blocks = words / 4;
    const uint bpg = GS / 32;
    const uint groups = K / GS;
    const uint slot = group / N;
    const uint row  = group % N;
    const uint expert = ids[slot];
    const ulong wrow_index = (ulong)expert * N + row;
    device const bfloat4* xv = (device const bfloat4*)x;
    float gsum = 0.0f;
    float usum = 0.0f;
    {
        device const uint4* wrow_g = (device const uint4*)(cg + wrow_index * words);
        device const uint4* wrow_u = (device const uint4*)(cu + wrow_index * words);
        for (uint i = lane; i < blocks; i += 32) {
            uint g = i / bpg;
            float sgv = float(sg[wrow_index * groups + g]);
            float bgv = float(bg[wrow_index * groups + g]);
            float suv = float(su[wrow_index * groups + g]);
            float buv = float(bu[wrow_index * groups + g]);
            uint4 wg = wrow_g[i];
            uint4 wu = wrow_u[i];
            gsum += dot_word_q4(wg.x, sgv, bgv, xv, 4 * i)
                + dot_word_q4(wg.y, sgv, bgv, xv, 4 * i + 1)
                + dot_word_q4(wg.z, sgv, bgv, xv, 4 * i + 2)
                + dot_word_q4(wg.w, sgv, bgv, xv, 4 * i + 3);
            usum += dot_word_q4(wu.x, suv, buv, xv, 4 * i)
                + dot_word_q4(wu.y, suv, buv, xv, 4 * i + 1)
                + dot_word_q4(wu.z, suv, buv, xv, 4 * i + 2)
                + dot_word_q4(wu.w, suv, buv, xv, 4 * i + 3);
        }
    }
    gsum = simd_sum(gsum);
    usum = simd_sum(usum);
    if (lane == 0) {
        const float gv = float(bfloat(gsum));
        const float silu = float(bfloat(gv / (1.0f + exp(-gv))));
        act[slot * N + row] = bfloat(silu * float(bfloat(usum)));
    }
}

// Decode dense FFN: the gate and up GEMVs (lfm_gemv_q4_bf16's per-lane block
// walk and simd_sum, one chain per projection, as the expert gather above) and
// silu_mul_bf16's expression on their BF16-rounded outputs, one row per
// 32-thread threadgroup.
kernel void lfm_gemv_q4_gu_act_bf16(device const uint*   cg  [[buffer(0)]],
                                    device const bfloat* sg  [[buffer(1)]],
                                    device const bfloat* bg  [[buffer(2)]],
                                    device const uint*   cu  [[buffer(3)]],
                                    device const bfloat* su  [[buffer(4)]],
                                    device const bfloat* bu  [[buffer(5)]],
                                    device const bfloat* x   [[buffer(6)]],
                                    device bfloat*       act [[buffer(7)]],
                                    constant uint&       K   [[buffer(8)]],
                                    constant uint&       GS  [[buffer(9)]],
                                    uint row  [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_threadgroup]]) {
    const uint words = K / 8;
    const uint blocks = words / 4;
    const uint bpg = GS / 32;
    const uint groups = K / GS;
    device const bfloat4* xv = (device const bfloat4*)x;
    float gsum = 0.0f;
    float usum = 0.0f;
    {
        device const uint4* wrow_g = (device const uint4*)(cg + (ulong)row * words);
        device const uint4* wrow_u = (device const uint4*)(cu + (ulong)row * words);
        for (uint i = lane; i < blocks; i += 32) {
            uint g = i / bpg;
            float sgv = float(sg[row * groups + g]);
            float bgv = float(bg[row * groups + g]);
            float suv = float(su[row * groups + g]);
            float buv = float(bu[row * groups + g]);
            uint4 wg = wrow_g[i];
            uint4 wu = wrow_u[i];
            gsum += dot_word_q4(wg.x, sgv, bgv, xv, 4 * i)
                + dot_word_q4(wg.y, sgv, bgv, xv, 4 * i + 1)
                + dot_word_q4(wg.z, sgv, bgv, xv, 4 * i + 2)
                + dot_word_q4(wg.w, sgv, bgv, xv, 4 * i + 3);
            usum += dot_word_q4(wu.x, suv, buv, xv, 4 * i)
                + dot_word_q4(wu.y, suv, buv, xv, 4 * i + 1)
                + dot_word_q4(wu.z, suv, buv, xv, 4 * i + 2)
                + dot_word_q4(wu.w, suv, buv, xv, 4 * i + 3);
        }
    }
    gsum = simd_sum(gsum);
    usum = simd_sum(usum);
    if (lane == 0) {
        const float gv = float(bfloat(gsum));
        const float silu = float(bfloat(gv / (1.0f + exp(-gv))));
        act[row] = bfloat(silu * float(bfloat(usum)));
    }
}

// Down expert GEMV with one simdgroup per selected expert, fused combine, at
// the compile-time decode shape. TOPK stays a runtime bound: a fixed four-term
// combine loop is re-associated by fast-math.
kernel void lfm_gemv_q4_gather_down_combine4_bf16(
    device const uint*   codes  [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const bfloat* act    [[buffer(3)]],
    device const uint*   ids    [[buffer(4)]],
    device const bfloat* weights[[buffer(5)]],
    device bfloat*       out    [[buffer(6)]],
    constant uint&       TOPK   [[buffer(7)]],
    uint row  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (sg >= TOPK) {
        return;
    }
    constexpr uint K = LFM_MOE_INTER;
    constexpr uint GS = LFM_GS;
    constexpr uint N = LFM_HIDDEN;
    const uint words = K / 8;
    const uint blocks = words / 4;
    const uint bpg = GS / 32;
    const uint groups = K / GS;
    const uint expert = ids[sg];
    const ulong wrow_index = (ulong)expert * N + row;
    device const uint4* wrow = (device const uint4*)(codes + wrow_index * words);
    device const bfloat4* xv = (device const bfloat4*)(act + (ulong)sg * K);
    float sum = 0.0f;
    for (uint i = lane; i < blocks; i += 32) {
        uint g = i / bpg;
        float s = float(scales[wrow_index * groups + g]);
        float b = float(biases[wrow_index * groups + g]);
        uint4 w4 = wrow[i];
        sum += dot_word_q4(w4.x, s, b, xv, 4 * i)
            + dot_word_q4(w4.y, s, b, xv, 4 * i + 1)
            + dot_word_q4(w4.z, s, b, xv, 4 * i + 2)
            + dot_word_q4(w4.w, s, b, xv, 4 * i + 3);
    }
    sum = simd_sum(sum);
    threadgroup float partial[4];
    if (lane == 0) {
        partial[sg] = float(bfloat(sum));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint r = 0; r < TOPK; ++r) {
            total += float(bfloat(float(weights[r]) * partial[r]));
        }
        out[row] = bfloat(total);
    }
}

// -------------------------------------------------- GPU MoE routing (prefill)

kernel void lfm_moe_route_batch(device const float*  logits [[buffer(0)]],
                                device const float*  bias   [[buffer(1)]],
                                device uint*         ids    [[buffer(2)]],
                                device bfloat*       gw     [[buffer(3)]],
                                constant uint&       E      [[buffer(4)]],
                                constant uint&       TOPK   [[buffer(5)]],
                                constant float&      eps    [[buffer(6)]],
                                uint token [[threadgroup_position_in_grid]],
                                uint tid   [[thread_index_in_threadgroup]]) {
    threadgroup float s[128];
    threadgroup uchar used[128];
    if (tid < E) {
        const float z = logits[(ulong)token * E + tid];
        s[tid] = 1.0f / (1.0f + exp(-z));
        used[tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint chosen[8];
        for (uint r = 0; r < TOPK; ++r) {
            float best = -INFINITY;
            uint best_e = 0;
            for (uint e = 0; e < E; ++e) {
                if (used[e]) {
                    continue;
                }
                const float v = s[e] + bias[e];
                if (v > best || (v == best && e < best_e)) {
                    best = v;
                    best_e = e;
                }
            }
            chosen[r] = best_e;
            used[best_e] = 1;
        }
        float denom = eps;
        for (uint r = 0; r < TOPK; ++r) {
            denom += s[chosen[r]];
        }
        for (uint r = 0; r < TOPK; ++r) {
            ids[(ulong)token * TOPK + r] = chosen[r];
            gw[(ulong)token * TOPK + r] = bfloat(s[chosen[r]] / denom);
        }
    }
}

kernel void lfm_moe_zero_counts(device uint* counts [[buffer(0)]],
                                constant uint& E [[buffer(1)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid < E) {
        counts[gid] = 0;
    }
}

kernel void lfm_moe_count(device const uint* ids    [[buffer(0)]],
                          device uint*       counts [[buffer(1)]],
                          constant uint&     S      [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid < S) {
        const uint e = ids[gid];
        atomic_fetch_add_explicit((device atomic_uint*)&counts[e], 1u,
                                  memory_order_relaxed);
    }
}

kernel void lfm_moe_offsets(device const uint* counts  [[buffer(0)]],
                            device uint*       offsets [[buffer(1)]],
                            constant uint&     E       [[buffer(2)]],
                            uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) {
        uint acc = 0;
        for (uint e = 0; e < E; ++e) {
            offsets[e] = acc;
            acc += counts[e];
        }
    }
}

// Stable expert scatter over a token range per thread: two passes (count,
// then write) with a threadgroup exclusive scan, so slots keep token-major,
// rank-major order and every position is identical to the serial walk.
kernel void lfm_moe_sorted_parallel(device const uint* ids        [[buffer(0)]],
                                    device const uint* offsets    [[buffer(1)]],
                                    device uint*       ids_sorted [[buffer(2)]],
                                    device uint*       slotpos    [[buffer(3)]],
                                    constant uint&     M          [[buffer(4)]],
                                    constant uint&     E          [[buffer(5)]],
                                    constant uint&     TOPK       [[buffer(6)]],
                                    uint e   [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint ntg [[threads_per_threadgroup]]) {
    threadgroup uint scan[256];
    if (e >= E || ntg > 256u) {
        return;
    }
    const uint per = (M + ntg - 1u) / ntg;
    const uint t0 = tid * per;
    const uint t1 = min(M, t0 + per);
    uint c = 0;
    for (uint t = t0; t < t1; ++t) {
        for (uint r = 0; r < TOPK; ++r) {
            if (ids[(ulong)t * TOPK + r] == e) {
                ++c;
            }
        }
    }
    scan[tid] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint acc = 0;
        for (uint i = 0; i < ntg; ++i) {
            const uint v = scan[i];
            scan[i] = acc;
            acc += v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint pos = offsets[e] + scan[tid];
    for (uint t = t0; t < t1; ++t) {
        for (uint r = 0; r < TOPK; ++r) {
            if (ids[(ulong)t * TOPK + r] == e) {
                ids_sorted[pos] = t;
                slotpos[(ulong)t * TOPK + r] = pos;
                ++pos;
            }
        }
    }
}

kernel void lfm_moe_zero_blocks(device uint4* gu      [[buffer(0)]],
                                device uint4* dn      [[buffer(1)]],
                                constant uint& n_gu   [[buffer(2)]],
                                constant uint& n_dn   [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid < n_gu) {
        gu[gid] = uint4(0);
    }
    if (gid < n_dn) {
        dn[gid] = uint4(0);
    }
}


// Per-expert output offsets for the two block maps (64 experts, one thread).
kernel void lfm_moe_block_offsets(device const uint* counts [[buffer(0)]],
                                  device uint*       gu_off [[buffer(1)]],
                                  device uint*       dn_off [[buffer(2)]],
                                  constant uint&     BM     [[buffer(3)]],
                                  constant uint&     BN     [[buffer(4)]],
                                  constant uint&     GU_N   [[buffer(5)]],
                                  constant uint&     DN_N   [[buffer(6)]],
                                  constant uint&     E      [[buffer(7)]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    if (tid != 0) {
        return;
    }
    uint ag = 0;
    uint ad = 0;
    const uint gu_tiles = GU_N / BN;
    const uint dn_tiles = DN_N / BN;
    for (uint e = 0; e < E; ++e) {
        const uint n = counts[e];
        gu_off[e] = ag;
        dn_off[e] = ad;
        const uint rows = (n + BM - 1) / BM;
        ag += rows * gu_tiles;
        ad += rows * dn_tiles;
    }
}

// One threadgroup per expert; each thread writes a different block entry, so
// consecutive threads produce consecutive coalesced uint4 stores.
kernel void lfm_moe_blocks_parallel(device const uint* counts  [[buffer(0)]],
                                    device const uint* offsets [[buffer(1)]],
                                    device const uint* gu_off  [[buffer(2)]],
                                    device const uint* dn_off  [[buffer(3)]],
                                    device uint4*      gu      [[buffer(4)]],
                                    device uint4*      dn      [[buffer(5)]],
                                    constant uint&     BM      [[buffer(6)]],
                                    constant uint&     GU_N    [[buffer(7)]],
                                    constant uint&     DN_N    [[buffer(8)]],
                                    constant uint&     BN      [[buffer(9)]],
                                    uint e   [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
    const uint n = counts[e];
    if (n == 0) {
        return;
    }
    const uint off = offsets[e];
    const uint rows = (n + BM - 1) / BM;
    const uint gu_tiles = GU_N / BN;
    const uint dn_tiles = DN_N / BN;
    const uint ngu = rows * gu_tiles;
    const uint ndn = rows * dn_tiles;
    const uint bg = gu_off[e];
    const uint bd = dn_off[e];
    for (uint i = tid; i < ngu; i += 128) {
        const uint mt = i / gu_tiles;
        const uint nt = i % gu_tiles;
        const uint m0 = off + mt * BM;
        const uint m_end = min(m0 + BM, off + n);
        const uint n0 = nt * BN;
        gu[bg + i] = uint4(m0, e * GU_N + n0, n0, m_end);
    }
    for (uint i = tid; i < ndn; i += 128) {
        const uint mt = i / dn_tiles;
        const uint nt = i % dn_tiles;
        const uint m0 = off + mt * BM;
        const uint m_end = min(m0 + BM, off + n);
        const uint n0 = nt * BN;
        dn[bd + i] = uint4(m0, e * DN_N + n0, n0, m_end);
    }
}


// Single-token attention with the reference's sdpa_vector arithmetic
// (MLX 0.32.3 sdpa_vector for D=64: two values per lane, 32 simdgroups,
// online FP32 softmax, FP32 output accumulation). The reference uses this
// kernel for L=1 calls below its two-pass threshold (4096 keys for GQA, or
// 1024 on GPUs whose architecture name ends in 'd' or 's'); the engine uses
// it below SDPA_2PASS_MIN_LEN, so the last prompt token and the decode steps
// take it instead of the tensor-op prefill tile.
kernel void lfm_sdpa_vector_bf16(device const bfloat* q       [[buffer(0)]],
                                 device const bfloat* k_cache [[buffer(1)]],
                                 device const bfloat* v_cache [[buffer(2)]],
                                 device bfloat*       out     [[buffer(3)]],
                                 constant uint&       max_seq [[buffer(4)]],
                                 constant uint&       len     [[buffer(5)]],
                                 constant uint&       nq      [[buffer(6)]],
                                 constant uint&       group   [[buffer(7)]],
                                 constant float&      scale   [[buffer(8)]],
                                 uint hq   [[threadgroup_position_in_grid]],
                                 uint sg   [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint D = 64;
    constexpr uint BD = 32;
    constexpr uint BN = 32;
    constexpr uint QPT = D / BD;  // 2
    constexpr uint VPT = D / BD;  // 2

    threadgroup float outputs[BN * BD];
    threadgroup float max_scores[BN];
    threadgroup float sum_exp_scores[BN];

    const uint kvh = hq / group;
    device const bfloat* qh = q + (ulong)hq * D + lane * QPT;
    device const bfloat* kb = k_cache + (ulong)kvh * max_seq * D + lane * QPT;
    device const bfloat* vb = v_cache + (ulong)kvh * max_seq * D + lane * QPT;

    float qv[QPT];
    float ov[VPT];
    for (uint i = 0; i < QPT; ++i) {
        qv[i] = scale * float(qh[i]);
    }
    for (uint i = 0; i < VPT; ++i) {
        ov[i] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;
    // Positions still fold into the online softmax one at a time, in the same
    // increasing order with the same per-lane partial sums, so every stored
    // value is unchanged. Only the issue order of the loads changes: U
    // positions' K and V rows are fetched up front, which keeps U independent
    // global reads in flight per lane instead of one dependent chain per
    // position.
    constexpr uint U = 8;
    uint p = sg;
    for (; p + (U - 1) * BN < len; p += U * BN) {
        float kk[U][QPT];
        float vv[U][VPT];
        for (uint u = 0; u < U; ++u) {
            const ulong off = (ulong)(p + u * BN) * D;
            for (uint j = 0; j < QPT; ++j) {
                kk[u][j] = float(kb[off + j]);
            }
            for (uint j = 0; j < VPT; ++j) {
                vv[u][j] = float(vb[off + j]);
            }
        }
        for (uint u = 0; u < U; ++u) {
            float s = 0.0f;
            for (uint j = 0; j < QPT; ++j) {
                s += qv[j] * kk[u][j];
            }
            s = simd_sum(s);
            const float new_max = max(max_score, s);
            const float factor = metal::fast::exp(max_score - new_max);
            const float exp_score = metal::fast::exp(s - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (uint j = 0; j < VPT; ++j) {
                ov[j] = ov[j] * factor + exp_score * vv[u][j];
            }
        }
    }
    for (; p < len; p += BN) {
        float s = 0.0f;
        for (uint j = 0; j < QPT; ++j) {
            s += qv[j] * float(kb[(ulong)p * D + j]);
        }
        s = simd_sum(s);
        const float new_max = max(max_score, s);
        const float factor = metal::fast::exp(max_score - new_max);
        const float exp_score = metal::fast::exp(s - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (uint j = 0; j < VPT; ++j) {
            ov[j] = ov[j] * factor + exp_score * float(vb[(ulong)p * D + j]);
        }
    }

    if (lane == 0) {
        max_scores[sg] = max_score;
        sum_exp_scores[sg] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = max_scores[lane];
    const float new_max = simd_max(max_score);
    const float factor = metal::fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[lane] * factor);

    for (uint i = 0; i < VPT; ++i) {
        outputs[lane * BD + sg] = ov[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ov[i] = simd_sum(outputs[sg * BD + lane] * factor);
        ov[i] = sum_exp_score == 0.0f ? ov[i] : (ov[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint i = 0; i < VPT; ++i) {
            out[(ulong)hq * D + sg * VPT + i] = bfloat(ov[i]);
        }
    }
}


// Decode-only: lfm_norm_rope_qk_scatter_bf16 folded into lfm_sdpa_vector_bf16
// for the single-pass range. In threadgroup hq, simdgroup 0 normalizes and
// rotates query head hq and simdgroup 1 the key row of its KV head at the
// current position len - 1, with the fused kernel's lane layout, expressions
// and bfloat roundings; the first query head of each KV head also writes that
// key row into the cache. Both rows are staged as bfloat in threadgroup
// memory, so the body reads exactly the values the separate kernels stored.
// The cached positions go through lfm_sdpa_vector_bf16's loops and
// partition; only the simdgroup owning the current position (its last one)
// takes that position's key from threadgroup memory, in a copy of the
// unrolled block or of the tail step, so no threadgroup reads the cache row
// this dispatch writes.
kernel void lfm_sdpa_vector_norm_rope_bf16(device const bfloat* xq      [[buffer(0)]],
                                           device const bfloat* wq      [[buffer(1)]],
                                           device const bfloat* xk      [[buffer(2)]],
                                           device const bfloat* wk      [[buffer(3)]],
                                           device bfloat*       k_cache [[buffer(4)]],
                                           device const bfloat* v_cache [[buffer(5)]],
                                           device bfloat*       out     [[buffer(6)]],
                                           constant uint&       max_seq [[buffer(7)]],
                                           constant uint&       len     [[buffer(8)]],
                                           constant uint&       nq      [[buffer(9)]],
                                           constant uint&       group   [[buffer(10)]],
                                           constant float&      scale   [[buffer(11)]],
                                           constant uint&       rot     [[buffer(12)]],
                                           constant float&      theta   [[buffer(13)]],
                                           constant float&      eps     [[buffer(14)]],
                                           uint hq   [[threadgroup_position_in_grid]],
                                           uint sg   [[simdgroup_index_in_threadgroup]],
                                           uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint D = 64;
    constexpr uint BD = 32;
    constexpr uint BN = 32;
    constexpr uint QPT = D / BD;  // 2
    constexpr uint VPT = D / BD;  // 2

    threadgroup float outputs[BN * BD];
    threadgroup float max_scores[BN];
    threadgroup float sum_exp_scores[BN];
    threadgroup bfloat qk_rows[2 * D];

    const uint kvh = hq / group;
    const uint base_pos = len - 1;
    if (sg < 2) {
        device const bfloat* x;
        device const bfloat* w;
        uint h;
        bool is_q;
        if (sg == 0) {
            x = xq; w = wq; h = hq; is_q = true;
        } else {
            x = xk; w = wk; h = kvh; is_q = false;
        }
        const ulong base = (ulong)h * D;
        const float v0 = float(x[base + lane]);
        const float v1 = float(x[base + lane + 32]);
        float acc = v0 * v0 + v1 * v1;
        acc = simd_sum(acc);
        const float inv = rsqrt(acc / 64.0f + eps);
        const float n0 = float(bfloat(v0 * inv * float(w[lane])));
        const float n1 = float(bfloat(v1 * inv * float(w[lane + 32])));
        float inv_freq = pow(theta, -2.0f * float(lane) / float(rot));
        float ang = float(base_pos) * inv_freq;
        float c = cos(ang);
        float s = sin(ang);
        const float t1 = n1 * s;
        const float t3 = n0 * s;
        const bfloat r0 = bfloat(fma(n0, c, -t1));
        const bfloat r1 = bfloat(fma(n1, c, t3));
        qk_rows[sg * D + lane] = r0;
        qk_rows[sg * D + lane + 32] = r1;
        if (!is_q && hq % group == 0) {
            const ulong dst = ((ulong)h * max_seq + base_pos) * D;
            k_cache[dst + lane] = r0;
            k_cache[dst + lane + 32] = r1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const bfloat* kb = k_cache + (ulong)kvh * max_seq * D + lane * QPT;
    device const bfloat* vb = v_cache + (ulong)kvh * max_seq * D + lane * QPT;

    float qv[QPT];
    float ov[VPT];
    for (uint i = 0; i < QPT; ++i) {
        qv[i] = scale * float(qk_rows[lane * QPT + i]);
    }
    for (uint i = 0; i < VPT; ++i) {
        ov[i] = 0.0f;
    }

    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;
    constexpr uint U = 8;
    uint p = sg;
    // Positions of other simdgroups never equal base_pos, so bounding by
    // base_pos instead of len leaves their blocks and tail steps unchanged;
    // the owner of base_pos stops before the block or step holding it.
    for (; p + (U - 1) * BN < base_pos; p += U * BN) {
        float kk[U][QPT];
        float vv[U][VPT];
        for (uint u = 0; u < U; ++u) {
            const ulong off = (ulong)(p + u * BN) * D;
            for (uint j = 0; j < QPT; ++j) {
                kk[u][j] = float(kb[off + j]);
            }
            for (uint j = 0; j < VPT; ++j) {
                vv[u][j] = float(vb[off + j]);
            }
        }
        for (uint u = 0; u < U; ++u) {
            float s = 0.0f;
            for (uint j = 0; j < QPT; ++j) {
                s += qv[j] * kk[u][j];
            }
            s = simd_sum(s);
            const float new_max = max(max_score, s);
            const float factor = metal::fast::exp(max_score - new_max);
            const float exp_score = metal::fast::exp(s - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (uint j = 0; j < VPT; ++j) {
                ov[j] = ov[j] * factor + exp_score * vv[u][j];
            }
        }
    }
    if (p + (U - 1) * BN == base_pos) {
        // The owner's last unrolled block: base_pos is its u = U - 1 entry.
        float kk[U][QPT];
        float vv[U][VPT];
        for (uint u = 0; u < U; ++u) {
            const ulong off = (ulong)(p + u * BN) * D;
            if (u < U - 1) {
                for (uint j = 0; j < QPT; ++j) {
                    kk[u][j] = float(kb[off + j]);
                }
            } else {
                for (uint j = 0; j < QPT; ++j) {
                    kk[u][j] = float(qk_rows[D + lane * QPT + j]);
                }
            }
            for (uint j = 0; j < VPT; ++j) {
                vv[u][j] = float(vb[off + j]);
            }
        }
        for (uint u = 0; u < U; ++u) {
            float s = 0.0f;
            for (uint j = 0; j < QPT; ++j) {
                s += qv[j] * kk[u][j];
            }
            s = simd_sum(s);
            const float new_max = max(max_score, s);
            const float factor = metal::fast::exp(max_score - new_max);
            const float exp_score = metal::fast::exp(s - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (uint j = 0; j < VPT; ++j) {
                ov[j] = ov[j] * factor + exp_score * vv[u][j];
            }
        }
        p += U * BN;
    }
    for (; p < base_pos; p += BN) {
        float s = 0.0f;
        for (uint j = 0; j < QPT; ++j) {
            s += qv[j] * float(kb[(ulong)p * D + j]);
        }
        s = simd_sum(s);
        const float new_max = max(max_score, s);
        const float factor = metal::fast::exp(max_score - new_max);
        const float exp_score = metal::fast::exp(s - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (uint j = 0; j < VPT; ++j) {
            ov[j] = ov[j] * factor + exp_score * float(vb[(ulong)p * D + j]);
        }
    }
    if (p == base_pos) {
        // The owner's last tail step.
        float s = 0.0f;
        for (uint j = 0; j < QPT; ++j) {
            s += qv[j] * float(qk_rows[D + lane * QPT + j]);
        }
        s = simd_sum(s);
        const float new_max = max(max_score, s);
        const float factor = metal::fast::exp(max_score - new_max);
        const float exp_score = metal::fast::exp(s - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (uint j = 0; j < VPT; ++j) {
            ov[j] = ov[j] * factor + exp_score * float(vb[(ulong)p * D + j]);
        }
    }

    if (lane == 0) {
        max_scores[sg] = max_score;
        sum_exp_scores[sg] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_score = max_scores[lane];
    const float new_max = simd_max(max_score);
    const float factor = metal::fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[lane] * factor);

    for (uint i = 0; i < VPT; ++i) {
        outputs[lane * BD + sg] = ov[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ov[i] = simd_sum(outputs[sg * BD + lane] * factor);
        ov[i] = sum_exp_score == 0.0f ? ov[i] : (ov[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint i = 0; i < VPT; ++i) {
            out[(ulong)hq * D + sg * VPT + i] = bfloat(ov[i]);
        }
    }
}


// Two-pass single-token attention over a split key range, following MLX
// 0.32.3 `sdpa_vector_2pass_1` / `sdpa_vector_2pass_2` for D=64: the first
// pass keeps one online FP32 softmax per (query head, block) and stores the
// BF16 partial plus its FP32 max/sum; the second pass combines the blocks.
// The reference routes L=1 attention here for long contexts, and the blocked
// grid exposes far more threadgroups than the single-pass vector kernel.
kernel void lfm_sdpa_vector_2pass1_bf16(
        device const bfloat* q        [[buffer(0)]],
        device const bfloat* k_cache  [[buffer(1)]],
        device const bfloat* v_cache  [[buffer(2)]],
        device bfloat*       partials [[buffer(3)]],
        device float*        sums     [[buffer(4)]],
        device float*        maxs     [[buffer(5)]],
        constant uint&       max_seq  [[buffer(6)]],
        constant uint&       len      [[buffer(7)]],
        constant uint&       group    [[buffer(8)]],
        constant float&      scale    [[buffer(9)]],
        constant uint&       blocks   [[buffer(10)]],
        uint2 tg   [[threadgroup_position_in_grid]],
        uint sg    [[simdgroup_index_in_threadgroup]],
        uint lane  [[thread_index_in_simdgroup]]) {
    constexpr uint D = 64;
    const uint kvh = tg.x;
    const uint block = tg.y;
    const uint hq = kvh * group + sg;
    device const bfloat* qh = q + (ulong)hq * D + lane * 2;
    device const bfloat* kp =
        k_cache + (ulong)kvh * max_seq * D + (ulong)block * D + lane * 2;
    device const bfloat* vp =
        v_cache + (ulong)kvh * max_seq * D + (ulong)block * D + lane * 2;
    const float q0 = scale * float(qh[0]);
    const float q1 = scale * float(qh[1]);
    float max_score = -INFINITY;
    float sum_exp_score = 0.0f;
    float o0 = 0.0f;
    float o1 = 0.0f;
    // Interleaved blocks: block b takes positions b, b + blocks, ... This is
    // the reference's partition and leaves no block without a position for
    // every length the two-pass path serves.
    for (uint t = block; t < len; t += blocks) {
        float s = q0 * float(kp[0]) + q1 * float(kp[1]);
        s = simd_sum(s);
        const float new_max = max(max_score, s);
        const float factor = metal::fast::exp(max_score - new_max);
        const float exp_score = metal::fast::exp(s - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        o0 = o0 * factor + exp_score * float(vp[0]);
        o1 = o1 * factor + exp_score * float(vp[1]);
        kp += (ulong)blocks * D;
        vp += (ulong)blocks * D;
    }

    device bfloat* op = partials + ((ulong)hq * blocks + block) * D + lane * 2;
    op[0] = bfloat(o0);
    op[1] = bfloat(o1);
    if (lane == 0) {
        sums[hq * blocks + block] = sum_exp_score;
        maxs[hq * blocks + block] = max_score;
    }
}

kernel void lfm_sdpa_vector_2pass2_bf16(
        device const bfloat* partials [[buffer(0)]],
        device const float*  sums     [[buffer(1)]],
        device const float*  maxs     [[buffer(2)]],
        device bfloat*       out      [[buffer(3)]],
        constant uint&       blocks   [[buffer(4)]],
        uint hq     [[threadgroup_position_in_grid]],
        uint sg     [[simdgroup_index_in_threadgroup]],
        uint lane   [[thread_index_in_simdgroup]]) {
    constexpr uint D = 64;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint EPT = D / BD;  // 2

    threadgroup float outputs[BN * BD];

    device const bfloat* pp = partials + (ulong)hq * blocks * D + sg * D + lane * EPT;
    device const float* sp = sums + (ulong)hq * blocks;
    device const float* mp = maxs + (ulong)hq * blocks;
    device bfloat* op = out + (ulong)hq * D + sg * EPT;

    float max_score = -INFINITY;
    for (uint b = 0; b < blocks / BN; ++b) {
        max_score = max(max_score, mp[lane + BN * b]);
    }
    max_score = simd_max(max_score);

    float sum_exp_score = 0.0f;
    for (uint b = 0; b < blocks / BN; ++b) {
        const float factor = metal::fast::exp(mp[lane + BN * b] - max_score);
        sum_exp_score += factor * sp[lane + BN * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);

    float o[EPT] = {0.0f, 0.0f};
    for (uint b = 0; b < blocks / BN; ++b) {
        const float factor = metal::fast::exp(mp[sg] - max_score);
        for (uint i = 0; i < EPT; ++i) {
            o[i] += factor * float(pp[i]);
        }
        mp += BN;
        sp += BN;
        pp += BN * D;
    }

    for (uint i = 0; i < EPT; ++i) {
        outputs[lane * BD + sg] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[sg * BD + lane]);
        o[i] = sum_exp_score == 0.0f ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint i = 0; i < EPT; ++i) {
            op[i] = bfloat(o[i]);
        }
    }
}


// Four-head variant of the two-pass stage one: one warp owns a block and
// accumulates all four query heads that share the KV head. K and V are read
// once per position instead of once per query head (the four warps of the
// generic stage share nothing at the register level). Each head keeps its own
// online max/sum and FP32 output pair, and the partial layout is the generic
// stage's, so the same combine kernel finishes the reduction.
kernel void lfm_sdpa_vector_2pass1_gqa4_bf16(
        device const bfloat* q        [[buffer(0)]],
        device const bfloat* k_cache  [[buffer(1)]],
        device const bfloat* v_cache  [[buffer(2)]],
        device bfloat*       partials [[buffer(3)]],
        device float*        sums     [[buffer(4)]],
        device float*        maxs     [[buffer(5)]],
        constant uint&       max_seq  [[buffer(6)]],
        constant uint&       len      [[buffer(7)]],
        constant float&      scale    [[buffer(9)]],
        constant uint&       blocks   [[buffer(10)]],
        uint2 tg   [[threadgroup_position_in_grid]],
        uint lane  [[thread_index_in_simdgroup]]) {
    constexpr uint D = 64;
    constexpr uint G = 4;
    const uint kvh = tg.x;
    const uint block = tg.y;

    float q0[G], q1[G], mx[G], sm[G], o0[G], o1[G];
    for (uint h = 0; h < G; ++h) {
        device const bfloat* qh = q + (ulong)(kvh * G + h) * D + lane * 2;
        q0[h] = scale * float(qh[0]);
        q1[h] = scale * float(qh[1]);
        mx[h] = -INFINITY;
        sm[h] = 0.0f;
        o0[h] = 0.0f;
        o1[h] = 0.0f;
    }
    device const bfloat* kp =
        k_cache + (ulong)kvh * max_seq * D + (ulong)block * D + lane * 2;
    device const bfloat* vp =
        v_cache + (ulong)kvh * max_seq * D + (ulong)block * D + lane * 2;
    for (uint t = block; t < len; t += blocks) {
        const float k0 = float(kp[0]);
        const float k1 = float(kp[1]);
        const float v0 = float(vp[0]);
        const float v1 = float(vp[1]);
        for (uint h = 0; h < G; ++h) {
            float s = simd_sum(q0[h] * k0 + q1[h] * k1);
            const float nm = max(mx[h], s);
            const float f = metal::fast::exp(mx[h] - nm);
            const float e = metal::fast::exp(s - nm);
            mx[h] = nm;
            sm[h] = sm[h] * f + e;
            o0[h] = o0[h] * f + e * v0;
            o1[h] = o1[h] * f + e * v1;
        }
        kp += (ulong)blocks * D;
        vp += (ulong)blocks * D;
    }
    for (uint h = 0; h < G; ++h) {
        const uint hq = kvh * G + h;
        device bfloat* op =
            partials + ((ulong)hq * blocks + block) * D + lane * 2;
        op[0] = bfloat(o0[h]);
        op[1] = bfloat(o1[h]);
        if (lane == 0) {
            sums[hq * blocks + block] = sm[h];
            maxs[hq * blocks + block] = mx[h];
        }
    }
}
