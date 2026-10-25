// Modified from Lily (Perplexity AI, Apache-2.0): see VENDORED.md for the changes to this file.
// Affine Q4/Q8 dequantization to BF16.
// Q4 codes pack eight low-nibble-first values per u32; w = scale*q + bias.
// Portions use MLX affine quantization constructions; see NOTICE.

#include <metal_stdlib>
using namespace metal;

// Materializes Q8 weights with one thread per code word.
kernel void dequant_q8_bf16(device const uint*   codes  [[buffer(0)]],
                            device const bfloat* scales [[buffer(1)]],
                            device const bfloat* biases [[buffer(2)]],
                            device bfloat*       out    [[buffer(3)]],
                            constant uint&       K      [[buffer(4)]],
                            constant uint&       GS     [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const uint words = K / 4;
    const uint w = gid.x;
    const uint row = gid.y;
    if (w >= words) {
        return;
    }
    const uint groups = K / GS;
    uint g = w / (GS / 4);
    float s = float(scales[row * groups + g]);
    float b = float(biases[row * groups + g]);
    uint word = codes[(ulong)row * words + w];
    float4 q = float4(float(word & 0xFF), float((word >> 8) & 0xFF),
                      float((word >> 16) & 0xFF), float((word >> 24) & 0xFF));
    ((device bfloat4*)(out + (ulong)row * K))[w] = bfloat4(q * s + b);
}

// Dequantizes one Q4 code word to eight BF16 values.
static inline void store_word_q4(uint word, float s, float b,
                                 device bfloat4* out, uint w) {
    float4 lo = float4(float((word >> 0) & 0xF), float((word >> 4) & 0xF),
                       float((word >> 8) & 0xF), float((word >> 12) & 0xF));
    float4 hi = float4(float((word >> 16) & 0xF), float((word >> 20) & 0xF),
                       float((word >> 24) & 0xF), float((word >> 28) & 0xF));
    out[2 * w] = bfloat4(lo * s + b);
    out[2 * w + 1] = bfloat4(hi * s + b);
}

// Materializes Q4 weights with one thread per code word.
kernel void dequant_q4_bf16(device const uint*   codes  [[buffer(0)]],
                            device const bfloat* scales [[buffer(1)]],
                            device const bfloat* biases [[buffer(2)]],
                            device bfloat*       out    [[buffer(3)]],
                            constant uint&       K      [[buffer(4)]],
                            constant uint&       GS     [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const uint words = K / 8;
    const uint w = gid.x;
    const uint row = gid.y;
    if (w >= words) {
        return;
    }
    const uint groups = K / GS;
    uint g = w / (GS / 8);
    float s = float(scales[row * groups + g]);
    float b = float(biases[row * groups + g]);
    uint word = codes[(ulong)row * words + w];
    store_word_q4(word, s, b, (device bfloat4*)(out + (ulong)row * K), w);
}

// Bulk Q4 -> BF16 dequantization, four 64-value quantization groups per
// thread. K is a multiple of 128, so a thread's 128 values stay inside one
// row and use exactly two scale/bias pairs.
kernel void dequant_q4_bf16_bulk(device const uint*   codes  [[buffer(0)]],
                                 device const bfloat* scales [[buffer(1)]],
                                 device const bfloat* biases [[buffer(2)]],
                                 device bfloat*       out    [[buffer(3)]],
                                 constant uint&       K      [[buffer(4)]],
                                 constant uint&       GS     [[buffer(5)]],
                                 constant uint&       N      [[buffer(6)]],
                                 uint gid [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    const uint idx = (gid * 256u + tid) * 128u;
    if (idx >= N * K) {
        return;
    }
    const uint words = K / 8u;
    const uint groups = K / GS;
    const uint row = idx / K;
    const uint col = idx % K;
    const uint g = col / GS;
    device const uint4* src =
        (device const uint4*)(codes + (ulong)row * words + col / 8u);
    device bfloat4* dst = (device bfloat4*)(out + idx);
    const float s0 = float(scales[row * groups + g]);
    const float b0 = float(biases[row * groups + g]);
    const float s1 = float(scales[row * groups + g + 1]);
    const float b1 = float(biases[row * groups + g + 1]);
    const uint4 w0 = src[0];
    const uint4 w1 = src[1];
    const uint4 w2 = src[2];
    const uint4 w3 = src[3];
    store_word_q4(w0.x, s0, b0, dst, 0);
    store_word_q4(w0.y, s0, b0, dst, 1);
    store_word_q4(w0.z, s0, b0, dst, 2);
    store_word_q4(w0.w, s0, b0, dst, 3);
    store_word_q4(w1.x, s0, b0, dst, 4);
    store_word_q4(w1.y, s0, b0, dst, 5);
    store_word_q4(w1.z, s0, b0, dst, 6);
    store_word_q4(w1.w, s0, b0, dst, 7);
    store_word_q4(w2.x, s1, b1, dst, 8);
    store_word_q4(w2.y, s1, b1, dst, 9);
    store_word_q4(w2.z, s1, b1, dst, 10);
    store_word_q4(w2.w, s1, b1, dst, 11);
    store_word_q4(w3.x, s1, b1, dst, 12);
    store_word_q4(w3.y, s1, b1, dst, 13);
    store_word_q4(w3.z, s1, b1, dst, 14);
    store_word_q4(w3.w, s1, b1, dst, 15);
}
