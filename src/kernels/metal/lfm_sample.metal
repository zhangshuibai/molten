// LFM2.5 decode sampling: repetition penalty, temperature, top-k, top-p and a
// counter-based random draw, all on the GPU (dispatch wrappers and the
// algorithm overview are in src/lfm/sample.rs). Written for this engine; the
// greedy kernels in lfm.metal are not used or changed by this file.
//
// Processed logits. w_i = pen(v_i) / T with v_i = float(bfloat logit), the
// value greedy decoding compares, and pen(v) = v / p (v > 0) or v * p
// (v <= 0) for the tokens of the sequence's seen set. -0 is folded to +0.
//
// Draw. The sampled token is argmax over the kept set of (w_i + G_i), where
// G_i is a Gumbel(0, 1) variate (the Gumbel-max trick): that is an exact draw
// from softmax(w) renormalized over the kept set. G_i comes from
// Philox4x32-10 (Salmon et al., SC 2011) with key = seed and counter =
// (i / 4, position of the sampled token, "SAMP", 0); word i % 4 is token i's
// 32-bit uniform. The draw depends only on (seed, position, logits, kept
// set), never on how the vocabulary is split across threadgroups or how a
// generation is split into requests. Ties go to the lower token id, as in
// the greedy argmax.
//
// Kept set. Always {w >= tau}: top-k keeps every token whose w is at least
// the k-th largest (ties at the boundary kept, as Hugging Face and vLLM do);
// top-p then keeps, among those, the smallest such set whose probability is
// at least top_p (a tie group straddling the boundary is kept whole).
// lfm_sample_prep, lfm_sample_levels, lfm_sample_gather and
// lfm_sample_finish find tau exactly; every sum in that decision is either
// integer (counts, fixed-point masses) or a float sum in a fixed order, so the
// result is bitwise deterministic.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint SMP_V = 128000;            // vocabulary
constant constexpr uint SMP_BLOCK = 1024;          // tokens per wide threadgroup
constant constexpr uint SMP_BLOCKS = SMP_V / SMP_BLOCK;
constant constexpr uint SMP_TG = 256;              // threads per wide threadgroup (4 tokens each)
constant constexpr uint SMP_SG = SMP_TG / 32;
constant constexpr uint SMP_LEVELS = 32;           // cutoffs per levels round
constant constexpr uint SMP_LTG = 64;              // threads per levels threadgroup
constant constexpr uint SMP_LSG = SMP_LTG / 32;
constant constexpr uint SMP_LV4 = SMP_BLOCK / 4u / SMP_LTG;   // float4 per levels thread
constant constexpr uint SMP_CAP = 2048;            // band sorted in threadgroup memory
constant constexpr uint SMP_REFINE = 256;          // a larger band gets another levels round
constant constexpr uint SMP_FTG = 256;             // threads of lfm_sample_finish
constant constexpr uint SMP_FSG = SMP_FTG / 32;
constant constexpr uint SMP_RADIX = 16;            // digits per narrowing round of lfm_sample_finish
constant constexpr uint SMP_DOMAIN = 0x53414D50u;  // "SAMP", Philox counter word 2

static_assert(SMP_BLOCKS * SMP_BLOCK == SMP_V, "the vocabulary is whole blocks");

// Byte layout shared with Sampling::params in src/lfm/sample.rs.
struct SmpParams {
    float inv_temp;      // 1 / temperature
    float penalty;       // repetition penalty p
    float inv_penalty;   // 1 / p
    uint  use_penalty;   // p != 1
    uint  top_k;         // 0 = off (the host maps k >= vocab to 0)
    float top_p;         // >= 1 = off
    uint  seed_lo;
    uint  seed_hi;
    uint  pos;           // absolute position of the sampled token
    uint  pad0;
    uint  pad1;
    uint  pad2;
    float delta[SMP_LEVELS];   // levels round 0: w cutoffs M - delta[j] (delta[0] = 0)
};

// Written by the levels passes, read by gather and finish.
struct SmpPlan {
    float M;            // max w
    float lo;           // band: lo <= w (< hi when has_above)
    float hi;
    uint  has_above;    // tokens with w >= hi are kept outright (band mode)
    uint  list;         // 1: top-k then top-p over every token with w >= lo
    uint  above_cnt;    // count and mass of the tokens with w >= hi (0 without hi)
    float above_mass;
    float Z;            // sum of exp(w - M) over the vocabulary
    uint  band_cnt;     // tokens with lo <= w < hi
    uint  refine;       // the next levels round subdivides [lo, hi)
    uint  has_hi;       // hi is a bound (else the band reaches the max)
    uint  pad[5];
};

// ------------------------------------------------------------------ helpers

// Orderable 32-bit key of a float: a larger key is a larger value; -0 maps to +0.
static inline uint smp_fkey(float v) {
    uint u = as_type<uint>(v);
    if ((u << 1) == 0u) {
        u = 0u;
    }
    return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}

static inline float smp_kfloat(uint k) {
    const uint u = (k & 0x80000000u) != 0u ? (k & 0x7FFFFFFFu) : ~k;
    return as_type<float>(u);
}

// The greedy head's 64-bit argmax key: value desc, then token id asc.
// lfm_token_tail reads the token from it unchanged.
static inline ulong smp_key(float v, uint i) {
    return (ulong(smp_fkey(v)) << 32) | ulong(~i);
}

static inline ulong smp_simd_max_key(ulong k) {
    for (ushort off = 16; off > 0; off >>= 1) {
        const uint hi = simd_shuffle_down(uint(k >> 32), off);
        const uint lo = simd_shuffle_down(uint(k & 0xFFFFFFFFul), off);
        k = max(k, (ulong(hi) << 32) | ulong(lo));
    }
    return k;  // lane 0 holds the simdgroup maximum
}

static inline float smp_w(float v, bool seen, constant SmpParams& p) {
    if (seen) {
        v = v > 0.0f ? v * p.inv_penalty : v * p.penalty;
    }
    uint u = as_type<uint>(v * p.inv_temp);
    if ((u << 1) == 0u) {
        u = 0u;
    }
    return as_type<float>(u);
}

static inline uint4 smp_philox(uint4 c, uint2 k) {
    for (uint r = 0; r < 10u; ++r) {
        const uint hi0 = mulhi(0xD2511F53u, c.x);
        const uint lo0 = 0xD2511F53u * c.x;
        const uint hi1 = mulhi(0xCD9E8D57u, c.z);
        const uint lo1 = 0xCD9E8D57u * c.z;
        c = uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
        k += uint2(0x9E3779B9u, 0xBB67AE85u);
    }
    return c;
}

// Uniform words of tokens 4q .. 4q + 3.
static inline uint4 smp_random4(uint q, constant SmpParams& p) {
    return smp_philox(uint4(q, p.pos, SMP_DOMAIN, 0u), uint2(p.seed_lo, p.seed_hi));
}

// Gumbel(0, 1) from a 32-bit uniform word: G = -log(E) with E = -log(1 - V)
// an Exp(1) variate, V = (x + 0.5) / 2^32. Small E (large G, the draws that
// win) comes from the series of -log1p(-V), so it keeps full relative
// precision down to V = 2^-33.
static inline float smp_gumbel(uint x) {
    const float v = (float(x) + 0.5f) * 2.3283064365386963e-10f;
    float e;
    if (v < 0.015625f) {
        e = v * (1.0f + v * (0.5f + v * (0.33333334f + v * 0.25f)));
    } else {
        e = -log(max(1.0f - v, 2.9802322387695312e-08f));
    }
    return -log(e);
}

// ------------------------------------------------- temperature-only draw

// No top-k / top-p: one pass over the logits draws the token. Each
// threadgroup folds the best (w + G, id) of its 1024 tokens into the same
// 64-bit key the greedy head uses; lfm_token_tail then reads and clears it.
[[kernel, max_total_threads_per_threadgroup(SMP_TG)]] void lfm_sample_gumbel(device const bfloat* logits [[buffer(0)]],
                              device const uint*   seen   [[buffer(1)]],
                              device atomic_ulong* best   [[buffer(2)]],
                              constant SmpParams&  p      [[buffer(3)]],
                              uint tg   [[threadgroup_position_in_grid]],
                              uint tid  [[thread_index_in_threadgroup]],
                              uint sg   [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup ulong part[SMP_SG];
    const uint q = tg * (SMP_BLOCK / 4u) + tid;
    const uint i0 = 4u * q;
    const float4 v = float4(((device const bfloat4*)logits)[q]);
    const uint bits = p.use_penalty != 0u ? (seen[i0 >> 5] >> (i0 & 31u)) : 0u;
    const uint4 r = smp_random4(q, p);
    ulong k = smp_key(smp_w(v.x, (bits & 1u) != 0u, p) + smp_gumbel(r.x), i0);
    k = max(k, smp_key(smp_w(v.y, (bits & 2u) != 0u, p) + smp_gumbel(r.y), i0 + 1u));
    k = max(k, smp_key(smp_w(v.z, (bits & 4u) != 0u, p) + smp_gumbel(r.z), i0 + 2u));
    k = max(k, smp_key(smp_w(v.w, (bits & 8u) != 0u, p) + smp_gumbel(r.w), i0 + 3u));
    k = smp_simd_max_key(k);
    if (lane == 0) {
        part[sg] = k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint s = 1; s < SMP_SG; ++s) {
            k = max(k, part[s]);
        }
        atomic_max_explicit(best, k, memory_order_relaxed);
    }
}

// ------------------------------------------------------ top-k / top-p path

// Pass 1: processed logits to F32, and the vocabulary's max and min w
// (stat[0] = max key, stat[1] = ~min key; both zero before the dispatch).
[[kernel, max_total_threads_per_threadgroup(SMP_TG)]] void lfm_sample_prep(device const bfloat* logits [[buffer(0)]],
                            device const uint*   seen   [[buffer(1)]],
                            device float4*       w      [[buffer(2)]],
                            device atomic_uint*  stat   [[buffer(3)]],
                            constant SmpParams&  p      [[buffer(4)]],
                            uint tg   [[threadgroup_position_in_grid]],
                            uint tid  [[thread_index_in_threadgroup]],
                            uint sg   [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint pmax[SMP_SG];
    threadgroup uint pmin[SMP_SG];
    const uint q = tg * (SMP_BLOCK / 4u) + tid;
    const uint i0 = 4u * q;
    const float4 v = float4(((device const bfloat4*)logits)[q]);
    const uint bits = p.use_penalty != 0u ? (seen[i0 >> 5] >> (i0 & 31u)) : 0u;
    const float4 x = float4(smp_w(v.x, (bits & 1u) != 0u, p), smp_w(v.y, (bits & 2u) != 0u, p),
                            smp_w(v.z, (bits & 4u) != 0u, p), smp_w(v.w, (bits & 8u) != 0u, p));
    w[q] = x;
    const uint k0 = smp_fkey(x.x), k1 = smp_fkey(x.y), k2 = smp_fkey(x.z), k3 = smp_fkey(x.w);
    uint kmax = simd_max(max(max(k0, k1), max(k2, k3)));
    uint kmin = simd_min(min(min(k0, k1), min(k2, k3)));
    if (lane == 0) {
        pmax[sg] = kmax;
        pmin[sg] = kmin;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint s = 1; s < SMP_SG; ++s) {
            kmax = max(kmax, pmax[s]);
            kmin = min(kmin, pmin[s]);
        }
        atomic_fetch_max_explicit(&stat[0], kmax, memory_order_relaxed);
        atomic_fetch_max_explicit(&stat[1], ~kmin, memory_order_relaxed);
    }
}

// Cutoff j of a levels round. Round 0: M - delta[j], and the vocabulary's
// min last (its level holds every token). Later rounds: 32 even steps down
// [lo, hi) of the band the previous round chose.
static inline float smp_cut(uint round, uint j, float M, float wmin, float lo, float hi,
                            constant SmpParams& p) {
    if (round == 0u) {
        return j + 1u < SMP_LEVELS ? max(M - p.delta[j], wmin) : wmin;
    }
    return j + 1u < SMP_LEVELS ? max(hi - (hi - lo) * (float(j + 1u) / float(SMP_LEVELS)), lo) : lo;
}

// Fixed-point scale of the level masses: a threadgroup's mass (at most 1024)
// times 2^40 stays below 2^50, and 125 of them below 2^57.
constant constexpr float SMP_FIX = 1099511627776.0f;      // 2^40
constant constexpr float SMP_UNFIX = 9.094947017729282e-13f; // 2^-40

// Adds v to the 64-bit accumulator acc[0] (low word) / acc[1] (high word).
// Integer addition commutes, so the total is independent of the order in
// which threadgroups add (each carry out of the low word is counted once).
static inline void smp_add64(device atomic_uint* acc, ulong v) {
    const uint lo = uint(v & 0xFFFFFFFFul);
    const uint hi = uint(v >> 32);
    const uint old = atomic_fetch_add_explicit(&acc[0], lo, memory_order_relaxed);
    const uint carry = old + lo < old ? 1u : 0u;
    if (hi + carry != 0u) {
        atomic_fetch_add_explicit(&acc[1], hi + carry, memory_order_relaxed);
    }
}

// Pass 2 (round 0, then refinement rounds): count and probability mass
// (exp(w - M), unnormalized) of the tokens with w >= theta_j at 32 cutoffs.
// Each threadgroup reduces its 1024 tokens in a fixed order and adds the
// result to integer accumulators (counts; masses in 2^-40 fixed point), so the
// totals are exact and order-independent. The threadgroup that finishes last
// (atomic counter stat[2], device-scope fences, nobody waits) reads them and
// updates the plan: the band [theta_j, theta_{j-1}) that holds the threshold,
// the totals above it, and whether the band still needs a refinement round
// (more than SMP_REFINE tokens to gather). A refinement round whose plan does not
// ask for one returns at once. Masses are only accumulated for a top-p target
// (top_k == 0). Round 0 also clears stat[0..1]; every round leaves stat[2]
// and the accumulators zero.
[[kernel, max_total_threads_per_threadgroup(SMP_LTG)]] void lfm_sample_levels(device const float4* w     [[buffer(0)]],
                              device atomic_uint*  stat  [[buffer(1)]],
                              device atomic_uint*  acc   [[buffer(2)]],
                              device SmpPlan*      plan  [[buffer(3)]],
                              constant SmpParams&  p     [[buffer(4)]],
                              constant uint&       round [[buffer(5)]],
                              uint tg   [[threadgroup_position_in_grid]],
                              uint tid  [[thread_index_in_threadgroup]],
                              uint sg   [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint  sc[SMP_LSG][SMP_LEVELS];
    threadgroup float sm[SMP_LSG][SMP_LEVELS];
    threadgroup uint  gc[SMP_LEVELS];
    threadgroup float gm[SMP_LEVELS];
    threadgroup uint  is_last[1];
    // acc layout: [0, 32) counts; [32 + 2 j, 32 + 2 j + 2) level j's mass.
    device atomic_uint* acnt = acc;
    device atomic_uint* amass = acc + SMP_LEVELS;
    SmpPlan prev;
    float M, wmin = 0.0f, lo = 0.0f, hi = 0.0f;
    if (round == 0u) {
        M = smp_kfloat(atomic_load_explicit(&stat[0], memory_order_relaxed));
        wmin = smp_kfloat(~atomic_load_explicit(&stat[1], memory_order_relaxed));
    } else {
        prev = plan[0];
        if (prev.refine == 0u) {
            return;
        }
        M = prev.M;
        lo = prev.lo;
        hi = prev.hi;
    }
    const bool need_mass = p.top_k == 0u;
    float4 x[SMP_LV4];
    float4 e[SMP_LV4];
    for (uint r = 0; r < SMP_LV4; ++r) {
        x[r] = w[tg * (SMP_BLOCK / 4u) + tid + SMP_LTG * r];
        e[r] = need_mass ? exp(x[r] - M) : float4(0.0f);
    }
    for (uint j = 0; j < SMP_LEVELS; ++j) {
        const float th = smp_cut(round, j, M, wmin, lo, hi, p);
        uint c = 0u;
        float m = 0.0f;
        for (uint r = 0; r < SMP_LV4; ++r) {
            const bool4 ge = x[r] >= th;
            c += uint(ge.x) + uint(ge.y) + uint(ge.z) + uint(ge.w);
            if (need_mass) {
                m += (ge.x ? e[r].x : 0.0f) + (ge.y ? e[r].y : 0.0f) + (ge.z ? e[r].z : 0.0f)
                    + (ge.w ? e[r].w : 0.0f);
            }
        }
        const uint cs = simd_sum(c);
        const float ms = need_mass ? simd_sum(m) : 0.0f;
        if (lane == 0) {
            sc[sg][j] = cs;
            sm[sg][j] = ms;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < SMP_LEVELS) {
        uint c = 0u;
        float m = 0.0f;
        for (uint s2 = 0; s2 < SMP_LSG; ++s2) {
            c += sc[s2][tid];
            m += sm[s2][tid];
        }
        atomic_fetch_add_explicit(&acnt[tid], c, memory_order_relaxed);
        if (need_mass) {
            smp_add64(amass + 2u * tid, ulong(m * SMP_FIX));
        }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                            thread_scope::thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                            thread_scope::thread_scope_device);
        const uint done = atomic_fetch_add_explicit(&stat[2], 1u, memory_order_relaxed);
        is_last[0] = done == SMP_BLOCKS - 1u ? 1u : 0u;
        if (is_last[0] != 0u) {
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                                thread_scope::thread_scope_device);
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (is_last[0] == 0u) {
        return;
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                        thread_scope::thread_scope_device);
    if (tid < SMP_LEVELS) {
        gc[tid] = atomic_load_explicit(&acnt[tid], memory_order_relaxed);
        const uint mlo = atomic_load_explicit(&amass[2u * tid], memory_order_relaxed);
        const uint mhi = atomic_load_explicit(&amass[2u * tid + 1u], memory_order_relaxed);
        gm[tid] = float((ulong(mhi) << 32) | ulong(mlo)) * SMP_UNFIX;
        atomic_store_explicit(&acnt[tid], 0u, memory_order_relaxed);
        atomic_store_explicit(&amass[2u * tid], 0u, memory_order_relaxed);
        atomic_store_explicit(&amass[2u * tid + 1u], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const float Z = round == 0u ? gm[SMP_LEVELS - 1u] : prev.Z;
        uint js = SMP_LEVELS - 1u;
        if (p.top_k != 0u) {
            for (uint j = 0; j < SMP_LEVELS; ++j) {
                if (gc[j] >= p.top_k) {
                    js = j;
                    break;
                }
            }
        } else {
            const float t = p.top_p * Z;
            for (uint j = 0; j < SMP_LEVELS; ++j) {
                if (gm[j] >= t) {
                    js = j;
                    break;
                }
            }
        }
        SmpPlan pl;
        pl.M = M;
        pl.Z = Z;
        pl.lo = smp_cut(round, js, M, wmin, lo, hi, p);
        const bool first = round == 0u;
        pl.hi = js > 0u ? smp_cut(round, js - 1u, M, wmin, lo, hi, p) : (first ? M : prev.hi);
        pl.has_hi = (js > 0u || (!first && prev.has_hi != 0u)) ? 1u : 0u;
        pl.above_cnt = js > 0u ? gc[js - 1u] : (first ? 0u : prev.above_cnt);
        pl.above_mass = js > 0u ? gm[js - 1u] : (first ? 0.0f : prev.above_mass);
        pl.band_cnt = gc[js] - pl.above_cnt;
        pl.list = (p.top_k != 0u && p.top_p < 1.0f) ? 1u : 0u;
        pl.has_above = (pl.list == 0u && pl.has_hi != 0u) ? 1u : 0u;
        const uint gathered = pl.list != 0u ? gc[js] : pl.band_cnt;
        pl.refine = (gathered > SMP_REFINE && pl.lo < pl.hi) ? 1u : 0u;
        for (uint i = 0; i < 5u; ++i) {
            pl.pad[i] = 0u;
        }
        plan[0] = pl;
        if (first) {
            atomic_store_explicit(&stat[0], 0u, memory_order_relaxed);
            atomic_store_explicit(&stat[1], 0u, memory_order_relaxed);
        }
        atomic_store_explicit(&stat[2], 0u, memory_order_relaxed);
    }
}

// Pass 3: copies the band's tokens (or, in list mode, every token with
// w >= lo) to the array, block b's in token order at [b * 1024, b * 1024 +
// acnt[b]), and folds the Gumbel draw of the tokens above the band into
// `best`.
[[kernel, max_total_threads_per_threadgroup(SMP_TG)]] void lfm_sample_gather(device const float4*  w    [[buffer(0)]],
                              device const SmpPlan* plan [[buffer(1)]],
                              device float*         aw   [[buffer(2)]],
                              device uint*          ai   [[buffer(3)]],
                              device uint*          acnt [[buffer(4)]],
                              device atomic_ulong*  best [[buffer(5)]],
                              constant SmpParams&   p    [[buffer(6)]],
                              uint tg   [[threadgroup_position_in_grid]],
                              uint tid  [[thread_index_in_threadgroup]],
                              uint sg   [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint  scnt[SMP_SG];
    threadgroup ulong sbest[SMP_SG];
    const SmpPlan pl = plan[0];
    const uint q = tg * (SMP_BLOCK / 4u) + tid;
    const uint i0 = 4u * q;
    const float4 x = w[q];
    const bool above_on = pl.has_above != 0u;
    bool4 abv;
    bool4 inb;
    for (uint c = 0; c < 4u; ++c) {
        abv[c] = above_on && x[c] >= pl.hi;
        inb[c] = x[c] >= pl.lo && !abv[c];
    }
    const uint n = uint(inb.x) + uint(inb.y) + uint(inb.z) + uint(inb.w);
    const uint pre = simd_prefix_exclusive_sum(n);
    if (lane == 31) {
        scnt[sg] = pre + n;
    }
    ulong k = 0ul;
    if (any(abv)) {
        const uint4 r = smp_random4(q, p);
        if (abv.x) {
            k = max(k, smp_key(x.x + smp_gumbel(r.x), i0));
        }
        if (abv.y) {
            k = max(k, smp_key(x.y + smp_gumbel(r.y), i0 + 1u));
        }
        if (abv.z) {
            k = max(k, smp_key(x.z + smp_gumbel(r.z), i0 + 2u));
        }
        if (abv.w) {
            k = max(k, smp_key(x.w + smp_gumbel(r.w), i0 + 3u));
        }
    }
    k = smp_simd_max_key(k);
    if (lane == 0) {
        sbest[sg] = k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = 0u;
    for (uint s = 0; s < sg; ++s) {
        base += scnt[s];
    }
    uint o = tg * SMP_BLOCK + base + pre;
    if (inb.x) { aw[o] = x.x; ai[o] = i0;      ++o; }
    if (inb.y) { aw[o] = x.y; ai[o] = i0 + 1u; ++o; }
    if (inb.z) { aw[o] = x.z; ai[o] = i0 + 2u; ++o; }
    if (inb.w) { aw[o] = x.w; ai[o] = i0 + 3u; }
    if (tid == 0) {
        uint total = 0u;
        ulong b = 0ul;
        for (uint s = 0; s < SMP_SG; ++s) {
            total += scnt[s];
            b = max(b, sbest[s]);
        }
        acnt[tg] = total;
        if (b != 0ul) {
            atomic_max_explicit(best, b, memory_order_relaxed);
        }
    }
}

// ------------------------------------------------------------ finish pass

// Threadgroup state of lfm_sample_finish.
struct SmpFinish {
    uint  off[SMP_BLOCKS + 1];   // array element g lives in block b with off[b] <= g < off[b + 1]
    uint  ured[SMP_FSG][2 * SMP_RADIX];
    float fred[SMP_FSG][SMP_RADIX];
    uint  bcast[4];
    float fcast[2];
};

static inline uint smp_block_of(threadgroup const SmpFinish& f, uint g) {
    uint lo = 0u, hi = SMP_BLOCKS;  // off[lo] <= g < off[hi]
    while (hi - lo > 1u) {
        const uint mid = (lo + hi) >> 1;
        if (f.off[mid] <= g) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static inline void smp_elem(threadgroup const SmpFinish& f, device const float* aw,
                            device const uint* ai, uint g, thread float& v, thread uint& id) {
    const uint b = smp_block_of(f, g);
    const uint o = b * SMP_BLOCK + (g - f.off[b]);
    v = aw[o];
    id = ai[o];
}

static inline float smp_tg_sum_f(threadgroup SmpFinish& f, float v, uint tid, uint sg, uint lane) {
    v = simd_sum(v);
    if (lane == 0) {
        f.fred[sg][0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (uint s = 0; s < SMP_FSG; ++s) {
            t += f.fred[s][0];
        }
        f.fcast[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float r = f.fcast[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

static inline uint smp_tg_min_u(threadgroup SmpFinish& f, uint v, uint tid, uint sg, uint lane) {
    v = simd_min(v);
    if (lane == 0) {
        f.ured[sg][0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint t = f.ured[0][0];
        for (uint s = 1; s < SMP_FSG; ++s) {
            t = min(t, f.ured[s][0]);
        }
        f.bcast[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint r = f.bcast[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

static inline uint smp_tg_max_u(threadgroup SmpFinish& f, uint v, uint tid, uint sg, uint lane) {
    return ~smp_tg_min_u(f, ~v, tid, sg, lane);
}

// Loads the n <= SMP_CAP array elements whose keys lie in [klo, khi] into
// keys[] as (orderable w << 32 | ~id), in array order, and sorts them in
// descending order (w desc, id asc) with a bitonic network; padding keys are
// 0 and sort last.
static inline void smp_load_sort(threadgroup SmpFinish& f, threadgroup ulong* keys,
                                 device const float* aw, device const uint* ai, uint N,
                                 uint klo, uint khi, uint tid, uint sg, uint lane) {
    uint base = 0u;
    for (uint g0 = 0; g0 < N; g0 += SMP_FTG) {
        const uint g = g0 + tid;
        float v = 0.0f;
        uint id = 0u;
        bool in = false;
        if (g < N) {
            smp_elem(f, aw, ai, g, v, id);
            const uint kv = smp_fkey(v);
            in = kv >= klo && kv <= khi;
        }
        const uint pre = simd_prefix_exclusive_sum(uint(in));
        if (lane == 31) {
            f.ured[sg][0] = pre + uint(in);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint before = 0u, total = 0u;
        for (uint s = 0; s < SMP_FSG; ++s) {
            const uint c = f.ured[s][0];
            before += s < sg ? c : 0u;
            total += c;
        }
        if (in) {
            keys[base + before + pre] = smp_key(v, id);
        }
        base += total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint n2 = 2u;
    while (n2 < base) {
        n2 <<= 1;
    }
    for (uint g = base + tid; g < n2; g += SMP_FTG) {
        keys[g] = 0ul;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2u; k <= n2; k <<= 1) {
        for (uint j = k >> 1; j > 0u; j >>= 1) {
            for (uint t = tid; t < (n2 >> 1); t += SMP_FTG) {
                // Pair t: i has bit j clear, l = i + j (j is a power of two).
                const uint i = ((t & ~(j - 1u)) << 1) | (t & (j - 1u));
                const uint l = i + j;
                const ulong a = keys[i];
                const ulong b = keys[l];
                const bool desc = (i & k) == 0u;
                if (desc ? (a < b) : (a > b)) {
                    keys[i] = b;
                    keys[l] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Inclusive prefix sums of exp(w - M) over the n sorted keys, into scan[].
static inline void smp_scan(threadgroup SmpFinish& f, threadgroup const ulong* keys,
                            threadgroup float* scan, uint n, float M,
                            uint tid, uint sg, uint lane) {
    constexpr uint PER = SMP_CAP / SMP_FTG;
    float loc[PER];
    float run = 0.0f;
    for (uint r = 0; r < PER; ++r) {
        const uint g = tid * PER + r;
        const float e = g < n ? exp(smp_kfloat(uint(keys[g] >> 32)) - M) : 0.0f;
        run += e;
        loc[r] = run;
    }
    const float pre = simd_prefix_exclusive_sum(run);
    if (lane == 31) {
        f.fred[sg][0] = pre + run;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float t = lane < SMP_FSG ? f.fred[lane][0] : 0.0f;
        const float ex = simd_prefix_exclusive_sum(t);
        if (lane < SMP_FSG) {
            f.fred[lane][1] = ex;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float off = f.fred[sg][1] + pre;
    for (uint r = 0; r < PER; ++r) {
        scan[tid * PER + r] = off + loc[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// First sorted index whose cumulative mass base + scan[g] reaches t (n - 1
// if none does).
static inline uint smp_first_mass(threadgroup SmpFinish& f, threadgroup const float* scan,
                                  uint n, float base, float t, uint tid, uint sg, uint lane) {
    constexpr uint PER = SMP_CAP / SMP_FTG;
    uint first = 0xFFFFFFFFu;
    for (uint r = 0; r < PER; ++r) {
        const uint g = tid * PER + r;
        if (g < n && base + scan[g] >= t) {
            first = min(first, g);
        }
    }
    first = smp_tg_min_u(f, first, tid, sg, lane);
    return first == 0xFFFFFFFFu ? n - 1u : first;
}

// Exact selection over the array (N elements, which may exceed SMP_CAP):
// the largest value tau such that base + stat({w >= tau}) >= target, where
// stat is the count (mass == false) or the mass exp(w - M). `base` holds the
// tokens kept outright above the array. Radix rounds on the orderable keys
// (4 bits each) narrow the band holding tau until it fits SMP_CAP or holds a
// single value; the band is then sorted and resolved exactly.
static inline float smp_select(threadgroup SmpFinish& f, threadgroup ulong* keys,
                               threadgroup float* scan, device const float* aw,
                               device const uint* ai, uint N, float M, bool mass,
                               float target, uint base_cnt, float base_mass,
                               uint tid, uint sg, uint lane) {
    uint klo = 0u, khi = 0xFFFFFFFFu;
    uint nband = N;
    if (N > SMP_CAP) {
        uint kmin = 0xFFFFFFFFu, kmax = 0u;
        for (uint g = tid; g < N; g += SMP_FTG) {
            float v;
            uint id;
            smp_elem(f, aw, ai, g, v, id);
            const uint kv = smp_fkey(v);
            kmin = min(kmin, kv);
            kmax = max(kmax, kv);
        }
        klo = smp_tg_min_u(f, kmin, tid, sg, lane);
        khi = smp_tg_max_u(f, kmax, tid, sg, lane);
    }
    while (nband > SMP_CAP && klo != khi) {
        const uint h = 31u - clz(klo ^ khi);
        const uint shift = (h / 4u) * 4u;
        const ulong span = 1ul << (shift + 4u);
        const ulong pref = ulong(khi) & ~(span - 1ul);
        uint cnt[SMP_RADIX];
        float ms[SMP_RADIX];
        for (uint d = 0; d < SMP_RADIX; ++d) {
            cnt[d] = 0u;
            ms[d] = 0.0f;
        }
        for (uint g = tid; g < N; g += SMP_FTG) {
            float v;
            uint id;
            smp_elem(f, aw, ai, g, v, id);
            const uint kv = smp_fkey(v);
            if (kv >= klo && kv <= khi) {
                const float e = exp(v - M);
                for (uint d = 0; d < SMP_RADIX; ++d) {
                    // cutoff d: digit >= 15 - d
                    const ulong cut = pref | (ulong(SMP_RADIX - 1u - d) << shift);
                    const bool ge = ulong(kv) >= cut;
                    cnt[d] += uint(ge);
                    ms[d] += ge ? e : 0.0f;
                }
            }
        }
        for (uint d = 0; d < SMP_RADIX; ++d) {
            const uint c = simd_sum(cnt[d]);
            const float m = simd_sum(ms[d]);
            if (lane == 0) {
                f.ured[sg][d] = c;
                f.fred[sg][d] = m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < SMP_RADIX) {
            uint c = 0u;
            float m = 0.0f;
            for (uint s = 0; s < SMP_FSG; ++s) {
                c += f.ured[s][tid];
                m += f.fred[s][tid];
            }
            f.ured[0][SMP_RADIX + tid] = c;
            scan[tid] = m;  // scratch: the band is not sorted yet
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint dstar = SMP_RADIX;
            for (uint d = 0; d < SMP_RADIX; ++d) {
                const bool hit = mass ? (base_mass + scan[d] >= target)
                                      : (float(base_cnt + f.ured[0][SMP_RADIX + d]) >= target);
                if (hit) {
                    dstar = d;
                    break;
                }
            }
            if (dstar == SMP_RADIX) {
                // Rounding left the whole band short of the target: keep it
                // all (tau = its lower bound).
                f.bcast[0] = klo;
                f.bcast[1] = klo;
                f.bcast[2] = nband;
                f.bcast[3] = base_cnt;
                f.fcast[0] = base_mass;
            } else {
                const uint above_c = dstar > 0u ? f.ured[0][SMP_RADIX + dstar - 1u] : 0u;
                const float above_m = dstar > 0u ? scan[dstar - 1u] : 0.0f;
                const ulong dlo = pref | (ulong(SMP_RADIX - 1u - dstar) << shift);
                const ulong dhi = dlo + (1ul << shift) - 1ul;
                f.bcast[0] = max(klo, uint(dlo));
                f.bcast[1] = min(khi, uint(dhi));
                f.bcast[2] = f.ured[0][SMP_RADIX + dstar] - above_c;
                f.bcast[3] = base_cnt + above_c;
                f.fcast[0] = base_mass + above_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        klo = f.bcast[0];
        khi = f.bcast[1];
        nband = f.bcast[2];
        base_cnt = f.bcast[3];
        base_mass = f.fcast[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (klo == khi) {
        return smp_kfloat(klo);  // a single value: the whole band is kept
    }
    smp_load_sort(f, keys, aw, ai, N, klo, khi, tid, sg, lane);
    uint idx;
    if (mass) {
        smp_scan(f, keys, scan, nband, M, tid, sg, lane);
        idx = smp_first_mass(f, scan, nband, base_mass, target, tid, sg, lane);
    } else {
        const float need = target - float(base_cnt);  // 1 <= need <= nband
        idx = uint(clamp(need, 1.0f, float(nband))) - 1u;
    }
    return smp_kfloat(uint(keys[idx] >> 32));
}

// Pass 4 (one threadgroup): resolves the threshold tau inside the band (or,
// in list mode, top-k then top-p over the gathered tokens) and folds the
// Gumbel draw of the kept array tokens into `best`, which already holds the
// draw of the tokens above the band.
[[kernel, max_total_threads_per_threadgroup(SMP_FTG)]] void lfm_sample_finish(device const SmpPlan* plan [[buffer(0)]],
                              device const float*   aw   [[buffer(1)]],
                              device const uint*    ai   [[buffer(2)]],
                              device const uint*    acnt [[buffer(3)]],
                              device atomic_ulong*  best [[buffer(4)]],
                              constant SmpParams&   p    [[buffer(5)]],
                              uint tid  [[thread_index_in_threadgroup]],
                              uint sg   [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup ulong keys[SMP_CAP];
    threadgroup float scan[SMP_CAP];
    threadgroup SmpFinish f;
    const SmpPlan pl = plan[0];
    // Block offsets of the array.
    {
        const uint c = tid < SMP_BLOCKS ? acnt[tid] : 0u;
        const uint pre = simd_prefix_exclusive_sum(c);
        if (lane == 31) {
            f.ured[sg][0] = pre + c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint before = 0u;
        for (uint s = 0; s < sg; ++s) {
            before += f.ured[s][0];
        }
        if (tid <= SMP_BLOCKS) {
            f.off[tid] = before + pre;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint N = f.off[SMP_BLOCKS];
    float tau;
    if (N == 0u) {
        tau = pl.hi;  // nothing below the tokens kept outright (not reached: the band holds tau)
    } else if (pl.list == 0u) {
        const bool mass = p.top_k == 0u;
        const float target = mass ? p.top_p * pl.Z : float(p.top_k);
        tau = smp_select(f, keys, scan, aw, ai, N, pl.M, mass, target,
                         pl.above_cnt, pl.above_mass, tid, sg, lane);
    } else if (N <= SMP_CAP) {
        // Top-k then top-p on one sorted list of every token with w >= lo.
        smp_load_sort(f, keys, aw, ai, N, 0u, 0xFFFFFFFFu, tid, sg, lane);
        smp_scan(f, keys, scan, N, pl.M, tid, sg, lane);
        const uint kk = min(p.top_k, N) - 1u;
        const float tau_k = smp_kfloat(uint(keys[kk] >> 32));
        // Last index of tau_k's tie group: the first index with w < tau_k, minus one.
        constexpr uint PER = SMP_CAP / SMP_FTG;
        uint after = 0xFFFFFFFFu;
        for (uint r = 0; r < PER; ++r) {
            const uint g = tid * PER + r;
            if (g < N && smp_kfloat(uint(keys[g] >> 32)) < tau_k) {
                after = min(after, g);
            }
        }
        after = smp_tg_min_u(f, after, tid, sg, lane);
        const uint lastk = after == 0xFFFFFFFFu ? N - 1u : after - 1u;
        const float zk = scan[lastk];
        const uint ip = smp_first_mass(f, scan, lastk + 1u, 0.0f, p.top_p * zk, tid, sg, lane);
        tau = smp_kfloat(uint(keys[ip] >> 32));
    } else {
        const float tau_k = smp_select(f, keys, scan, aw, ai, N, pl.M, false, float(p.top_k),
                                       0u, 0.0f, tid, sg, lane);
        float zk = 0.0f;
        for (uint g = tid; g < N; g += SMP_FTG) {
            float v;
            uint id;
            smp_elem(f, aw, ai, g, v, id);
            zk += v >= tau_k ? exp(v - pl.M) : 0.0f;
        }
        zk = smp_tg_sum_f(f, zk, tid, sg, lane);
        const float tau_p = smp_select(f, keys, scan, aw, ai, N, pl.M, true, p.top_p * zk,
                                       0u, 0.0f, tid, sg, lane);
        tau = max(tau_k, tau_p);
    }
    // Gumbel draw over the kept array tokens (w >= tau). When the array fits
    // SMP_CAP, keys[0..N) holds all of it (sorted).
    ulong k = 0ul;
    for (uint g = tid; g < N; g += SMP_FTG) {
        float v;
        uint id;
        if (N <= SMP_CAP) {
            const ulong key = keys[g];
            v = smp_kfloat(uint(key >> 32));
            id = ~uint(key & 0xFFFFFFFFul);
        } else {
            smp_elem(f, aw, ai, g, v, id);
        }
        if (v >= tau) {
            const uint4 r = smp_random4(id >> 2, p);
            const uint x = (id & 3u) == 0u ? r.x : (id & 3u) == 1u ? r.y : (id & 3u) == 2u ? r.z : r.w;
            k = max(k, smp_key(v + smp_gumbel(x), id));
        }
    }
    k = smp_simd_max_key(k);
    if (lane == 0) {
        f.ured[sg][0] = uint(k >> 32);
        f.ured[sg][1] = uint(k & 0xFFFFFFFFul);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        ulong b = 0ul;
        for (uint s = 0; s < SMP_FSG; ++s) {
            b = max(b, (ulong(f.ured[s][0]) << 32) | ulong(f.ured[s][1]));
        }
        if (b != 0ul) {
            atomic_max_explicit(best, b, memory_order_relaxed);
        }
    }
}

// After lfm_token_tail: adds the drawn token to the GPU copy of the seen set
// (only dispatched with a repetition penalty).
kernel void lfm_sample_mark(device const uint* tok  [[buffer(0)]],
                            device uint*       seen [[buffer(1)]],
                            uint tid [[thread_index_in_threadgroup]]) {
    if (tid == 0) {
        const uint t = tok[0];
        if (t < SMP_V) {
            seen[t >> 5] |= 1u << (t & 31u);
        }
    }
}
