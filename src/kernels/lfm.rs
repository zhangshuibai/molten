//! Rust dispatch wrappers for the LFM2.5 kernels.

use anyhow::{Result, ensure};

use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};
use crate::weights::QuantWeights;

pub const SOURCE: &str = include_str!("metal/lfm.metal");

pub const ARGMAX_GROUPS: usize = 64;

/// The decode GEMV shapes lfm.metal fixes at compile time (`LFM_HIDDEN`,
/// `LFM_MOE_INTER`, `LFM_GS`, the `lfm_gemv_q4_bf16_k*` entries); the fused
/// decode wrappers check them.
const FIXED_HIDDEN: usize = 2048;
const FIXED_MOE_INTER: usize = 1536;
const FIXED_DENSE_INTER: usize = 11776;
const FIXED_GS: usize = 64;

/// Per-head RMSNorm over `[rows, 64]` BF16 rows with plain gain.
pub fn rmsnorm64(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    w: &Tensor,
    out: &Tensor,
    eps: f32,
) -> Result<()> {
    let rows = x.numel() / 64;
    ensure!(x.numel() % 64 == 0 && out.numel() == x.numel(), "bad rmsnorm64 shape");
    ensure!(w.numel() == 64, "rmsnorm64 weight must be 64");
    for t in [x, w, out] {
        ensure!(t.dtype() == DType::BF16, "rmsnorm64 expects BF16");
    }
    let pipeline = ctx.pipeline("rmsnorm64_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), w.binding(), out.binding()],
        &[&u32_bytes(rows), &eps.to_ne_bytes()],
        Grid::Threadgroups { groups: (rows.div_ceil(4), 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Decode-only: per-head RMSNorm of the query/key rows followed by NeoX RoPE,
/// writing the rotated queries and the rotated keys straight into the cache.
/// Every stored value keeps the bfloat roundings of a separate per-head
/// RMSNorm followed by RoPE.
#[allow(clippy::too_many_arguments)]
pub fn norm_rope_qk_scatter(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q_in: &Tensor,
    q_w: &Tensor,
    q_out: &Tensor,
    k_in: &Tensor,
    k_w: &Tensor,
    k_cache: &Tensor,
    pos: usize,
    theta: f32,
    rows_q: usize,
    rows_k: usize,
) -> Result<()> {
    let d = 64usize;
    ensure!(
        q_in.numel() == rows_q * d
            && q_out.numel() == rows_q * d
            && k_in.numel() == rows_k * d,
        "norm rope qk shapes"
    );
    ensure!(q_w.numel() == d && k_w.numel() == d, "norm rope qk weights");
    ensure!(
        k_cache.shape()[2] == d && k_cache.shape()[0] == rows_k,
        "norm rope qk cache"
    );
    let max_seq = k_cache.shape()[1];
    ensure!(pos < max_seq, "norm rope qk position");
    for t in [q_in, q_w, q_out, k_in, k_w, k_cache] {
        ensure!(t.dtype() == DType::BF16, "norm rope qk BF16");
    }
    let rows = rows_q + rows_k;
    let pipeline =
        ctx.pipeline("lfm_norm_rope_qk_scatter_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            q_in.binding(),
            q_w.binding(),
            q_out.binding(),
            k_in.binding(),
            k_w.binding(),
            k_cache.binding(),
        ],
        &[
            &u32_bytes(rows_q),
            &u32_bytes(rows_k),
            &u32_bytes(d),
            &u32_bytes(pos),
            &theta.to_ne_bytes(),
            &1e-5f32.to_ne_bytes(),
            &u32_bytes(max_seq),
        ],
        Grid::Threadgroups {
            groups: (rows.div_ceil(8), 1, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// One-token affine quantized GEMV (4- or 6-bit), BF16 in and out.
pub fn gemv_quant(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    x: &Tensor,
    y: &Tensor,
) -> Result<()> {
    let (n, k) = (w.out_features(), w.in_features());
    ensure!(x.numel() == k && y.numel() == n, "gemv shape mismatch");
    ensure!(x.dtype() == DType::BF16 && y.dtype() == DType::BF16, "gemv BF16 only");
    if w.bits == 6 && n % 16 == 0 {
        let pipeline = ctx.pipeline("lfm_gemv_q6_bf16_x4", SOURCE, MslVersion::V3_1)?;
        return pass.dispatch_at(
            &pipeline,
            &[w.codes.binding(), w.scales.binding(), w.biases.binding(), x.binding(), y.binding()],
            &[&u32_bytes(k), &u32_bytes(w.group_size)],
            Grid::Threadgroups { groups: (n / 16, 1, 1), threadgroup: (128, 1, 1) },
        );
    }
    // The model's 4-bit/64 shapes have compile-time-K entries (same values);
    // the runtime-shape kernels take K and the group size as arguments.
    let (fn_name, runtime_shape) = match (w.bits, w.group_size, k) {
        (4, FIXED_GS, FIXED_HIDDEN) => ("lfm_gemv_q4_bf16_k2048", false),
        (4, FIXED_GS, FIXED_DENSE_INTER) => ("lfm_gemv_q4_bf16_k11776", false),
        (4, ..) => ("lfm_gemv_q4_bf16", true),
        (6, ..) => ("lfm_gemv_q6_bf16", true),
        (b, ..) => anyhow::bail!("unsupported gemv bits {b}"),
    };
    let pipeline = ctx.pipeline(fn_name, SOURCE, MslVersion::V3_1)?;
    let (kb, gb) = (u32_bytes(k), u32_bytes(w.group_size));
    let shape: &[&[u8]] = if runtime_shape { &[&kb, &gb] } else { &[] };
    pass.dispatch_at(
        &pipeline,
        &[w.codes.binding(), w.scales.binding(), w.biases.binding(), x.binding(), y.binding()],
        shape,
        Grid::Threadgroups { groups: (n, 1, 1), threadgroup: (32, 1, 1) },
    )
}


/// Decode-only: Q, K and V projections in one dispatch, V scattered into the
/// KV cache at `pos` (the separate projection + scatter are pure copies).
#[allow(clippy::too_many_arguments)]
pub fn proj3_gemv_scatter(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &QuantWeights,
    k: &QuantWeights,
    v: &QuantWeights,
    x: &Tensor,
    q_out: &Tensor,
    k_out: &Tensor,
    v_cache: &Tensor,
    pos: usize,
) -> Result<()> {
    let kdim = x.numel();
    let (nq, nk, nv) = (q.out_features(), k.out_features(), v.out_features());
    ensure!(
        q.in_features() == kdim && k.in_features() == kdim && v.in_features() == kdim,
        "proj3 input width"
    );
    ensure!(
        q.bits == 4 && k.bits == 4 && v.bits == 4 && q.group_size == FIXED_GS
            && k.group_size == FIXED_GS && v.group_size == FIXED_GS,
        "proj3 is affine 4-bit/64"
    );
    ensure!(kdim == FIXED_HIDDEN, "proj3 input width is the compile-time hidden size");
    ensure!(
        q_out.numel() == nq && k_out.numel() == nk && nv % 64 == 0,
        "proj3 output shape"
    );
    ensure!(x.dtype() == DType::BF16 && q_out.dtype() == DType::BF16 && k_out.dtype() == DType::BF16, "proj3 BF16");
    let max_seq = v_cache.shape()[1];
    ensure!(
        v_cache.shape()[0] * 64 == nv && v_cache.shape()[2] == 64 && pos < max_seq,
        "proj3 KV cache shape"
    );
    let pipeline = ctx.pipeline("lfm_proj3_gemv_scatter_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            q.codes.binding(),
            q.scales.binding(),
            q.biases.binding(),
            k.codes.binding(),
            k.scales.binding(),
            k.biases.binding(),
            v.codes.binding(),
            v.scales.binding(),
            v.biases.binding(),
            x.binding(),
            q_out.binding(),
            k_out.binding(),
            v_cache.binding(),
        ],
        &[
            &u32_bytes(nq),
            &u32_bytes(nk),
            &u32_bytes(nv),
            &u32_bytes(max_seq),
            &u32_bytes(pos),
        ],
        Grid::Threadgroups { groups: (nq + nk + nv, 1, 1), threadgroup: (32, 1, 1) },
    )
}

/// Decode-only: short-conv B/C/X projection and the conv in one dispatch.
#[allow(clippy::too_many_arguments)]
pub fn conv3_gemv(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    wproj: &QuantWeights,
    x: &Tensor,
    w: &Tensor,
    state: &Tensor,
    out: &Tensor,
    new_state: &Tensor,
) -> Result<()> {
    let kdim = x.numel();
    let h = out.numel();
    ensure!(
        wproj.out_features() == 3 * h && wproj.in_features() == kdim,
        "conv3 projection shape"
    );
    ensure!(wproj.bits == 4 && wproj.group_size == FIXED_GS, "conv3 projection 4-bit/64");
    ensure!(kdim == FIXED_HIDDEN, "conv3 input width is the compile-time hidden size");
    ensure!(w.numel() == 3 * h, "conv3 tap shape");
    ensure!(state.shape() == [2, h] && new_state.shape() == [2, h], "conv3 state shape");
    let pipeline = ctx.pipeline("lfm_conv3_gemv_split_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            wproj.codes.binding(),
            wproj.scales.binding(),
            wproj.biases.binding(),
            x.binding(),
            w.binding(),
            state.binding(),
            out.binding(),
            new_state.binding(),
        ],
        &[&u32_bytes(h)],
        Grid::Threadgroups { groups: (h, 1, 1), threadgroup: (96, 1, 1) },
    )
}

/// Decode-only: residual add + RMSNorm, the F32 router GEMV and
/// `route_token` in one dispatch. `hout` receives the updated BF16 residual;
/// `normed` the normalized row; `logits` the 64 F32 router outputs. The
/// threadgroup that publishes the last router logit runs the warp top-k;
/// `counter` is a zeroed U32 word that the kernel leaves zeroed again. The
/// kernel needs MSL 3.2's device-scope `atomic_thread_fence`, so it comes
/// from the Metal 4 compile of the source.
#[allow(clippy::too_many_arguments)]
pub fn router_addnorm_route(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    h: &Tensor,
    b: &Tensor,
    nw: &Tensor,
    hout: &Tensor,
    normed: &Tensor,
    w: &Tensor,
    logits: &Tensor,
    bias: &Tensor,
    ids: &Tensor,
    weights: &Tensor,
    counter: &Tensor,
    eps: f32,
) -> Result<()> {
    let e = w.shape()[0];
    let k = w.shape()[1];
    ensure!(e == 64 && k == 2048, "fused router route requires E=64, H=2048");
    ensure!(w.dtype() == DType::F32 && logits.numel() == e && logits.dtype() == DType::F32, "router logits shape");
    ensure!(bias.numel() == e && bias.dtype() == DType::F32, "router bias shape");
    ensure!(h.numel() == k && b.numel() == k && hout.numel() == k && normed.numel() == k, "router norm shapes");
    ensure!(nw.numel() == k, "router norm weight");
    ensure!(ids.numel() == 4 && ids.dtype() == DType::U32, "route ids shape");
    ensure!(weights.numel() == 4 && weights.dtype() == DType::BF16, "route weights shape");
    ensure!(counter.numel() == 1 && counter.dtype() == DType::U32, "route counter shape");
    let pipeline = ctx.pipeline("lfm_router_addnorm_route_f32", SOURCE, MslVersion::V4_0)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.binding(),
            h.binding(),
            b.binding(),
            nw.binding(),
            hout.binding(),
            normed.binding(),
            logits.binding(),
            bias.binding(),
            ids.binding(),
            weights.binding(),
            counter.binding(),
        ],
        &[&eps.to_ne_bytes(), &0.0f32.to_ne_bytes(), &(1e-6f32).to_ne_bytes()],
        Grid::Threadgroups { groups: (e, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// F32 dense matrix times BF16 vector, F32 output.
pub fn gemv_f32(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &Tensor,
    x: &Tensor,
    y: &Tensor,
) -> Result<()> {
    let (n, k) = (w.shape()[0], w.shape()[1]);
    ensure!(x.numel() == k && y.numel() == n, "gemv_f32 shape mismatch");
    ensure!(w.dtype() == DType::F32 && x.dtype() == DType::BF16, "gemv_f32 dtypes");
    ensure!(y.dtype() == DType::F32, "gemv_f32 output F32");
    // Staged eight-element loads: the lane's additions stay in increasing k
    // order, so the router logits are bitwise unchanged while the dependent
    // load stalls of the scalar loop disappear.
    let pipeline = ctx.pipeline("lfm_gemv_f32_bf16x_f32out_u8", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[w.binding(), x.binding(), y.binding()],
        &[&u32_bytes(k)],
        Grid::Threadgroups { groups: (n, 1, 1), threadgroup: (32, 1, 1) },
    )
}

/// Dequantizing embedding gather from the 6-bit table.
pub fn gather_rows_q6(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    ids: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let (_, k) = (w.out_features(), w.in_features());
    ensure!(w.bits == 6 && w.group_size == 64, "embedding gather is 6-bit/64");
    let m = ids.numel();
    ensure!(ids.dtype() == DType::U32, "ids must be U32");
    ensure!(out.numel() == m * k && out.dtype() == DType::BF16, "gather out shape");
    let pipeline = ctx.pipeline("lfm_gather_rows_q6_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            ids.binding(),
            out.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(w.group_size)],
        Grid::Threads { grid: (k / 4, m, 1), threadgroup: (32, 1, 1) },
    )
}

/// Router logits for a prompt chunk, F32 `[m, E]`.
pub fn router_prefill(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    w: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let m = x.shape()[0];
    let k = x.shape()[1];
    let e = w.shape()[0];
    ensure!(w.shape() == [e, k] && w.dtype() == DType::F32, "router weight shape");
    ensure!(out.numel() == m * e && out.dtype() == DType::F32, "router out shape");
    // Eight tokens share each threadgroup so each router weight row is read
    // once for all eight; the per-(token, expert) reduction order is unchanged.
    ensure!(e % 4 == 0, "router expert count must be a multiple of four");
    let pipeline = ctx.pipeline("lfm_router_prefill_f32_batch8", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), w.binding(), out.binding()],
        &[&u32_bytes(m), &u32_bytes(k), &u32_bytes(e)],
        Grid::Threadgroups { groups: (m.div_ceil(8), 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Sigmoid top-k routing for one token: expert ids and BF16 combine weights.
pub fn route_token(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    bias: &Tensor,
    ids: &Tensor,
    weights: &Tensor,
) -> Result<()> {
    let e = logits.numel();
    ensure!(bias.numel() == e && e <= 128, "router bias shape");
    let topk = ids.numel();
    ensure!(weights.numel() == topk && topk <= 8, "route outputs");
    // The single-warp form holds experts lane and lane+32, so it covers the
    // model's 64-expert router; other shapes keep the barrier-tree kernel.
    let (name, threads) = if e == 64 {
        ("lfm_route_warp_f32", 32)
    } else {
        ("lfm_route_f32", 128)
    };
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[logits.binding(), bias.binding(), ids.binding(), weights.binding()],
        &[&u32_bytes(e), &u32_bytes(topk), &(1e-6f32).to_ne_bytes()],
        Grid::Threadgroups { groups: (1, 1, 1), threadgroup: (threads, 1, 1) },
    )
}

/// Gate/up expert gather GEMV fused with the SwiGLU activation.
#[allow(clippy::too_many_arguments)]
pub fn gemv_gather_act(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &QuantWeights,
    up: &QuantWeights,
    x: &Tensor,
    ids: &Tensor,
    act: &Tensor,
) -> Result<()> {
    let k = x.numel();
    let n = gate.out_features() / 64;
    let topk = ids.numel();
    ensure!(act.numel() == topk * n, "act shape");
    ensure!(
        gate.bits == 4 && up.bits == 4 && gate.group_size == FIXED_GS && up.group_size == FIXED_GS,
        "gather act is affine 4-bit/64"
    );
    ensure!(
        k == FIXED_HIDDEN && gate.in_features() == k && up.in_features() == k
            && n == FIXED_MOE_INTER && up.out_features() == gate.out_features(),
        "gather act is the compile-time expert shape"
    );
    let pipeline = ctx.pipeline("lfm_gemv_q4_gather_act_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            gate.codes.binding(),
            gate.scales.binding(),
            gate.biases.binding(),
            up.codes.binding(),
            up.scales.binding(),
            up.biases.binding(),
            x.binding(),
            ids.binding(),
            act.binding(),
        ],
        &[],
        Grid::Threadgroups { groups: (topk * n, 1, 1), threadgroup: (32, 1, 1) },
    )
}

/// Decode dense FFN front half: `act = silu(x W_gate^T) * (x W_up^T)` with the
/// separate Q4 GEMVs' sums and `silu_mul_bf16`'s rounding, in one dispatch.
pub fn gemv_gu_act(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &QuantWeights,
    up: &QuantWeights,
    x: &Tensor,
    act: &Tensor,
) -> Result<()> {
    let (n, k) = (gate.out_features(), gate.in_features());
    ensure!(gate.bits == 4 && up.bits == 4, "dense gate/up must be 4-bit");
    ensure!(
        up.out_features() == n && up.in_features() == k && up.group_size == gate.group_size,
        "dense gate/up shapes"
    );
    ensure!(x.numel() == k && act.numel() == n, "dense gate/up rows");
    ensure!(x.dtype() == DType::BF16 && act.dtype() == DType::BF16, "dense gate/up BF16");
    let pipeline = ctx.pipeline("lfm_gemv_q4_gu_act_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            gate.codes.binding(),
            gate.scales.binding(),
            gate.biases.binding(),
            up.codes.binding(),
            up.scales.binding(),
            up.biases.binding(),
            x.binding(),
            act.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(gate.group_size)],
        Grid::Threadgroups { groups: (n, 1, 1), threadgroup: (32, 1, 1) },
    )
}

/// Down expert GEMV with four expert simdgroups per output row, fused combine.
pub fn gemv_gather_down_combine4(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    down: &QuantWeights,
    act: &Tensor,
    ids: &Tensor,
    weights: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let topk = ids.numel();
    let k = down.in_features();
    let n = down.out_features() / 64;
    ensure!(topk == 4, "fused four-expert down needs topk 4");
    ensure!(act.numel() == topk * k && out.numel() == n && weights.numel() == topk, "combine shapes");
    ensure!(
        down.bits == 4 && down.group_size == FIXED_GS && k == FIXED_MOE_INTER && n == FIXED_HIDDEN,
        "fused down is the compile-time affine 4-bit/64 expert shape"
    );
    let pipeline =
        ctx.pipeline("lfm_gemv_q4_gather_down_combine4_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            down.codes.binding(),
            down.scales.binding(),
            down.biases.binding(),
            act.binding(),
            ids.binding(),
            weights.binding(),
            out.binding(),
        ],
        &[&u32_bytes(topk)],
        Grid::Threadgroups { groups: (n, 1, 1), threadgroup: (128, 1, 1) },
    )
}

/// GPU top-k routing for a prompt chunk.
pub fn moe_route_batch(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    bias: &Tensor,
    ids: &Tensor,
    gw: &Tensor,
    m: usize,
) -> Result<()> {
    let e = bias.numel();
    let topk = gw.numel() / m;
    ensure!(logits.numel() == m * e && ids.numel() == m * topk, "route batch shapes");
    let pipeline = ctx.pipeline("lfm_moe_route_batch", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[logits.binding(), bias.binding(), ids.binding(), gw.binding()],
        &[&u32_bytes(e), &u32_bytes(topk), &(1e-6f32).to_ne_bytes()],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (64, 1, 1) },
    )
}

pub fn moe_zero_counts(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    counts: &Tensor,
) -> Result<()> {
    let e = counts.numel();
    let pipeline = ctx.pipeline("lfm_moe_zero_counts", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[counts.binding()],
        &[&u32_bytes(e)],
        Grid::Threads { grid: (e, 1, 1), threadgroup: (64.min(e).max(1), 1, 1) },
    )
}

pub fn moe_count_slots(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    ids: &Tensor,
    counts: &Tensor,
    s: usize,
) -> Result<()> {
    let pipeline = ctx.pipeline("lfm_moe_count", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[ids.binding(), counts.binding()],
        &[&u32_bytes(s)],
        Grid::Threads { grid: (s, 1, 1), threadgroup: (256, 1, 1) },
    )
}

pub fn moe_offsets(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    counts: &Tensor,
    offsets: &Tensor,
) -> Result<()> {
    let e = counts.numel();
    ensure!(offsets.numel() == e, "offsets shape");
    let pipeline = ctx.pipeline("lfm_moe_offsets", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[counts.binding(), offsets.binding()],
        &[&u32_bytes(e)],
        Grid::Threadgroups { groups: (1, 1, 1), threadgroup: (32, 1, 1) },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn moe_sorted(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    ids: &Tensor,
    offsets: &Tensor,
    ids_sorted: &Tensor,
    slotpos: &Tensor,
    m: usize,
    topk: usize,
) -> Result<()> {
    let e = offsets.numel();
    let pipeline = ctx.pipeline("lfm_moe_sorted_parallel", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[ids.binding(), offsets.binding(), ids_sorted.binding(), slotpos.binding()],
        &[&u32_bytes(m), &u32_bytes(e), &u32_bytes(topk)],
        Grid::Threadgroups { groups: (e, 1, 1), threadgroup: (256, 1, 1) },
    )
}

pub fn moe_zero_blocks(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gu: &Tensor,
    dn: &Tensor,
    n_gu: usize,
    n_dn: usize,
) -> Result<()> {
    let n = n_gu.max(n_dn);
    let pipeline = ctx.pipeline("lfm_moe_zero_blocks", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[gu.binding(), dn.binding()],
        &[&u32_bytes(n_gu), &u32_bytes(n_dn)],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (256, 1, 1) },
    )
}

pub fn moe_block_offsets(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    counts: &Tensor,
    gu_off: &Tensor,
    dn_off: &Tensor,
    bm: usize,
    bn: usize,
    gu_n: usize,
    dn_n: usize,
) -> Result<()> {
    let e = counts.numel();
    ensure!(gu_off.numel() == e && dn_off.numel() == e, "block offset shape");
    let pipeline = ctx.pipeline("lfm_moe_block_offsets", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[counts.binding(), gu_off.binding(), dn_off.binding()],
        &[&u32_bytes(bm), &u32_bytes(bn), &u32_bytes(gu_n), &u32_bytes(dn_n), &u32_bytes(e)],
        Grid::Threadgroups { groups: (1, 1, 1), threadgroup: (32, 1, 1) },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn moe_blocks(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    counts: &Tensor,
    offsets: &Tensor,
    gu_off: &Tensor,
    dn_off: &Tensor,
    gu: &Tensor,
    dn: &Tensor,
    bm: usize,
    bn: usize,
    gu_n: usize,
    dn_n: usize,
) -> Result<()> {
    let e = counts.numel();
    ensure!(gu_off.numel() == e && dn_off.numel() == e, "block offset shape");
    let pipeline = ctx.pipeline("lfm_moe_blocks_parallel", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            counts.binding(),
            offsets.binding(),
            gu_off.binding(),
            dn_off.binding(),
            gu.binding(),
            dn.binding(),
        ],
        &[&u32_bytes(bm), &u32_bytes(gu_n), &u32_bytes(dn_n), &u32_bytes(bn)],
        Grid::Threadgroups { groups: (e, 1, 1), threadgroup: (128, 1, 1) },
    )
}

/// Weighted combine of the sorted expert rows back to token rows.
#[allow(clippy::too_many_arguments)]
pub fn moe_combine_rows(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    sorted: &Tensor,
    slotpos: &Tensor,
    weights: &Tensor,
    out: &Tensor,
    topk: usize,
) -> Result<()> {
    let h = out.shape()[1];
    let m = out.shape()[0];
    ensure!(
        sorted.shape()[1] == h && sorted.shape()[0] == m * topk,
        "combine sorted shape"
    );
    ensure!(slotpos.numel() == m * topk && weights.numel() == m * topk, "combine index shape");
    for t in [sorted, weights, out] {
        ensure!(t.dtype() == DType::BF16, "combine BF16");
    }
    ensure!(slotpos.dtype() == DType::U32, "combine slot positions");
    if h.is_multiple_of(16) {
        let h16 = h / 16;
        let pipeline =
            ctx.pipeline("lfm_moe_combine_rows_bf16_v16", SOURCE, MslVersion::V3_1)?;
        pass.dispatch_at(
            &pipeline,
            &[sorted.binding(), slotpos.binding(), weights.binding(), out.binding()],
            &[&u32_bytes(h16), &u32_bytes(topk)],
            Grid::Threads {
                grid: (h16, m, 1),
                threadgroup: (h16.min(256), 1, 1),
            },
        )
    } else if h.is_multiple_of(8) {
        let h8 = h / 8;
        let pipeline =
            ctx.pipeline("lfm_moe_combine_rows_bf16_v8", SOURCE, MslVersion::V3_1)?;
        pass.dispatch_at(
            &pipeline,
            &[sorted.binding(), slotpos.binding(), weights.binding(), out.binding()],
            &[&u32_bytes(h8), &u32_bytes(topk)],
            Grid::Threads {
                grid: (h8, m, 1),
                threadgroup: (h8.min(256), 1, 1),
            },
        )
    } else {
        let pipeline = ctx.pipeline("lfm_moe_combine_rows_bf16", SOURCE, MslVersion::V3_1)?;
        pass.dispatch_at(
            &pipeline,
            &[sorted.binding(), slotpos.binding(), weights.binding(), out.binding()],
            &[&u32_bytes(h), &u32_bytes(topk)],
            Grid::Threads { grid: (h, m, 1), threadgroup: (256, 1, 1) },
        )
    }
}

/// Short convolution reading the packed BCx projection output directly.
pub fn conv3_gate_bcx(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    bcx: &Tensor,
    w: &Tensor,
    state: &Tensor,
    out: &Tensor,
    new_state: &Tensor,
) -> Result<()> {
    let h = out.shape()[1];
    let m = out.shape()[0];
    ensure!(
        bcx.shape() == [m, 3 * h] && w.numel() == h * 3,
        "conv3 bcx shapes"
    );
    ensure!(state.shape() == [2, h] && new_state.shape() == [2, h], "conv3 state shape");
    for t in [bcx, w, state, out, new_state] {
        ensure!(t.dtype() == DType::BF16, "conv3 BF16");
    }
    let pipeline = ctx.pipeline("lfm_conv3_gate_bcx_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            bcx.binding(),
            w.binding(),
            state.binding(),
            out.binding(),
            new_state.binding(),
        ],
        &[&u32_bytes(h), &u32_bytes(m)],
        Grid::Threads { grid: (h, m, 1), threadgroup: (256, 1, 1) },
    )
}

/// Argmax over a BF16 logits row, ties to the lowest token id.
pub fn argmax_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    partials: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let n = x.numel();
    ensure!(x.dtype() == DType::BF16, "argmax input BF16");
    ensure!(partials.numel() * partials.dtype().size() >= ARGMAX_GROUPS * 8, "argmax scratch");
    let chunk = n.div_ceil(ARGMAX_GROUPS);
    let partial = ctx.pipeline("lfm_argmax_bf16_partial", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &partial,
        &[x.binding(), partials.binding()],
        &[&u32_bytes(n), &u32_bytes(chunk)],
        Grid::Threadgroups { groups: (ARGMAX_GROUPS, 1, 1), threadgroup: (256, 1, 1) },
    )?;
    pass.memory_barrier()?;
    let final_stage = ctx.pipeline("lfm_argmax_final", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &final_stage,
        &[partials.binding(), out.binding()],
        &[&u32_bytes(ARGMAX_GROUPS)],
        Grid::Threadgroups { groups: (1, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Decode-step output head with an argmax epilogue: the BF16 logits of
/// `lfm_gemv_q6_bf16_x4` are never stored; every threadgroup folds its argmax
/// candidate into the 64-bit key `best` (zero before the dispatch), which
/// [`token_tail`] reads and clears.
pub fn gemv_head_argmax(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    x: &Tensor,
    best: &Tensor,
) -> Result<()> {
    let (n, k) = (w.out_features(), w.in_features());
    ensure!(w.bits == 6 && n % 16 == 0, "head argmax needs the 6-bit x4 head");
    ensure!(x.numel() == k && x.dtype() == DType::BF16, "head argmax input");
    ensure!(best.byte_len() >= 8 && best.binding().1.is_multiple_of(8), "head argmax key");
    let pipeline = ctx.pipeline("lfm_gemv_q6_bf16_x4_argmax", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[w.codes.binding(), w.scales.binding(), w.biases.binding(), x.binding(), best.binding()],
        &[&u32_bytes(k), &u32_bytes(w.group_size)],
        Grid::Threadgroups { groups: (n / 16, 1, 1), threadgroup: (128, 1, 1) },
    )
}

/// Decode token tail: reads and clears the head's argmax key and writes the
/// token to `out_tok`. With `next = Some((embed, norm_w, x, normed))` it also
/// writes that token's dequantized embedding row to `x` and its RMSNorm to
/// `normed`, i.e. the next step's `gather_rows_q6` + `rmsnorm_bf16_v8`.
pub fn token_tail(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    best: &Tensor,
    out_tok: &Tensor,
    next: Option<(&QuantWeights, &Tensor, &Tensor, &Tensor)>,
    eps: f32,
) -> Result<()> {
    ensure!(out_tok.dtype() == DType::U32 && out_tok.numel() == 1, "token tail output");
    let key = best.binding();
    // Without `next` the row bindings are never read; the key buffer fills them.
    let (rows, k, gs) = match next {
        Some((e, w, x, normed)) => {
            let k = e.in_features();
            ensure!(e.bits == 6 && e.group_size == 64 && k == 2048, "token tail embedding");
            for t in [w, x, normed] {
                ensure!(t.numel() == k && t.dtype() == DType::BF16, "token tail rows");
                ensure!(t.binding().1.is_multiple_of(16), "token tail row alignment");
            }
            let r = [e.codes.binding(), e.scales.binding(), e.biases.binding(), w.binding(), x.binding(), normed.binding()];
            (r, k, e.group_size)
        }
        None => ([key; 6], 0, 0),
    };
    let pipeline = ctx.pipeline("lfm_token_tail", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[key, out_tok.binding(), rows[0], rows[1], rows[2], rows[3], rows[4], rows[5]],
        &[
            &u32_bytes(k),
            &u32_bytes(gs),
            &u32_bytes(k),
            &eps.to_ne_bytes(),
            &0.0f32.to_ne_bytes(),
            &u32_bytes(usize::from(next.is_some())),
        ],
        Grid::Threadgroups { groups: (1, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Length at which L=1 attention switches from the single-pass vector kernel
/// to the blocked two-pass form the reference uses for long contexts.
pub const SDPA_2PASS_MIN_LEN: usize = 8_192;
/// Sequence blocks in the two-pass L=1 attention (a multiple of 32).
pub const SDPA_2PASS_BLOCKS: usize = 512;
/// Largest block count `Scratch` reserves for the two-pass path.
pub const SDPA_2PASS_MAX_BLOCKS: usize = 1024;

/// Block count for a given L=1 context length: fewer, longer blocks keep the
/// per-block loop overhead low up to 48k positions, while 256 blocks expose
/// enough warps to saturate the long-context case.
fn sdpa_2pass_blocks(len: usize) -> usize {
    if len <= 49_152 { 128 } else { SDPA_2PASS_BLOCKS }
}

/// Single-token GQA attention with the reference sdpa_vector arithmetic.
/// The reference uses this kernel for every L=1 call (the last prompt token
/// and every decode step), so an L=1 request must not take the prefill tile.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_vector(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    out: &Tensor,
    base_len: usize,
    m: usize,
    nq: usize,
    group: usize,
    scale: f32,
) -> Result<()> {
    ensure!(m == 1, "sdpa_vector handles one query row");
    ensure!(k_cache.shape()[2] == 64 && v_cache.shape()[2] == 64, "vector D=64");
    ensure!(q.numel() == nq * 64 && out.numel() == nq * 64, "vector q/out shape");
    ensure!(q.dtype() == DType::BF16 && out.dtype() == DType::BF16, "vector BF16");
    let max_seq = k_cache.shape()[1];
    let kvh = k_cache.shape()[0];
    ensure!(kvh * group == nq, "vector GQA head map");
    let len = base_len + m;
    let pipeline = ctx.pipeline("lfm_sdpa_vector_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            q.binding(),
            k_cache.binding(),
            v_cache.binding(),
            out.binding(),
        ],
        &[
            &u32_bytes(max_seq),
            &u32_bytes(len),
            &u32_bytes(nq),
            &u32_bytes(group),
            &scale.to_ne_bytes(),
        ],
        Grid::Threadgroups { groups: (nq, 1, 1), threadgroup: (1024, 1, 1) },
    )
}

/// Decode-only: `norm_rope_qk_scatter` folded into `sdpa_vector`. Each
/// query head's threadgroup normalizes and rotates its own query row and its
/// KV head's key row at position `pos` (the first query head of the KV head
/// writes that key row into the cache), then runs the single-pass body with
/// those staged values; every stored value is the two kernels'. Only the
/// single-pass length range (`pos + 1 < SDPA_2PASS_MIN_LEN`) takes it.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_vector_norm_rope(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q_in: &Tensor,
    q_w: &Tensor,
    k_in: &Tensor,
    k_w: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    out: &Tensor,
    pos: usize,
    theta: f32,
    nq: usize,
    group: usize,
    scale: f32,
) -> Result<()> {
    let d = 64usize;
    ensure!(k_cache.shape()[2] == d && v_cache.shape()[2] == d, "vector D=64");
    let max_seq = k_cache.shape()[1];
    let kvh = k_cache.shape()[0];
    ensure!(kvh * group == nq, "vector GQA head map");
    ensure!(
        q_in.numel() == nq * d && out.numel() == nq * d && k_in.numel() == kvh * d,
        "vector norm rope shapes"
    );
    ensure!(q_w.numel() == d && k_w.numel() == d, "vector norm rope weights");
    ensure!(pos < max_seq, "vector norm rope position");
    let len = pos + 1;
    ensure!(len < SDPA_2PASS_MIN_LEN, "vector norm rope is the single-pass range");
    for t in [q_in, q_w, k_in, k_w, k_cache, v_cache, out] {
        ensure!(t.dtype() == DType::BF16, "vector norm rope BF16");
    }
    let pipeline =
        ctx.pipeline("lfm_sdpa_vector_norm_rope_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            q_in.binding(),
            q_w.binding(),
            k_in.binding(),
            k_w.binding(),
            k_cache.binding(),
            v_cache.binding(),
            out.binding(),
        ],
        &[
            &u32_bytes(max_seq),
            &u32_bytes(len),
            &u32_bytes(nq),
            &u32_bytes(group),
            &scale.to_ne_bytes(),
            &u32_bytes(d),
            &theta.to_ne_bytes(),
            &1e-5f32.to_ne_bytes(),
        ],
        Grid::Threadgroups { groups: (nq, 1, 1), threadgroup: (1024, 1, 1) },
    )
}

/// Blocked two-pass L=1 attention. `partials` is BF16 `[nq, blocks, 64]`,
/// `sums`/`maxs` are F32 `[nq, blocks]`; the callers reserve at least
/// `SDPA_2PASS_MAX_BLOCKS` entries per head.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_vector_2pass(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    out: &Tensor,
    partials: &Tensor,
    sums: &Tensor,
    maxs: &Tensor,
    base_len: usize,
    m: usize,
    nq: usize,
    group: usize,
    scale: f32,
    blocks: usize,
) -> Result<()> {
    ensure!(m == 1, "sdpa_vector_2pass handles one query row");
    ensure!(k_cache.shape()[2] == 64 && v_cache.shape()[2] == 64, "vector D=64");
    ensure!(q.numel() == nq * 64 && out.numel() == nq * 64, "vector q/out shape");
    ensure!(q.dtype() == DType::BF16 && out.dtype() == DType::BF16, "vector BF16");
    ensure!(blocks % 32 == 0 && blocks <= SDPA_2PASS_MAX_BLOCKS, "2pass block count");
    let max_seq = k_cache.shape()[1];
    let kvh = k_cache.shape()[0];
    ensure!(kvh * group == nq, "vector GQA head map");
    ensure!(
        partials.numel() >= nq * blocks * 64 && partials.dtype() == DType::BF16,
        "2pass partials scratch too small"
    );
    ensure!(
        sums.numel() >= nq * blocks && sums.dtype() == DType::F32,
        "2pass sums scratch too small"
    );
    ensure!(
        maxs.numel() >= nq * blocks && maxs.dtype() == DType::F32,
        "2pass maxs scratch too small"
    );
    let len = base_len + m;
    let gqa4 = group == 4;
    let stage1 = ctx.pipeline(
        if gqa4 {
            "lfm_sdpa_vector_2pass1_gqa4_bf16"
        } else {
            "lfm_sdpa_vector_2pass1_bf16"
        },
        SOURCE,
        MslVersion::V3_1,
    )?;
    pass.dispatch_at(
        &stage1,
        &[
            q.binding(),
            k_cache.binding(),
            v_cache.binding(),
            partials.binding(),
            sums.binding(),
            maxs.binding(),
        ],
        &[
            &u32_bytes(max_seq),
            &u32_bytes(len),
            &u32_bytes(group),
            &scale.to_ne_bytes(),
            &u32_bytes(blocks),
        ],
        Grid::Threadgroups {
            groups: (kvh, blocks, 1),
            threadgroup: if gqa4 { (32, 1, 1) } else { (32, group, 1) },
        },
    )?;
    pass.level_barrier(&[partials, sums, maxs])?;
    let stage2 = ctx.pipeline("lfm_sdpa_vector_2pass2_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &stage2,
        &[partials.binding(), sums.binding(), maxs.binding(), out.binding()],
        &[&u32_bytes(blocks)],
        Grid::Threadgroups { groups: (nq, 1, 1), threadgroup: (1024, 1, 1) },
    )
}

/// Causal GQA prefill/decode attention over the KV cache.
#[allow(clippy::too_many_arguments)]
pub fn sdpa(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    out: &Tensor,
    base_len: usize,
    m: usize,
    nq: usize,
    group: usize,
    scale: f32,
    vector_scratch: Option<(&Tensor, &Tensor, &Tensor)>,
) -> Result<()> {
    // The reference's L=1 path is sdpa_vector, not the chunked prefill tile;
    // its online softmax and output accumulation determine the L=1 logits.
    // Long L=1 contexts use the same blocked two-pass layout the reference
    // selects there.
    if m == 1 {
        let len = base_len + m;
        if len >= SDPA_2PASS_MIN_LEN
            && let Some((partials, sums, maxs)) = vector_scratch
        {
            return sdpa_vector_2pass(
                ctx, pass, q, k_cache, v_cache, out, partials, sums, maxs,
                base_len, m, nq, group, scale, sdpa_2pass_blocks(len),
            );
        }
        return sdpa_vector(
            ctx, pass, q, k_cache, v_cache, out, base_len, m, nq, group, scale,
        );
    }
    ensure!(k_cache.shape()[2] == 64 && v_cache.shape()[2] == 64, "attention D=64");
    // Every causal prefill chunk uses the steel NAX flash tile: MLX 0.32.3's
    // own attention arithmetic, which the reference uses. It keeps the score
    // tile in registers.
    crate::kernels::steel::attention(
        ctx, pass, q, k_cache, v_cache, out, base_len, m, nq, group, scale,
    )
}
