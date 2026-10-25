// Modified from Lily (Perplexity AI, Apache-2.0): see VENDORED.md for the changes to this file.
//! RoPE and KV-cache updates.

use anyhow::{Result, ensure};

use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::Tensor;

/// Attention Metal source.
pub const SOURCE: &str = include_str!("metal/attention.metal");

/// In-place partial NeoX RoPE over `[M, heads, D]`; token `m` rotates at
/// position `base_pos + m` (decode is the M=1 case).
#[allow(clippy::too_many_arguments)]
pub fn rope_neox(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    heads: usize,
    rot: usize,
    base_pos: usize,
    theta: f32,
) -> Result<()> {
    let d = *x.shape().last().ok_or_else(|| anyhow::anyhow!("rope on 0-d tensor"))?;
    ensure!(x.numel().is_multiple_of(heads * d), "x not a multiple of heads*D");
    let m = x.numel() / (heads * d);
    ensure!(rot.is_multiple_of(2) && rot <= d, "rotary dim {rot} invalid for D {d}");
    let pipeline = ctx.pipeline("rope_neox_bf16", SOURCE, MslVersion::V3_1)?;
    let pairs = heads * rot / 2;
    pass.dispatch_at(
        &pipeline,
        &[x.binding()],
        &[
            &u32_bytes(d),
            &u32_bytes(rot),
            &u32_bytes(base_pos),
            &theta.to_ne_bytes(),
            &u32_bytes(heads * d),
        ],
        Grid::Threads { grid: (pairs, m, 1), threadgroup: (256.min(pairs), 1, 1) },
    )
}

/// Appends M tokens' per-head rows (`[M, KVH, D]`) into `cache`
/// (`[KVH, max_seq, D]`) at positions `base_pos..base_pos + M`.
pub fn scatter_kv(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    cache: &Tensor,
    rows: &Tensor,
    base_pos: usize,
) -> Result<()> {
    let (kvh, max_seq, d) = (cache.shape()[0], cache.shape()[1], cache.shape()[2]);
    ensure!(
        rows.numel().is_multiple_of(kvh * d),
        "rows numel {} not [M, KVH, D]",
        rows.numel()
    );
    let m = rows.numel() / (kvh * d);
    ensure!(
        base_pos + m <= max_seq,
        "positions {base_pos}+{m} exceed max_seq {max_seq}"
    );
    let pipeline = ctx.pipeline("scatter_kv_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[cache.binding(), rows.binding()],
        &[&u32_bytes(d), &u32_bytes(max_seq), &u32_bytes(base_pos), &u32_bytes(kvh)],
        Grid::Threads { grid: (m * kvh * d, 1, 1), threadgroup: (256, 1, 1) },
    )
}
