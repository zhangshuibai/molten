// Modified from Lily (Perplexity AI, Apache-2.0): see VENDORED.md for the changes to this file.
//! BF16 matrix multiplication over row-major `[out, in]` weights.

use std::sync::OnceLock;

use anyhow::{Result, ensure};

use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::Tensor;

const SOURCE: &str = include_str!("metal/gemm.metal");

/// Smallest row count that uses the 128x128 dense-GEMM tile. Reference
/// prefill chunks are at most 2048 rows and the merged submissions are 8191 or
/// 8192 rows, so this selects the wide tile only for merged groups. The wide
/// tile stages half the activation and weight traffic per output element and
/// measured faster there; at the 2048-row chunks it was neutral to slightly
/// slower, so those keep the 64x64 tile.
const WIDE_MIN_M: usize = 4096;

fn nax_version(ctx: &MetalContext) -> Result<MslVersion> {
    static VERSION: OnceLock<Result<Option<MslVersion>, String>> = OnceLock::new();
    let resolved = VERSION.get_or_init(|| {
        let probe = |v: MslVersion| ctx.pipeline("gemm_bf16_nt_nax", SOURCE, v).is_ok();
        Ok([MslVersion::V4_0, MslVersion::V4_1].into_iter().find(|v| probe(*v)))
    });
    resolved.as_ref().map_err(|e| anyhow::anyhow!("{e}"))?.ok_or_else(|| {
        anyhow::anyhow!("the NAX GEMM compiles at no supported language version")
    })
}

pub fn gemm_bf16_nt(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
) -> Result<()> {
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[0];
    ensure!(b.shape() == [n, k], "B shape {:?} != [{n}, {k}]", b.shape());
    // Output shape is arbitrary but must contain M*N contiguous elements.
    ensure!(c.numel() == m * n, "C numel {} != {m}x{n}", c.numel());
    let pipeline = if m >= WIDE_MIN_M {
        ctx.pipeline("gemm_bf16_nt_nax_wide", SOURCE, nax_version(ctx)?)?
    } else {
        ctx.pipeline("gemm_bf16_nt_nax", SOURCE, nax_version(ctx)?)?
    };
    let (bm, bn, threads) = if m >= WIDE_MIN_M { (128, 128, 256) } else { (64, 64, 128) };
    pass.dispatch_at(
        &pipeline,
        &[a.binding(), b.binding(), c.binding()],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(m)],
        Grid::Threadgroups {
            groups: (n.div_ceil(bn), m.div_ceil(bm), 1),
            threadgroup: (threads, 1, 1),
        },
    )
}
