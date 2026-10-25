// Modified from Lily (Perplexity AI, Apache-2.0): see VENDORED.md for the changes to this file.
//! Element-wise and embedding-row gather kernels.

use anyhow::{Result, ensure};

use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};

const SOURCE: &str = include_str!("metal/elementwise.metal");

fn binary_op(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    fn_name: &'static str,
    dtype: DType,
    a: &Tensor,
    b: &Tensor,
    out: &Tensor,
) -> Result<()> {
    binary_op_source(ctx, pass, fn_name, dtype, a, b, out, SOURCE)
}

#[allow(clippy::too_many_arguments)]
fn binary_op_source(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    fn_name: &'static str,
    dtype: DType,
    a: &Tensor,
    b: &Tensor,
    out: &Tensor,
    source: &'static str,
) -> Result<()> {
    let n = a.numel();
    ensure!(b.numel() == n && out.numel() == n, "size mismatch in {fn_name}");
    for t in [a, b, out] {
        ensure!(t.dtype() == dtype, "{fn_name} expects {dtype:?}, got {:?}", t.dtype());
    }
    let pipeline = ctx.pipeline(fn_name, source, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[a.binding(), b.binding(), out.binding()],
        &[],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (256.min(n), 1, 1) },
    )
}

pub fn silu_mul_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &Tensor,
    up: &Tensor,
    out: &Tensor,
) -> Result<()> {
    binary_op(ctx, pass, "silu_mul_bf16", DType::BF16, gate, up, out)
}

/// Batched embedding lookup: `out[i, :] = table[ids[i], :]`.
pub fn gather_rows_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    table: &Tensor,
    ids: &Tensor,
    out: &Tensor,
) -> Result<()> {
    ensure!(ids.dtype() == DType::U32, "ids must be U32");
    let m = ids.numel();
    ensure!(m > 0 && out.numel().is_multiple_of(m), "out rows mismatch");
    let h = out.numel() / m;
    ensure!(table.numel().is_multiple_of(h), "table not a multiple of row size");
    if h.is_multiple_of(16) {
        let h16 = h / 16;
        let pipeline = ctx.pipeline("gather_rows_bf16_v16", SOURCE, MslVersion::V3_1)?;
        pass.dispatch_at(
            &pipeline,
            &[table.binding(), ids.binding(), out.binding()],
            &[&crate::kernels::u32_bytes(h16)],
            Grid::Threads {
                grid: (h16, m, 1),
                threadgroup: (h16.min(256), 1, 1),
            },
        )
    } else if h.is_multiple_of(8) {
        let h8 = h / 8;
        let pipeline = ctx.pipeline("gather_rows_bf16_v8", SOURCE, MslVersion::V3_1)?;
        pass.dispatch_at(
            &pipeline,
            &[table.binding(), ids.binding(), out.binding()],
            &[&crate::kernels::u32_bytes(h8)],
            Grid::Threads {
                grid: (h8, m, 1),
                threadgroup: (h8.min(256), 1, 1),
            },
        )
    } else {
        let pipeline = ctx.pipeline("gather_rows_bf16", SOURCE, MslVersion::V3_1)?;
        pass.dispatch_at(
            &pipeline,
            &[table.binding(), ids.binding(), out.binding()],
            &[&crate::kernels::u32_bytes(h)],
            Grid::Threads { grid: (m * h, 1, 1), threadgroup: (256.min(m * h), 1, 1) },
        )
    }
}
