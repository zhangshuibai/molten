// Modified from Lily (Perplexity AI, Apache-2.0): see VENDORED.md for the changes to this file.
//! Affine Q4/Q8 kernels with BF16 activations and F32 accumulation.
//! Q4 packs eight low-nibble-first codes per u32; `w = scale*q + bias`.

use anyhow::{Result, ensure};

use crate::kernels::gemm::gemm_bf16_nt;
use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};
use crate::weights::QuantWeights;

const SOURCE: &str = include_str!("metal/quant.metal");
/// MLX 0.32.3's NAX qmm_t body, flattened by `tools/gen_quant_nax.py`.
const SOURCE_NAX: &str = include_str!("metal/quant_nax.metal");

/// Row-tile height T of a grouped Q4 GEMM; the block-map granularity.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoeTile {
    T32Sg4,
    /// 128-row tile with a single 64-column stripe per threadgroup (128
    /// threads): the activation tile is staged once per tile, not once per
    /// column stripe.
    T128N64Sg4Wn1,
}

impl MoeTile {
    /// Row-tile height T — the block map's granularity.
    pub fn rows(self) -> usize {
        match self {
            Self::T32Sg4 => 32,
            Self::T128N64Sg4Wn1 => 128,
        }
    }

    /// Output-tile width — the block map's column granularity.
    pub fn cols(self) -> usize {
        64
    }
}

fn check_quant(w: &QuantWeights) -> Result<(usize, usize)> {
    let (n, k) = (w.out_features(), w.in_features());
    ensure!(matches!(w.bits, 4 | 8), "unsupported bit width {}", w.bits);
    // A uint4 block must not cross a quantization group.
    ensure!(
        k.is_multiple_of(w.group_size) && w.group_size.is_multiple_of(128 / w.bits),
        "in_features {k} / group size {} not block-packable at {} bits",
        w.group_size,
        w.bits
    );
    ensure!(w.codes.dtype() == DType::U32, "codes must be U32");
    // uint4 code loads require 16-byte alignment.
    ensure!(
        w.codes.binding().1.is_multiple_of(16),
        "codes byte offset {} not 16-aligned; the kernels read codes as uint4",
        w.codes.binding().1
    );
    ensure!(
        w.scales.shape() == [n, k / w.group_size]
            && w.biases.shape() == [n, k / w.group_size],
        "scales/biases shape mismatch for [{n}, {k}] gs={}",
        w.group_size
    );
    Ok((n, k))
}

/// Writes `out[N, K] = dequant(W)` in bf16.
pub fn dequant_to_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    out: &Tensor,
) -> Result<()> {
    let (n, k) = check_quant(w)?;
    ensure!(
        out.shape() == [n, k] && out.dtype() == DType::BF16,
        "dequant out must be BF16 [{n}, {k}], got {:?} {:?}",
        out.shape(),
        out.dtype()
    );
    let fn_name = if w.bits == 4 { "dequant_q4_bf16" } else { "dequant_q8_bf16" };
    let pipeline = ctx.pipeline(fn_name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[w.codes.binding(), w.scales.binding(), w.biases.binding(), out.binding()],
        &[&u32_bytes(k), &u32_bytes(w.group_size)],
        Grid::Threads { grid: (k / (32 / w.bits), n, 1), threadgroup: (32, 1, 1) },
    )
}

/// Bulk-dequantizes a whole Q4 matrix into BF16 (four 64-value groups per
/// thread), for MoE layers whose routed rows make per-tile dequantization
/// redundant.
pub fn dequant_q4_bf16_bulk(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    out: &Tensor,
) -> Result<()> {
    let (n, k) = check_quant(w)?;
    ensure!(
        w.bits == 4 && k.is_multiple_of(128),
        "bulk dequant needs 4-bit weights with K % 128 == 0"
    );
    ensure!(
        out.shape() == [n, k] && out.dtype() == DType::BF16,
        "bulk dequant out must be BF16 [{n}, {k}]"
    );
    let pipeline = ctx.pipeline("dequant_q4_bf16_bulk", SOURCE, MslVersion::V3_1)?;
    let threads = (n * k / 128).div_ceil(256);
    pass.dispatch_at(
        &pipeline,
        &[w.codes.binding(), w.scales.binding(), w.biases.binding(), out.binding()],
        &[
            &u32_bytes(k),
            &u32_bytes(w.group_size),
            &u32_bytes(n),
        ],
        Grid::Threadgroups { groups: (threads, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Dequantizes `W` into bf16 scratch, then computes
/// `C[M, N] = A[M, K] . W[N, K]^T`.
pub fn gemm_quant_bf16_nt(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    scratch: &Tensor,
) -> Result<()> {
    let (n, k) = check_quant(w)?;
    ensure!(
        scratch.numel() >= n * k && scratch.dtype() == DType::BF16,
        "dequant scratch holds {} bf16 elements, needs {}",
        scratch.numel(),
        n * k
    );
    let staged = scratch.view(0, &[n, k])?;
    dequant_to_bf16(ctx, pass, w, &staged)?;
    gemm_bf16_nt(ctx, pass, a, &staged, c)
}

/// Fused gate/up grouped expert GEMM: one threadgroup stages both weight
/// tiles for a block-map entry and loads the activation tile once per K
/// slice, so the two GEMMs share A traffic. Each accumulator runs the same
/// per-slice MMA as `gemm_q4_grouped_mlx`, so both outputs are bitwise the
/// separate dispatches' results.
pub fn gemm_q4_grouped_mlx_gu(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    gate: &QuantWeights,
    up: &QuantWeights,
    gate_out: &Tensor,
    up_out: &Tensor,
    blocks: &Tensor,
    n_blocks: usize,
    tile: MoeTile,
) -> Result<()> {
    let (_, k) = check_quant(gate)?;
    let (_, ku) = check_quant(up)?;
    ensure!(
        gate.bits == 4 && gate.group_size == 64 && k.is_multiple_of(64),
        "fused gate/up GEMM needs 4-bit group-64 weights and K % 64 == 0"
    );
    ensure!(ku == k, "fused gate/up K mismatch");
    let (s, ak) = (a.shape()[0], a.shape()[1]);
    ensure!(ak == k, "A K {ak} != weight K {k}");
    let n = gate_out.shape()[1];
    ensure!(up_out.shape() == gate_out.shape(), "fused gate/up output shapes");
    ensure!(gate_out.shape() == [s, n], "fused gate/up C shape");
    ensure!(
        gate.out_features() == up.out_features() && gate.out_features().is_multiple_of(n),
        "fused gate/up expert width"
    );
    ensure!(n.is_multiple_of(64), "fused gate/up needs N % 64 == 0");
    ensure!(blocks.dtype() == DType::U32, "block map must be U32");
    ensure!(blocks.numel() >= n_blocks * 4 && n_blocks > 0, "bad block map");
    let (name, threads) = match tile {
        MoeTile::T32Sg4 => ("lfm_q4_grouped_mlx_gu_t32", 128),
        t => anyhow::bail!("fused gate/up GEMM has no {:?} tile", t),
    };
    let pipeline = ctx.pipeline(name, SOURCE_NAX, MslVersion::V4_0)?;
    pass.dispatch_at(
        &pipeline,
        &[
            gate.codes.binding(),
            gate.scales.binding(),
            gate.biases.binding(),
            up.codes.binding(),
            up.scales.binding(),
            up.biases.binding(),
            a.binding(),
            gate_out.binding(),
            up_out.binding(),
            blocks.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n)],
        Grid::Threadgroups {
            groups: (n_blocks, 1, 1),
            threadgroup: (threads, 1, 1),
        },
    )
}

/// Block-mapped grouped Q4 GEMM using MLX 0.32.3's `qmm_t` NAX body.
///
/// Each block-map entry is one (row tile, column tile) threadgroup over one
/// expert's stacked weight rows. The body keeps MLX's padded threadgroup B
/// stride (`BK + 16/sizeof(T)`) and its SK=32 `tile_matmad_nax` loop, so the
/// dequantized BF16 B tile and every output element are the values MLX's
/// reference qmm_t produces. `tile` selects the kernel whose `BM` matches the
/// map's row granularity.
///
/// `T128N64Sg4Wn1` selects the shared-B variant: one threadgroup per 128-row
/// tile stages the 64x64 weight tile once and runs its 64-row sub-tiles
/// through it, and a sub-tile that lies beyond the expert's segment skips its
/// own MMA work. The live sub-tiles execute MLX's body verbatim, so outputs
/// are unchanged.
pub fn gemm_q4_grouped_mlx(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    blocks: &Tensor,
    n_blocks: usize,
    tile: MoeTile,
) -> Result<()> {
    let (_, k) = check_quant(w)?;
    ensure!(
        w.bits == 4 && w.group_size == 64 && k.is_multiple_of(64),
        "MLX grouped GEMM needs 4-bit group-64 weights and K % 64 == 0"
    );
    let (s, ak) = (a.shape()[0], a.shape()[1]);
    ensure!(ak == k, "A K {ak} != weight K {k}");
    let n = c.shape()[1];
    ensure!(c.shape() == [s, n], "C shape {:?} != [{s}, {n}]", c.shape());
    ensure!(n.is_multiple_of(64), "grouped GEMM needs N % 64 == 0 (got {n})");
    ensure!(blocks.dtype() == DType::U32, "block map must be U32");
    ensure!(blocks.numel() >= n_blocks * 4 && n_blocks > 0, "bad block map");
    let (name, threads) = match tile {
        MoeTile::T32Sg4 => ("lfm_q4_grouped_mlx_t32", 128),
        MoeTile::T128N64Sg4Wn1 => ("lfm_q4_grouped_mlx_s128w1", 128),
    };
    let pipeline = ctx.pipeline(name, SOURCE_NAX, MslVersion::V4_0)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
            blocks.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n)],
        Grid::Threadgroups {
            groups: (n_blocks, 1, 1),
            threadgroup: (threads, 1, 1),
        },
    )
}
