//! MLX 0.32.3 steel attention adapted to this engine's BF16
//! `[tokens, heads, 64]` Q/O layout and `[KV heads, seq, 64]` cache.
//!
//! The Metal-4 (NAX) tensor-op flash attention is generated from the pinned
//! MLX headers by `tools/gen_steel_attn.py` and compiled at run time like
//! every other kernel. It keeps the score tile in registers and carries the
//! online softmax state per row, so no FP32 score tile round-trips through
//! threadgroup memory. `AttnParams` mirrors `mlx::steel::AttnParams`.

use anyhow::{ensure, Result};

use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};

const SOURCE_NAX: &str = include_str!("metal/steel_attn_nax.metal");

/// Matches `mlx::steel::AttnParams` field for field.
#[repr(C)]
struct AttnParams {
    b: i32,
    h: i32,
    d: i32,
    q_l: i32,
    k_l: i32,
    gqa_factor: i32,
    scale: f32,
    nq: i32,
    nk: i32,
    nq_aligned: i32,
    nk_aligned: i32,
    ql_rem: i32,
    kl_rem: i32,
    ql_off: i32,
    q_strides: [i64; 3],
    k_strides: [i64; 3],
    v_strides: [i64; 3],
    o_strides: [i64; 3],
}

impl AttnParams {
    fn bytes(&self) -> Vec<u8> {
        // SAFETY: AttnParams is repr(C) and contains only integers and f32, so
        // every byte of the value is initialized and readable.
        let raw = unsafe {
            core::slice::from_raw_parts(
                (self as *const AttnParams).cast::<u8>(),
                core::mem::size_of::<AttnParams>(),
            )
        };
        raw.to_vec()
    }
}

/// A generated kernel variant: source, function name, query-tile height and
/// threadgroup size. The values follow the instantiation in the generated
/// MSL (`BQ`, and `WM * WN * 32` threads).
#[derive(Clone, Copy)]
struct Variant {
    source: &'static str,
    func: &'static str,
    bq: usize,
    bk: usize,
    threads: usize,
    version: MslVersion,
}

fn variant() -> Variant {
    // The shipped long-context tile: 64 query rows, 32 keys per block, two
    // simdgroups. `bk` sizes the launcher's block loop and must equal the BK
    // template argument of the generated kernel.
    Variant {
        source: SOURCE_NAX,
        func: "lfm_steel_nax_bq64k32",
        bq: 64,
        bk: 32,
        threads: 128,
        version: MslVersion::V4_0,
    }
}

/// Causal GQA attention for a prefill chunk of `m` query rows.
#[allow(clippy::too_many_arguments)]
pub fn attention(
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
    ensure!(m > 0, "steel attention needs query rows");
    ensure!(
        k_cache.dtype() == DType::BF16
            && v_cache.dtype() == DType::BF16
            && q.dtype() == DType::BF16
            && out.dtype() == DType::BF16,
        "steel attention BF16"
    );
    let kvh = k_cache.shape()[0];
    let max_seq = k_cache.shape()[1];
    ensure!(k_cache.shape()[2] == 64 && v_cache.shape()[2] == 64, "steel D=64");
    ensure!(kvh * group == nq, "steel GQA head map");
    ensure!(q.numel() >= m * nq * 64 && out.numel() >= m * nq * 64, "steel q/o size");

    let cfg = variant();
    let bq = cfg.bq;
    let q_l = m;
    let k_l = base_len + m;
    let nq_blocks = q_l.div_ceil(bq);
    let nk_blocks = k_l.div_ceil(cfg.bk);
    let align_q = q_l % bq == 0;
    let align_k = k_l % cfg.bk == 0;
    let params = AttnParams {
        b: 1,
        h: nq as i32,
        d: 64,
        q_l: q_l as i32,
        k_l: k_l as i32,
        gqa_factor: group as i32,
        scale,
        nq: nq_blocks as i32,
        nk: nk_blocks as i32,
        nq_aligned: (q_l / bq) as i32,
        nk_aligned: (k_l / cfg.bk) as i32,
        ql_rem: (q_l % bq) as i32,
        kl_rem: (k_l % cfg.bk) as i32,
        ql_off: base_len as i32,
        q_strides: [0, 64, (nq * 64) as i64],
        k_strides: [0, (max_seq * 64) as i64, 64],
        v_strides: [0, (max_seq * 64) as i64, 64],
        o_strides: [0, 64, (nq * 64) as i64],
    };
    let flags = (align_q as u64) | ((align_k as u64) << 1);
    let kernel = ctx.pipeline_fc(
        cfg.func,
        cfg.source,
        cfg.version,
        flags,
        &[(200, align_q), (201, align_k), (300, false), (301, true), (302, false)],
    )?;
    let bytes = params.bytes();
    pass.dispatch_at(
        &kernel,
        &[
            q.binding(),
            k_cache.binding(),
            v_cache.binding(),
            out.binding(),
        ],
        &[&bytes],
        Grid::Threadgroups {
            // GQA-adjacent head order: the four query heads of one KV head
            // are consecutive in dispatch order for each query block, so
            // their reads of the same causal K/V prefix hit the cache instead
            // of streaming it once per head. The generated NAX kernel reads
            // the axes as (head within the GQA group, query block, KV head);
            // every threadgroup's rows, causal range and arithmetic are
            // unchanged.
            groups: (group, nq_blocks, kvh),
            threadgroup: (cfg.threads, 1, 1),
        },
    )
}
