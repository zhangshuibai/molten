//! LFM2.5-24B-A2B (mlx `lfm2_moe`) engine: checkpoint loading, the 40-layer
//! graph on Metal, logical state, checkpoints and greedy or sampled
//! generation (`sample`).

pub mod sample;

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context as _, Result, anyhow, ensure};
use objc2_metal::MTLBuffer;
use serde::Deserialize;

use crate::kernels::{attention, elementwise, lfm as klfm, norm, quant};
use crate::metal::{ComputePass, MetalContext};
use crate::safetensors::{Checkpoint, SafetensorsDType};
use crate::tensor::{DType, Tensor};
use crate::weights::QuantWeights;
use sample::{SampleScratch, Sampling};

pub const HIDDEN: usize = 2048;
pub const VOCAB: usize = 128_000;
pub const N_LAYERS: usize = 40;
pub const N_HEADS: usize = 32;
pub const N_KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 64;
pub const N_EXPERTS: usize = 64;
pub const TOP_K: usize = 4;
pub const MOE_INTER: usize = 1536;
pub const DENSE_INTER: usize = 11776;
pub const PREFILL_CHUNK: usize = 2048;
/// Largest number of rows one forward submission may cover. Consecutive
/// reference chunks are merged up to this size; the reference partition's
/// final L=1 chunk always stays separate. A merged chunk keeps every row's
/// attention and expert-GEMM arithmetic unchanged (each row's causal key
/// range, K order and tile row are the same); it only reuses each staged
/// projection weight across more rows and halves the submission count.
pub const PREFILL_CAP: usize = 8192;
/// Upper bound on grouped-GEMM block-map entries for one chunk.
const BLOCK_MAP_ENTRIES: usize = 65_536;
const BLOCK_MAP_WORDS: usize = BLOCK_MAP_ENTRIES * 4;

/// Size of the next prefill chunk: the reference's partition, `min(2048,
/// remaining - 1)`, then the last token alone (a size-1 chunk is an L=1
/// step). Its rounding defines the expected logits and tokens.
fn next_chunk(remaining: usize) -> usize {
    if remaining <= 1 {
        1
    } else {
        remaining.saturating_sub(1).min(PREFILL_CHUNK)
    }
}

/// One forward submission: the next reference chunk plus as many following
/// reference chunks as fit under `PREFILL_CAP`. Size-1 chunks (the reference's
/// final L=1 step) are never merged into a larger submission.
fn next_group(remaining: usize) -> usize {
    let mut total = next_chunk(remaining);
    if total <= 1 {
        return total;
    }
    while total < PREFILL_CAP {
        let left = remaining - total;
        if left == 0 {
            break;
        }
        let next = next_chunk(left);
        if next <= 1 || total + next > PREFILL_CAP {
            break;
        }
        total += next;
    }
    total
}

/// Conv-state slots per conv layer. A token (or prefill chunk) reads slot
/// `conv_slot` and writes slot `conv_slot + 1` (mod `CONV_SLOTS`), so the
/// states of the last `CONV_SLOTS - 1` steps stay intact. The protocol paths
/// only need two; the streaming decode loop needs more because it encodes up
/// to `STREAM_AHEAD` steps beyond the last token it has read back, and when it
/// stops early it rewinds to the state after the last kept token.
pub const CONV_SLOTS: usize = 32;
/// Decode steps the streaming loop keeps committed beyond the last token it
/// has handed to its callback.
pub const STREAM_AHEAD: usize = 16;
/// Slots of the device-resident token ring (`Scratch::gen_tokens`).
const TOKEN_RING: usize = 256;
/// Decode steps per command buffer. The host can encode the next batch while
/// the GPU executes the previous one, so encoding dozens of steps does not
/// delay the first GPU work; a batch of four keeps the per-submission overhead
/// small. The first batch holds a single step so the GPU starts after one
/// step's encode rather than four.
const FIRST_DECODE_BATCH: usize = 1;
const DECODE_BATCH: usize = 4;
const _: () = assert!(STREAM_AHEAD < CONV_SLOTS && STREAM_AHEAD < TOKEN_RING);

const NORM_EPS: f32 = 1e-5;
const ATTENTION_SCALE: f32 = 0.125;

// -------------------------------------------------------------------- config

#[derive(Deserialize)]
struct RopeParameters {
    rope_theta: f32,
}

#[derive(Deserialize)]
struct QuantEntry {
    group_size: usize,
    bits: usize,
}

#[derive(Deserialize)]
struct RawConfig {
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    moe_intermediate_size: usize,
    num_hidden_layers: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    num_dense_layers: usize,
    norm_eps: f32,
    #[serde(rename = "conv_L_cache")]
    conv_l_cache: usize,
    rope_parameters: RopeParameters,
    routed_scaling_factor: f32,
    tie_word_embeddings: bool,
    layer_types: Vec<String>,
    quantization: QuantEntry,
    #[serde(default)]
    quantization_config: Option<QuantEntry>,
    #[serde(default)]
    use_expert_bias: bool,
}

#[derive(Clone, Debug)]
pub struct LfmConfig {
    pub num_dense_layers: usize,
    pub max_position_embeddings: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub layer_types: Vec<bool>, // true = full attention
}

impl LfmConfig {
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let path = dir.as_ref().join("config.json");
        let bytes = std::fs::read(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let raw: RawConfig = serde_json::from_slice(&bytes).context("parsing config.json")?;
        ensure!(raw.model_type == "lfm2_moe", "unsupported model_type {:?}", raw.model_type);
        let q = raw.quantization_config.as_ref().unwrap_or(&raw.quantization);
        ensure!(
            q.group_size == 64 && q.bits == 4,
            "unsupported quantization: bits={} group={}",
            q.bits,
            q.group_size
        );
        ensure!(
            raw.vocab_size == VOCAB
                && raw.hidden_size == HIDDEN
                && raw.intermediate_size == DENSE_INTER
                && raw.moe_intermediate_size == MOE_INTER
                && raw.num_hidden_layers == N_LAYERS
                && raw.num_experts == N_EXPERTS
                && raw.num_experts_per_tok == TOP_K
                && raw.num_attention_heads == N_HEADS
                && raw.num_key_value_heads == N_KV_HEADS
                && raw.num_dense_layers == 2
                && raw.conv_l_cache == 3
                && raw.norm_topk_prob
                && raw.tie_word_embeddings
                && raw.use_expert_bias
                && raw.routed_scaling_factor == 1.0
                && (raw.norm_eps - NORM_EPS).abs() < 1e-12,
            "checkpoint is not the supported LFM2.5-24B-A2B shape"
        );
        ensure!(raw.layer_types.len() == N_LAYERS, "layer_types length");
        let mut layer_types = Vec::with_capacity(N_LAYERS);
        for (i, t) in raw.layer_types.iter().enumerate() {
            let full = t == "full_attention";
            ensure!(t == "conv" || full, "unknown layer type {t}");
            ensure!(full == (i % 4 == 2), "unexpected layer type pattern at {i}");
            layer_types.push(full);
        }
        Ok(Self {
            num_dense_layers: raw.num_dense_layers,
            max_position_embeddings: raw.max_position_embeddings,
            norm_eps: raw.norm_eps,
            rope_theta: raw.rope_parameters.rope_theta,
            layer_types,
        })
    }
}

// ------------------------------------------------------------------- weights

pub struct DenseMlp {
    pub gate: QuantWeights,
    pub up: QuantWeights,
    pub down: QuantWeights,
}

pub struct MoeFfn {
    /// F32 `[E, H]` router.
    pub router: Tensor,
    /// F32 `[E]` selection bias.
    pub expert_bias: Tensor,
    /// Stacked `[E * I, H]` and `[E * I, H]`, and `[E * H, I]`.
    pub gate: QuantWeights,
    pub up: QuantWeights,
    pub down: QuantWeights,
}

pub struct ConvMixer {
    pub in_proj: QuantWeights,
    pub out_proj: QuantWeights,
    /// BF16 `[H, 3]`, tap 0 oldest.
    pub conv_w: Tensor,
}

pub struct AttnMixer {
    pub q_proj: QuantWeights,
    pub k_proj: QuantWeights,
    pub v_proj: QuantWeights,
    pub out_proj: QuantWeights,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
}

pub enum Mixer {
    Conv(ConvMixer),
    Attn(AttnMixer),
}

pub enum Ffn {
    Dense(DenseMlp),
    Moe(MoeFfn),
}

pub struct LayerWeights {
    pub operator_norm: Tensor,
    pub ffn_norm: Tensor,
    pub mixer: Mixer,
    pub ffn: Ffn,
}

pub struct LfmWeights {
    pub embed: QuantWeights,
    pub final_norm: Tensor,
    pub layers: Vec<LayerWeights>,
}

fn load_tensor(ctx: &MetalContext, ckpt: &Checkpoint, name: &str) -> Result<Tensor> {
    let meta = ckpt.meta(name).with_context(|| format!("tensor {name} not in checkpoint"))?;
    let dtype = match &meta.dtype {
        SafetensorsDType::BF16 => DType::BF16,
        SafetensorsDType::F32 => DType::F32,
        SafetensorsDType::U32 => DType::U32,
        SafetensorsDType::Other(s) => anyhow::bail!("unsupported dtype {s} for {name}"),
    };
    let total: usize = meta.shape.iter().product();
    let buf = ctx.new_buffer(meta.byte_len())?;
    ckpt.read_with(name, |meta| {
        // SAFETY: freshly allocated buffer of exactly byte_len bytes.
        Ok(unsafe {
            core::slice::from_raw_parts_mut(
                buf.contents().as_ptr().cast::<u8>(),
                meta.byte_len(),
            )
        })
    })?;
    Tensor::from_buffer(buf, &[total], dtype)
}

fn load_quant(
    ctx: &MetalContext,
    ckpt: &Checkpoint,
    base: &str,
    expected_rows: usize,
    expected_k: usize,
    expected_bits: usize,
) -> Result<QuantWeights> {
    let wname = format!("{base}.weight");
    let smeta = ckpt.meta(&format!("{base}.scales")).with_context(|| format!("missing {base}.scales"))?;
    let bmeta = ckpt.meta(&format!("{base}.biases")).with_context(|| format!("missing {base}.biases"))?;
    let wmeta = ckpt.meta(&wname).with_context(|| format!("missing {wname}"))?;
    ensure!(wmeta.dtype == SafetensorsDType::U32, "{base} codes must be U32");
    let cols = *wmeta.shape.last().ok_or_else(|| anyhow!("{base}: empty shape"))?;
    ensure!(
        expected_k.is_multiple_of(64) && expected_k / 64 == *smeta.shape.last().unwrap_or(&0),
        "{base}: scales group count mismatch"
    );
    let bits = 32 * cols / expected_k;
    ensure!(bits == expected_bits, "{base}: expected {expected_bits}-bit, found {bits}");
    let rows: usize = wmeta.shape.iter().product::<usize>() / cols;
    ensure!(rows == expected_rows, "{base}: rows {rows} != {expected_rows}");
    let srows: usize = smeta.shape.iter().product::<usize>() / (expected_k / 64);
    let brows: usize = bmeta.shape.iter().product::<usize>() / (expected_k / 64);
    ensure!(srows == rows && brows == rows, "{base}: scale rows mismatch");
    let codes = load_tensor(ctx, ckpt, &wname)?;
    let scales = load_tensor(ctx, ckpt, &format!("{base}.scales"))?;
    let biases = load_tensor(ctx, ckpt, &format!("{base}.biases"))?;
    ensure!(scales.dtype() == DType::BF16 && biases.dtype() == DType::BF16, "{base}: scales dtype");
    let codes = codes.reshape(&[rows, cols])?;
    let scales = scales.reshape(&[rows, expected_k / 64])?;
    let biases = biases.reshape(&[rows, expected_k / 64])?;
    Ok(QuantWeights { codes, scales, biases, group_size: 64, bits })
}

impl LfmWeights {
    pub fn load(ctx: &MetalContext, dir: impl AsRef<Path>, cfg: &LfmConfig) -> Result<Self> {
        let ckpt = Checkpoint::open(&dir)?;
        let embed = load_quant(ctx, &ckpt, "model.embed_tokens", VOCAB, HIDDEN, 6)?;
        let final_norm = load_tensor(ctx, &ckpt, "model.embedding_norm.weight")?;
        ensure!(final_norm.numel() == HIDDEN, "final norm shape");
        let mut layers = Vec::with_capacity(N_LAYERS);
        for (idx, full) in cfg.layer_types.iter().enumerate() {
            let p = format!("model.layers.{idx}");
            let operator_norm = load_tensor(ctx, &ckpt, &format!("{p}.operator_norm.weight"))?;
            let ffn_norm = load_tensor(ctx, &ckpt, &format!("{p}.ffn_norm.weight"))?;
            ensure!(
                operator_norm.numel() == HIDDEN && ffn_norm.numel() == HIDDEN,
                "norm shape at layer {idx}"
            );
            let mixer = if *full {
                Mixer::Attn(AttnMixer {
                    q_proj: load_quant(ctx, &ckpt, &format!("{p}.self_attn.q_proj"), N_HEADS * HEAD_DIM, HIDDEN, 4)?,
                    k_proj: load_quant(ctx, &ckpt, &format!("{p}.self_attn.k_proj"), N_KV_HEADS * HEAD_DIM, HIDDEN, 4)?,
                    v_proj: load_quant(ctx, &ckpt, &format!("{p}.self_attn.v_proj"), N_KV_HEADS * HEAD_DIM, HIDDEN, 4)?,
                    out_proj: load_quant(ctx, &ckpt, &format!("{p}.self_attn.out_proj"), HIDDEN, N_HEADS * HEAD_DIM, 4)?,
                    q_norm: load_tensor(ctx, &ckpt, &format!("{p}.self_attn.q_layernorm.weight"))?,
                    k_norm: load_tensor(ctx, &ckpt, &format!("{p}.self_attn.k_layernorm.weight"))?,
                })
            } else {
                Mixer::Conv(ConvMixer {
                    in_proj: load_quant(ctx, &ckpt, &format!("{p}.conv.in_proj"), 3 * HIDDEN, HIDDEN, 4)?,
                    out_proj: load_quant(ctx, &ckpt, &format!("{p}.conv.out_proj"), HIDDEN, HIDDEN, 4)?,
                    conv_w: load_tensor(ctx, &ckpt, &format!("{p}.conv.conv.weight"))?,
                })
            };
            let ffn = if idx < cfg.num_dense_layers {
                Ffn::Dense(DenseMlp {
                    gate: load_quant(ctx, &ckpt, &format!("{p}.feed_forward.gate_proj"), DENSE_INTER, HIDDEN, 4)?,
                    up: load_quant(ctx, &ckpt, &format!("{p}.feed_forward.up_proj"), DENSE_INTER, HIDDEN, 4)?,
                    down: load_quant(ctx, &ckpt, &format!("{p}.feed_forward.down_proj"), HIDDEN, DENSE_INTER, 4)?,
                })
            } else {
                let moe = &format!("{p}.feed_forward");
                let router = load_tensor(ctx, &ckpt, &format!("{moe}.gate.weight"))?
                    .reshape(&[N_EXPERTS, HIDDEN])?;
                let expert_bias = load_tensor(ctx, &ckpt, &format!("{moe}.expert_bias"))?
                    .reshape(&[N_EXPERTS])?;
                Ffn::Moe(MoeFfn {
                    router,
                    expert_bias,
                    gate: load_quant(ctx, &ckpt, &format!("{moe}.switch_mlp.gate_proj"), N_EXPERTS * MOE_INTER, HIDDEN, 4)?,
                    up: load_quant(ctx, &ckpt, &format!("{moe}.switch_mlp.up_proj"), N_EXPERTS * MOE_INTER, HIDDEN, 4)?,
                    down: load_quant(ctx, &ckpt, &format!("{moe}.switch_mlp.down_proj"), N_EXPERTS * HIDDEN, MOE_INTER, 4)?,
                })
            };
            layers.push(LayerWeights { operator_norm, ffn_norm, mixer, ffn });
        }
        ensure!(ckpt.names().count() == 870, "unexpected checkpoint tensor count");
        Ok(Self { embed, final_norm, layers })
    }
}

// --------------------------------------------------------------------- state

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    TeacherForced,
    Greedy,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::TeacherForced => "teacher_forced",
            Mode::Greedy => "greedy",
        }
    }
}

pub struct LfmState {
    pub mode: Mode,
    pub pos: usize,
    pub pending: Option<u32>,
    pub logits: BTreeMap<usize, Vec<f32>>,
    /// Per conv layer: `CONV_SLOTS` `[2, H]` BF16 state buffers; slot
    /// `conv_slot` holds the current state.
    pub conv: Vec<Option<Vec<Tensor>>>,
    pub conv_slot: usize,
    /// Per layer: `([KVH, max_seq, D], [KVH, max_seq, D])`.
    pub kv: Vec<Option<(Tensor, Tensor)>>,
    /// Bitmap of the distinct token ids of this sequence (prompt and output
    /// tokens of every `generate` since the reset), which the repetition
    /// penalty reads.
    pub seen: Vec<u32>,
    max_seq: usize,
}

impl LfmState {
    pub fn new(ctx: &MetalContext, max_seq: usize) -> Result<Self> {
        let mut conv = Vec::with_capacity(N_LAYERS);
        let mut kv = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            if i % 4 == 2 {
                conv.push(None);
                kv.push(Some((
                    Tensor::zeros(ctx, &[N_KV_HEADS, max_seq, HEAD_DIM], DType::BF16)?,
                    Tensor::zeros(ctx, &[N_KV_HEADS, max_seq, HEAD_DIM], DType::BF16)?,
                )));
            } else {
                conv.push(Some(
                    (0..CONV_SLOTS)
                        .map(|_| Tensor::zeros(ctx, &[2, HIDDEN], DType::BF16))
                        .collect::<Result<Vec<_>>>()?,
                ));
                kv.push(None);
            }
        }
        Ok(Self {
            mode: Mode::TeacherForced,
            pos: 0,
            pending: None,
            logits: BTreeMap::new(),
            conv,
            conv_slot: 0,
            kv,
            seen: vec![0; sample::SEEN_WORDS],
            max_seq,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        for s in self.conv.iter_mut().flatten() {
            for t in s.iter_mut() {
                t.zero_fill();
            }
        }
        self.pos = 0;
        self.pending = None;
        self.logits.clear();
        self.conv_slot = 0;
        self.seen.fill(0);
        self.mode = Mode::TeacherForced;
        Ok(())
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// The current state buffer of conv layer `idx` and the one the next
    /// token (or chunk) writes.
    fn conv_pair(&self, idx: usize) -> (Tensor, Tensor) {
        let slots = self.conv[idx].as_ref().expect("conv state");
        (
            slots[self.conv_slot].clone(),
            slots[(self.conv_slot + 1) % CONV_SLOTS].clone(),
        )
    }

    fn advance_conv(&mut self) {
        self.conv_slot = (self.conv_slot + 1) % CONV_SLOTS;
    }

    /// Steps the conv state back by `steps` tokens; their states are still in
    /// the slot ring.
    fn rewind_conv(&mut self, steps: usize) {
        debug_assert!(steps < CONV_SLOTS);
        self.conv_slot = (self.conv_slot + CONV_SLOTS - steps % CONV_SLOTS) % CONV_SLOTS;
    }

    pub fn state_json(&self) -> serde_json::Value {
        serde_json::json!({
            "mode": self.mode.as_str(),
            "consumed_tokens": self.pos,
            "pending_token": self.pending,
            "ended": false,
        })
    }
}

// ------------------------------------------------------------------- scratch

struct MoeScratch {
    router: Tensor,
    gx: Tensor,
    gate: Tensor,
    up: Tensor,
    act: Tensor,
    ed: Tensor,
    ids: Tensor,
    counts: Tensor,
    offsets: Tensor,
    ids_sorted: Tensor,
    slotpos: Tensor,
    gw: Tensor,
    route_ids: Tensor,
    route_w: Tensor,
    router1: Tensor,
    block_gu_off: Tensor,
    block_dn_off: Tensor,
}

pub struct Scratch {
    cap: usize,
    x: Tensor,
    /// Second residual buffer for the decode step's fused router norm, which
    /// writes the updated residual to a buffer other than the one it reads.
    x_alt: Tensor,
    normed: Tensor,
    branch: Tensor,
    bcx: Tensor,
    small_a: Tensor,
    q: Tensor,
    qn: Tensor,
    k: Tensor,
    kn: Tensor,
    v: Tensor,
    attn_o: Tensor,
    attn_partials: Tensor,
    attn_sums: Tensor,
    attn_maxs: Tensor,
    ffn_gate: Tensor,
    ffn_up: Tensor,
    ffn_act: Tensor,
    final_h: Tensor,
    ids: Tensor,
    gen_tokens: Tensor,
    logits_rows: Tensor,
    logits1: Tensor,
    argmax_partials: Tensor,
    /// The decode head's 64-bit argmax key (zero between steps).
    argmax_key: Tensor,
    dequant: Tensor,
    blocks_gu: Tensor,
    blocks_dn: Tensor,
    moe: MoeScratch,
    /// Zeroed U32 completion counters for the decode kernels whose last
    /// finishing threadgroup runs an epilogue (word 0: the router top-k).
    /// Each kernel resets its word before the dispatch ends.
    fence_counter: Tensor,
    sample: SampleScratch,
}

impl Scratch {
    fn new(ctx: &MetalContext, cap: usize) -> Result<Self> {
        let bf = DType::BF16;
        let s = cap * TOP_K;
        let dequant_elems = (DENSE_INTER * HIDDEN).max(s * MOE_INTER);
        Ok(Self {
            cap,
            x: Tensor::zeros(ctx, &[cap, HIDDEN], bf)?,
            x_alt: Tensor::zeros(ctx, &[cap, HIDDEN], bf)?,
            normed: Tensor::zeros(ctx, &[cap, HIDDEN], bf)?,
            branch: Tensor::zeros(ctx, &[cap, HIDDEN], bf)?,
            bcx: Tensor::zeros(ctx, &[cap, 3 * HIDDEN], bf)?,
            small_a: Tensor::zeros(ctx, &[cap, N_HEADS * HEAD_DIM], bf)?,
            q: Tensor::zeros(ctx, &[cap, N_HEADS * HEAD_DIM], bf)?,
            qn: Tensor::zeros(ctx, &[cap, N_HEADS * HEAD_DIM], bf)?,
            k: Tensor::zeros(ctx, &[cap, N_KV_HEADS * HEAD_DIM], bf)?,
            kn: Tensor::zeros(ctx, &[cap, N_KV_HEADS * HEAD_DIM], bf)?,
            v: Tensor::zeros(ctx, &[cap, N_KV_HEADS * HEAD_DIM], bf)?,
            attn_o: Tensor::zeros(ctx, &[cap, N_HEADS * HEAD_DIM], bf)?,
            attn_partials: Tensor::zeros(
                ctx,
                &[N_HEADS * klfm::SDPA_2PASS_MAX_BLOCKS * HEAD_DIM],
                bf,
            )?,
            attn_sums: Tensor::zeros(
                ctx,
                &[N_HEADS * klfm::SDPA_2PASS_MAX_BLOCKS],
                DType::F32,
            )?,
            attn_maxs: Tensor::zeros(
                ctx,
                &[N_HEADS * klfm::SDPA_2PASS_MAX_BLOCKS],
                DType::F32,
            )?,
            ffn_gate: Tensor::zeros(ctx, &[cap, DENSE_INTER], bf)?,
            ffn_up: Tensor::zeros(ctx, &[cap, DENSE_INTER], bf)?,
            ffn_act: Tensor::zeros(ctx, &[cap, DENSE_INTER], bf)?,
            final_h: Tensor::zeros(ctx, &[cap, HIDDEN], bf)?,
            ids: Tensor::zeros(ctx, &[cap], DType::U32)?,
            gen_tokens: Tensor::zeros(ctx, &[TOKEN_RING], DType::U32)?,
            logits_rows: Tensor::zeros(ctx, &[64, VOCAB], bf)?,
            logits1: Tensor::zeros(ctx, &[VOCAB], bf)?,
            argmax_partials: Tensor::zeros(ctx, &[klfm::ARGMAX_GROUPS * 2], DType::U32)?,
            argmax_key: Tensor::zeros(ctx, &[2], DType::U32)?,
            dequant: Tensor::zeros(ctx, &[dequant_elems], bf)?,
            blocks_gu: Tensor::zeros(ctx, &[BLOCK_MAP_WORDS], DType::U32)?,
            blocks_dn: Tensor::zeros(ctx, &[BLOCK_MAP_WORDS], DType::U32)?,
            fence_counter: Tensor::zeros(ctx, &[2], DType::U32)?,
            sample: SampleScratch::new(ctx)?,
            moe: MoeScratch {
                router: Tensor::zeros(ctx, &[cap, N_EXPERTS], DType::F32)?,
                gx: Tensor::zeros(ctx, &[s, HIDDEN], bf)?,
                gate: Tensor::zeros(ctx, &[s, MOE_INTER], bf)?,
                up: Tensor::zeros(ctx, &[s, MOE_INTER], bf)?,
                act: Tensor::zeros(ctx, &[s, MOE_INTER], bf)?,
                ed: Tensor::zeros(ctx, &[s, HIDDEN], bf)?,
                ids: Tensor::zeros(ctx, &[s], DType::U32)?,
                counts: Tensor::zeros(ctx, &[N_EXPERTS], DType::U32)?,
                offsets: Tensor::zeros(ctx, &[N_EXPERTS], DType::U32)?,
                ids_sorted: Tensor::zeros(ctx, &[s], DType::U32)?,
                slotpos: Tensor::zeros(ctx, &[cap * TOP_K], DType::U32)?,
                gw: Tensor::zeros(ctx, &[cap * TOP_K], bf)?,
                route_ids: Tensor::zeros(ctx, &[TOP_K], DType::U32)?,
                route_w: Tensor::zeros(ctx, &[TOP_K], bf)?,
                router1: Tensor::zeros(ctx, &[N_EXPERTS], DType::F32)?,
                block_gu_off: Tensor::zeros(ctx, &[N_EXPERTS], DType::U32)?,
                block_dn_off: Tensor::zeros(ctx, &[N_EXPERTS], DType::U32)?,
            },
        })
    }

    fn view_rows(t: &Tensor, rows: usize) -> Result<Tensor> {
        let mut shape = t.shape().to_vec();
        shape[0] = rows;
        t.view(0, &shape)
    }
}

// --------------------------------------------------------------------- model

pub struct LfmModel {
    pub cfg: LfmConfig,
    pub weights: LfmWeights,
}

impl LfmModel {
    pub fn load(ctx: &MetalContext, dir: impl AsRef<Path>) -> Result<Self> {
        let cfg = LfmConfig::from_model_dir(&dir)?;
        let weights = LfmWeights::load(ctx, &dir, &cfg)?;
        Ok(Self { cfg, weights })
    }

    fn project(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        x: &Tensor,
        w: &QuantWeights,
        out: &Tensor,
        scratch: &Scratch,
        m: usize,
    ) -> Result<()> {
        if m == 1 {
            klfm::gemv_quant(ctx, pass, w, x, out)
        } else if w.bits == 4 && w.group_size == 64 {
            // Materialize this projection's weights once per chunk and run the
            // BF16 tensor-op GEMM: a chunk has many 64-row tiles, and the fused
            // Q4 path re-dequantizes the same B tile for every one of them. The
            // dequantized values are the same BF16 values, so the result is
            // unchanged.
            let n = w.out_features();
            let k = w.in_features();
            let staged = scratch.dequant.view(0, &[n, k])?;
            quant::dequant_q4_bf16_bulk(ctx, pass, w, &staged)?;
            crate::kernels::gemm::gemm_bf16_nt(ctx, pass, x, &staged, out)
        } else {
            quant::gemm_quant_bf16_nt(ctx, pass, x, w, out, &scratch.dequant)
        }
    }

    fn rmsnorm(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        x: &Tensor,
        w: &Tensor,
        out: &Tensor,
    ) -> Result<()> {
        norm::rmsnorm_bf16(ctx, pass, x, w, out, self.cfg.norm_eps, 0.0)
    }

    /// Fused BF16 residual add and RMSNorm: `x += b; out = rmsnorm(x) * w`.
    fn add_rmsnorm(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        x: &Tensor,
        b: &Tensor,
        w: &Tensor,
        out: &Tensor,
    ) -> Result<()> {
        norm::add_rmsnorm_bf16(ctx, pass, x, b, w, out, self.cfg.norm_eps, 0.0)
    }

    fn embed(&self, ctx: &MetalContext, pass: &ComputePass<'_>, ids: &Tensor, out: &Tensor) -> Result<()> {
        klfm::gather_rows_q6(ctx, pass, &self.weights.embed, ids, out)
    }

    /// Runs all 40 layers over `m` tokens, leaving the final RMSNorm in
    /// `scratch.final_h`. `state.pos` must already equal the chunk's base
    /// position; it is advanced by `m` before returning.
    fn forward_chunk(
        &self,
        ctx: &MetalContext,
        state: &mut LfmState,
        scratch: &mut Scratch,
        tokens: &[u32],
    ) -> Result<()> {
        let m = tokens.len();
        ensure!(m > 0 && m <= scratch.cap, "bad chunk length {m}");
        ensure!(
            state.pos + m <= state.max_seq,
            "context length {} exceeds {}",
            state.pos + m,
            state.max_seq
        );
        let ids = Scratch::view_rows(&scratch.ids, m)?;
        ids.write_bytes(bytemuck::cast_slice(tokens))?;
        let h = Scratch::view_rows(&scratch.x, m)?;
        let normed = Scratch::view_rows(&scratch.normed, m)?;
        let branch = Scratch::view_rows(&scratch.branch, m)?;

        let pass = ctx.begin()?;
        self.embed(ctx, &pass, &ids, &h)?;
        let final_h = Scratch::view_rows(&scratch.final_h, m)?;

        let base_pos = state.pos;
        for (idx, layer) in self.weights.layers.iter().enumerate() {
            // The previous iteration already produced this layer's operator
            // norm together with the residual add.
            if idx == 0 {
                self.rmsnorm(ctx, &pass, &h, &layer.operator_norm, &normed)?;
            }
            match &layer.mixer {
                Mixer::Conv(c) => {
                    let bcx = Scratch::view_rows(&scratch.bcx, m)?;
                    self.project(ctx, &pass, &normed, &c.in_proj, &bcx, scratch, m)?;
                    let (cur, next) = state.conv_pair(idx);
                    let conv_out = Scratch::view_rows(&scratch.small_a, m)?;
                    klfm::conv3_gate_bcx(
                        ctx,
                        &pass,
                        &bcx,
                        &c.conv_w,
                        &cur,
                        &conv_out,
                        &next,
                    )?;
                    self.project(ctx, &pass, &conv_out, &c.out_proj, &branch, scratch, m)?;
                }
                Mixer::Attn(a) => {
                    let q = Scratch::view_rows(&scratch.q, m)?;
                    let k = Scratch::view_rows(&scratch.k, m)?;
                    let v = Scratch::view_rows(&scratch.v, m)?;
                    self.project(ctx, &pass, &normed, &a.q_proj, &q, scratch, m)?;
                    self.project(ctx, &pass, &normed, &a.k_proj, &k, scratch, m)?;
                    self.project(ctx, &pass, &normed, &a.v_proj, &v, scratch, m)?;
                    let qn = Scratch::view_rows(&scratch.qn, m)?;
                    let kn = Scratch::view_rows(&scratch.kn, m)?;
                    let qrows = q.reshape(&[m * N_HEADS, HEAD_DIM])?;
                    let qnrows = qn.reshape(&[m * N_HEADS, HEAD_DIM])?;
                    let krows = k.reshape(&[m * N_KV_HEADS, HEAD_DIM])?;
                    let knrows = kn.reshape(&[m * N_KV_HEADS, HEAD_DIM])?;
                    klfm::rmsnorm64(ctx, &pass, &qrows, &a.q_norm, &qnrows, self.cfg.norm_eps)?;
                    klfm::rmsnorm64(ctx, &pass, &krows, &a.k_norm, &knrows, self.cfg.norm_eps)?;
                    attention::rope_neox(ctx, &pass, &qnrows, N_HEADS, HEAD_DIM, base_pos, self.cfg.rope_theta)?;
                    attention::rope_neox(ctx, &pass, &knrows, N_KV_HEADS, HEAD_DIM, base_pos, self.cfg.rope_theta)?;
                    let (kc, vc) = state.kv[idx].as_ref().expect("kv cache");
                    attention::scatter_kv(ctx, &pass, kc, &kn, base_pos)?;
                    attention::scatter_kv(ctx, &pass, vc, &v, base_pos)?;
                    let attn_o = Scratch::view_rows(&scratch.attn_o, m)?;
                    klfm::sdpa(
                        ctx,
                        &pass,
                        &qn,
                        kc,
                        vc,
                        &attn_o,
                        base_pos,
                        m,
                        N_HEADS,
                        N_HEADS / N_KV_HEADS,
                        ATTENTION_SCALE,
                        Some((
                            &scratch.attn_partials,
                            &scratch.attn_sums,
                            &scratch.attn_maxs,
                        )),
                    )?;
                    self.project(ctx, &pass, &attn_o, &a.out_proj, &branch, scratch, m)?;
                }
            }
            self.add_rmsnorm(ctx, &pass, &h, &branch, &layer.ffn_norm, &normed)?;
            match &layer.ffn {
                Ffn::Dense(f) => {
                    let gate = Scratch::view_rows(&scratch.ffn_gate, m)?;
                    let up = Scratch::view_rows(&scratch.ffn_up, m)?;
                    let act = Scratch::view_rows(&scratch.ffn_act, m)?;
                    self.project(ctx, &pass, &normed, &f.gate, &gate, scratch, m)?;
                    self.project(ctx, &pass, &normed, &f.up, &up, scratch, m)?;
                    elementwise::silu_mul_bf16(ctx, &pass, &gate, &up, &act)?;
                    self.project(ctx, &pass, &act, &f.down, &branch, scratch, m)?;
                }
                Ffn::Moe(f) => {
                    if m == 1 {
                        self.moe_decode(ctx, &pass, scratch, &normed, &branch, f, false)?;
                    } else {
                        self.moe_prefill(ctx, &pass, state, scratch, &normed, &branch, f, m)?;
                    }
                }
            }
            if let Some(next) = self.weights.layers.get(idx + 1) {
                self.add_rmsnorm(
                    ctx,
                    &pass,
                    &h,
                    &branch,
                    &next.operator_norm,
                    &normed,
                )?;
            } else {
                self.add_rmsnorm(
                    ctx,
                    &pass,
                    &h,
                    &branch,
                    &self.weights.final_norm,
                    &final_h,
                )?;
            }
        }

        pass.commit_wait()?;
        state.advance_conv();
        state.pos += m;
        Ok(())
    }

    /// MoE block for a prefill chunk, entirely device-resident: GPU top-k,
    /// deterministic counting sort and block maps, grouped expert GEMMs.
    fn moe_prefill(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        _state: &mut LfmState,
        scratch: &mut Scratch,
        normed: &Tensor,
        out: &Tensor,
        f: &MoeFfn,
        m: usize,
    ) -> Result<()> {
        let s = m * TOP_K;
        let router = Scratch::view_rows(&scratch.moe.router, m)?;
        let ids = Scratch::view_rows(&scratch.moe.ids, s)?;
        let gw = Scratch::view_rows(&scratch.moe.gw, s)?;
        let ids_sorted = Scratch::view_rows(&scratch.moe.ids_sorted, s)?;
        let slotpos = Scratch::view_rows(&scratch.moe.slotpos, s)?;
        klfm::router_prefill(ctx, pass, normed, &f.router, &router)?;
        klfm::moe_route_batch(ctx, pass, &router, &f.expert_bias, &ids, &gw, m)?;
        klfm::moe_zero_counts(ctx, pass, &scratch.moe.counts)?;
        klfm::moe_count_slots(ctx, pass, &ids, &scratch.moe.counts, s)?;
        klfm::moe_offsets(ctx, pass, &scratch.moe.counts, &scratch.moe.offsets)?;
        klfm::moe_sorted(
            ctx,
            pass,
            &ids,
            &scratch.moe.offsets,
            &ids_sorted,
            &slotpos,
            m,
            TOP_K,
        )?;

        // The reference partition caps a prompt chunk at 2048 rows. At 512
        // rows or fewer every expert owns at most 32 routed rows, so the
        // 32-row tile removes the 64-row tile's padding without the 16-row
        // tile's lower tensor-op efficiency. Larger chunks use the 128-row
        // shared-B tile: its threadgroup stages one 64x64 weight tile and
        // feeds both 64-row sub-tiles, halving the weight staging for the
        // ~128 routed rows an expert owns at a 2048-row chunk. The tile
        // changes grouping, not the per-output K order.
        // One 64-column stripe per threadgroup (`T128N64Sg4Wn1`): the same
        // 128-row block tile and the same MLX `qmm_t` body, but the staged
        // activation tile is read once per tile rather than once per column
        // stripe, at twice the tensor-op tiles per simdgroup. Paired
        // same-process A/B against the two-stripe tile measured it faster at
        // every chunk length from 1023 to 32767 rows with identical tokens, so
        // every chunk above 512 rows uses it. Chunks of at most 512 rows keep
        // the 32-row tile.
        let tile = if m <= 512 {
            quant::MoeTile::T32Sg4
        } else {
            quant::MoeTile::T128N64Sg4Wn1
        };
        let bm = tile.rows();
        let bn = tile.cols();
        let m_tiles = s.div_ceil(bm) + N_EXPERTS;
        let max_gu = m_tiles * (MOE_INTER / bn);
        let max_dn = m_tiles * (HIDDEN / bn);
        let gu_map = Scratch::view_rows(&scratch.blocks_gu, max_gu * 4)?;
        let dn_map = Scratch::view_rows(&scratch.blocks_dn, max_dn * 4)?;
        klfm::moe_zero_blocks(ctx, pass, &gu_map, &dn_map, max_gu, max_dn)?;
        klfm::moe_block_offsets(
            ctx,
            pass,
            &scratch.moe.counts,
            &scratch.moe.block_gu_off,
            &scratch.moe.block_dn_off,
            bm,
            bn,
            MOE_INTER,
            HIDDEN,
        )?;
        klfm::moe_blocks(
            ctx,
            pass,
            &scratch.moe.counts,
            &scratch.moe.offsets,
            &scratch.moe.block_gu_off,
            &scratch.moe.block_dn_off,
            &gu_map,
            &dn_map,
            bm,
            bn,
            MOE_INTER,
            HIDDEN,
        )?;

        let gx = Scratch::view_rows(&scratch.moe.gx, s)?;
        elementwise::gather_rows_bf16(ctx, pass, normed, &ids_sorted, &gx)?;
        let gate = Scratch::view_rows(&scratch.moe.gate, s)?;
        let up = Scratch::view_rows(&scratch.moe.up, s)?;
        let act = Scratch::view_rows(&scratch.moe.act, s)?;
        let ed_all = Scratch::view_rows(&scratch.moe.ed, s)?;
        // MLX's qmm_t NAX body over this engine's block map: same block map,
        // same dequantized BF16 values and same outputs, with MLX's padded
        // threadgroup B staging and SK=32 tile_matmad_nax loop.
        // At a chunk of at most 512 rows the two 32-row expert GEMMs share
        // the block map, the activation rows and the K order. One
        // threadgroup stages both weight tiles and loads each A slice once,
        // so gate and up share activation traffic. Both accumulators run
        // MLX's per-slice MMA, so the values are the separate dispatches'
        // bitwise results. The 128-row shared-B tile keeps the two separate
        // dispatches: doubling its threadgroup B staging to 18 KiB costs more
        // occupancy than the shared A load saves (measured slower at 1024 and
        // 2048 rows).
        if tile == quant::MoeTile::T32Sg4 {
            quant::gemm_q4_grouped_mlx_gu(
                ctx, pass, &gx, &f.gate, &f.up, &gate, &up, &gu_map, max_gu, tile,
            )?;
        } else {
            quant::gemm_q4_grouped_mlx(ctx, pass, &gx, &f.gate, &gate, &gu_map, max_gu, tile)?;
            quant::gemm_q4_grouped_mlx(ctx, pass, &gx, &f.up, &up, &gu_map, max_gu, tile)?;
        }
        elementwise::silu_mul_bf16(ctx, pass, &gate, &up, &act)?;
        quant::gemm_q4_grouped_mlx(ctx, pass, &act, &f.down, &ed_all, &dn_map, max_dn, tile)?;
        klfm::moe_combine_rows(ctx, pass, &ed_all, &slotpos, &gw, out, TOP_K)?;
        Ok(())
    }

    fn moe_decode(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        scratch: &mut Scratch,
        normed: &Tensor,
        out: &Tensor,
        f: &MoeFfn,
        routed: bool,
    ) -> Result<()> {
        if !routed {
            // The single-token prompt chunk: router GEMV and the warp top-k as
            // separate dispatches. The decode step arrives with the routing
            // already selected by its fused router dispatch.
            klfm::gemv_f32(ctx, pass, &f.router, normed, &scratch.moe.router1)?;
            klfm::route_token(
                ctx,
                pass,
                &scratch.moe.router1,
                &f.expert_bias,
                &scratch.moe.route_ids,
                &scratch.moe.route_w,
            )?;
        }
        let act = Scratch::view_rows(&scratch.moe.act, TOP_K)?;
        klfm::gemv_gather_act(
            ctx,
            pass,
            &f.gate,
            &f.up,
            normed,
            &scratch.moe.route_ids,
            &act,
        )?;
        klfm::gemv_gather_down_combine4(
            ctx,
            pass,
            &f.down,
            &act,
            &scratch.moe.route_ids,
            &scratch.moe.route_w,
            out,
        )?;
        Ok(())
    }

    /// One decode step for `in_tok`, leaving the next token in `out_tok`:
    /// the greedy argmax, or a draw when `sampler` is set. With `embed_in`
    /// false the previous step's tail already left this token's embedding in
    /// `scratch.x` and its first RMSNorm in `scratch.normed`; with
    /// `embed_next` the tail does that for `out_tok`.
    #[allow(clippy::too_many_arguments)]
    fn decode_step_into(
        &self,
        ctx: &MetalContext,
        state: &mut LfmState,
        scratch: &mut Scratch,
        pass: &ComputePass<'_>,
        in_tok: &Tensor,
        out_tok: &Tensor,
        embed_in: bool,
        embed_next: bool,
        sampler: Option<&Sampling>,
    ) -> Result<()> {
        ensure!(state.pos < state.max_seq, "context length exceeded");
        let ids = in_tok;
        let mut h = Scratch::view_rows(&scratch.x, 1)?;
        let mut h_alt = Scratch::view_rows(&scratch.x_alt, 1)?;
        let normed = Scratch::view_rows(&scratch.normed, 1)?;
        let branch = Scratch::view_rows(&scratch.branch, 1)?;
        if embed_in {
            self.embed(ctx, pass, &ids, &h)?;
        }
        let final_h = Scratch::view_rows(&scratch.final_h, 1)?;
        let base_pos = state.pos;
        for (idx, layer) in self.weights.layers.iter().enumerate() {
            if idx == 0 && embed_in {
                self.rmsnorm(ctx, pass, &h, &layer.operator_norm, &normed)?;
            }
            match &layer.mixer {
                Mixer::Conv(c) => {
                    // Decode fuses the B/C/X projection with the conv: one warp
                    // owns a channel's three rows, so the dot products, bfloat
                    // roundings and tap order are those of the separate in_proj
                    // GEMV and conv3_gate_bcx.
                    let (cur, next) = state.conv_pair(idx);
                    let conv_out = Scratch::view_rows(&scratch.small_a, 1)?;
                    klfm::conv3_gemv(
                        ctx,
                        pass,
                        &c.in_proj,
                        &normed,
                        &c.conv_w,
                        &cur,
                        &conv_out,
                        &next,
                    )?;
                    self.project(ctx, pass, &conv_out, &c.out_proj, &branch, scratch, 1)?;
                }
                Mixer::Attn(a) => {
                    let q = Scratch::view_rows(&scratch.q, 1)?;
                    let k = Scratch::view_rows(&scratch.k, 1)?;
                    let (kc, vc) = state.kv[idx].as_ref().expect("kv cache");
                    // Q/K/V projections in one dispatch, V straight to the
                    // cache; then the two per-head norm+RoPE pairs, K written
                    // to the cache. Each row's dot product and each rounding
                    // boundary is the separate-kernel path's.
                    klfm::proj3_gemv_scatter(
                        ctx,
                        pass,
                        &a.q_proj,
                        &a.k_proj,
                        &a.v_proj,
                        &normed,
                        &q,
                        &k,
                        vc,
                        base_pos,
                    )?;
                    let attn_o = Scratch::view_rows(&scratch.attn_o, 1)?;
                    if base_pos + 1 < klfm::SDPA_2PASS_MIN_LEN {
                        // Single-pass range: the per-head norm + RoPE and the
                        // K cache write run in the attention kernel's
                        // prologue; the values are the separate kernels'.
                        klfm::sdpa_vector_norm_rope(
                            ctx,
                            pass,
                            &q,
                            &a.q_norm,
                            &k,
                            &a.k_norm,
                            kc,
                            vc,
                            &attn_o,
                            base_pos,
                            self.cfg.rope_theta,
                            N_HEADS,
                            N_HEADS / N_KV_HEADS,
                            ATTENTION_SCALE,
                        )?;
                    } else {
                        let qn = Scratch::view_rows(&scratch.qn, 1)?;
                        // Per-head norm + RoPE for Q and K plus the K cache
                        // write in one dispatch; the values are the separate
                        // kernels'.
                        klfm::norm_rope_qk_scatter(
                            ctx,
                            pass,
                            &q,
                            &a.q_norm,
                            &qn,
                            &k,
                            &a.k_norm,
                            kc,
                            base_pos,
                            self.cfg.rope_theta,
                            N_HEADS,
                            N_KV_HEADS,
                        )?;
                        klfm::sdpa(
                            ctx,
                            pass,
                            &qn,
                            kc,
                            vc,
                            &attn_o,
                            base_pos,
                            1,
                            N_HEADS,
                            N_HEADS / N_KV_HEADS,
                            ATTENTION_SCALE,
                            Some((
                                &scratch.attn_partials,
                                &scratch.attn_sums,
                                &scratch.attn_maxs,
                            )),
                        )?;
                    }
                    self.project(ctx, pass, &attn_o, &a.out_proj, &branch, scratch, 1)?;
                }
            }
            if matches!(layer.ffn, Ffn::Moe(_)) {
                let Ffn::Moe(f) = &layer.ffn else { unreachable!() };
                // Residual add + RMSNorm, the router GEMV and the top-k in one
                // dispatch. The updated residual goes to the other h buffer;
                // group 0 writes it and `normed` for the gather dispatches, the
                // last-finishing group writes the routing.
                klfm::router_addnorm_route(
                    ctx,
                    pass,
                    &h,
                    &branch,
                    &layer.ffn_norm,
                    &h_alt,
                    &normed,
                    &f.router,
                    &scratch.moe.router1,
                    &f.expert_bias,
                    &scratch.moe.route_ids,
                    &scratch.moe.route_w,
                    &scratch.fence_counter.view(0, &[1])?,
                    self.cfg.norm_eps,
                )?;
                std::mem::swap(&mut h, &mut h_alt);
            } else {
                self.add_rmsnorm(ctx, pass, &h, &branch, &layer.ffn_norm, &normed)?;
            }
            match &layer.ffn {
                Ffn::Dense(f) => {
                    // Gate, up and SwiGLU in one dispatch; the values are the
                    // two GEMVs' and silu_mul_bf16's.
                    let act = Scratch::view_rows(&scratch.ffn_act, 1)?;
                    klfm::gemv_gu_act(ctx, pass, &f.gate, &f.up, &normed, &act)?;
                    self.project(ctx, pass, &act, &f.down, &branch, scratch, 1)?;
                }
                Ffn::Moe(f) => {
                    self.moe_decode(ctx, pass, scratch, &normed, &branch, f, true)?;
                }
            }
            if let Some(next) = self.weights.layers.get(idx + 1) {
                self.add_rmsnorm(
                    ctx,
                    pass,
                    &h,
                    &branch,
                    &next.operator_norm,
                    &normed,
                )?;
            } else {
                self.add_rmsnorm(
                    ctx,
                    pass,
                    &h,
                    &branch,
                    &self.weights.final_norm,
                    &final_h,
                )?;
            }
        }
        // Greedy decode logits are never read: the head folds its argmax into
        // one key and the tail turns it into the token (and, unless this is
        // the call's last step, the next step's embedding and first norm). A
        // sampled step stores the logits with the plain head GEMV (the same
        // values) and draws the token into the same key.
        match sampler {
            None => klfm::gemv_head_argmax(
                ctx,
                pass,
                &self.weights.embed,
                &final_h,
                &scratch.argmax_key,
            )?,
            Some(s) => {
                klfm::gemv_quant(ctx, pass, &self.weights.embed, &final_h, &scratch.logits1)?;
                scratch.sample.encode_draw(
                    ctx,
                    pass,
                    &scratch.logits1,
                    &scratch.argmax_key,
                    s,
                    base_pos + 1,
                )?;
            }
        }
        let x0 = Scratch::view_rows(&scratch.x, 1)?;
        let next = if embed_next {
            Some((&self.weights.embed, &self.weights.layers[0].operator_norm, &x0, &normed))
        } else {
            None
        };
        klfm::token_tail(ctx, pass, &scratch.argmax_key, out_tok, next, self.cfg.norm_eps)?;
        if sampler.is_some_and(Sampling::uses_penalty) {
            scratch.sample.encode_mark(ctx, pass, out_tok)?;
        }
        Ok(())
    }

    /// Retains the requested logits rows for the latest forward.
    fn retain_logits(
        &self,
        ctx: &MetalContext,
        scratch: &mut Scratch,
        positions: &[usize],
        base_pos: usize,
        m: usize,
    ) -> Result<Vec<(usize, Vec<f32>)>> {
        // A request may name a row by its input position (row t predicts
        // token t+1) or by the predicted token's position (t+1); accept both
        // for the frontier row, the one that predicts the next token.
        let wanted: Vec<(usize, usize)> = positions
            .iter()
            .copied()
            .filter_map(|p| {
                if p >= base_pos && p < base_pos + m {
                    Some((p, p - base_pos))
                } else if p > base_pos && p <= base_pos + m {
                    Some((p, p - 1 - base_pos))
                } else {
                    None
                }
            })
            .collect();
        if wanted.is_empty() {
            return Ok(Vec::new());
        }
        ensure!(wanted.len() <= 64, "too many logits rows");
        let pass = ctx.begin()?;
        for (i, (_p, r)) in wanted.iter().enumerate() {
            let row = scratch.final_h.view(*r * HIDDEN, &[1, HIDDEN])?;
            let dst = scratch.logits_rows.view(i * VOCAB, &[VOCAB])?;
            klfm::gemv_quant(ctx, &pass, &self.weights.embed, &row, &dst)?;
        }
        pass.commit_wait()?;
        let mut out = Vec::with_capacity(wanted.len());
        for (i, (p, _r)) in wanted.iter().enumerate() {
            let dst = scratch.logits_rows.view(i * VOCAB, &[VOCAB])?;
            out.push((*p, dst.to_f32()?));
        }
        Ok(out)
    }

    pub fn forward(
        &self,
        ctx: &MetalContext,
        state: &mut LfmState,
        scratch: &mut Scratch,
        tokens: &[u32],
        logits_positions: &[usize],
        stop: &AtomicBool,
    ) -> Result<Vec<(usize, Vec<f32>)>> {
        ensure!(state.mode == Mode::TeacherForced, "forward requires teacher_forced mode");
        let base = state.pos;
        let mut rows = Vec::new();
        let mut off = 0usize;
        let mut remaining = tokens.len();
        while remaining > 0 {
            if stop.load(Ordering::SeqCst) {
                anyhow::bail!("stdin closed");
            }
            let m = next_group(remaining);
            let chunk = &tokens[off..off + m];
            self.forward_chunk(ctx, state, scratch, chunk)?;
            let got = self.retain_logits(ctx, scratch, logits_positions, state.pos - m, m)?;
            rows.extend(got);
            off += m;
            remaining -= m;
        }
        rows.sort_by_key(|(p, _)| *p);
        state.pending = None;
        state.logits = rows.iter().cloned().collect();
        let _ = base;
        Ok(rows)
    }

    /// Greedy (`sampling` None) or sampled generation with the reference
    /// prefill partition. All decode steps are encoded back-to-back and the
    /// tokens stay device-resident, so one submission covers the whole
    /// generation.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        ctx: &MetalContext,
        state: &mut LfmState,
        scratch: &mut Scratch,
        prompt: &[u32],
        count: usize,
        sampling: Option<&Sampling>,
        stop: &AtomicBool,
    ) -> Result<Vec<u32>> {
        ensure!(state.mode == Mode::Greedy, "generate requires greedy mode");
        ensure!(count <= TOKEN_RING, "output count exceeds the token ring");
        if count == 0 {
            return Ok(Vec::new());
        }
        let buf = scratch.gen_tokens.clone();
        let started_with_prompt = !prompt.is_empty();
        // The seen set covers the prompt from here on; the GPU copy extends
        // it with each drawn token during the call.
        sample::mark_seen(&mut state.seen, prompt);
        if sampling.is_some_and(Sampling::uses_penalty) {
            scratch.sample.upload_seen(&state.seen)?;
        }
        if started_with_prompt {
            let last_chunk = self.prefill(ctx, state, scratch, prompt, stop, None, &mut |_| Ok(()))?;
            let first = buf.view(0, &[1])?;
            match sampling {
                None => self.next_from_final_to(ctx, state, scratch, &first, last_chunk - 1)?,
                Some(s) => {
                    self.sample_from_final_to(ctx, state, scratch, &first, last_chunk - 1, s)?
                }
            }
        } else {
            let pending = state
                .pending
                .ok_or_else(|| anyhow!("continuation without a pending token"))?;
            let first = buf.view(0, &[1])?;
            first.write_bytes(&pending.to_le_bytes())?;
        }

        // Submit decode steps in short batches (FIRST_DECODE_BATCH, then
        // DECODE_BATCH). Steps stay device-resident and are executed in order
        // by the single command queue.
        let first_step = if started_with_prompt { 1 } else { 0 };
        let mut pending = Vec::new();
        let mut closed = false;
        let mut i = first_step;
        let mut batch = FIRST_DECODE_BATCH;
        while i < count {
            let end = (i + batch).min(count);
            batch = DECODE_BATCH;
            let pass = ctx.begin()?;
            for j in i..end {
                if stop.load(Ordering::SeqCst) {
                    closed = true;
                    break;
                }
                let (in_idx, out_idx) = if j == 0 { (0, 0) } else { (j - 1, j) };
                let input = buf.view(in_idx, &[1])?;
                let output = buf.view(out_idx, &[1])?;
                self.decode_step_into(
                    ctx,
                    state,
                    scratch,
                    &pass,
                    &input,
                    &output,
                    j == first_step,
                    j + 1 < count,
                    sampling,
                )?;
                state.pos += 1;
                state.advance_conv();
            }
            pending.push(pass.commit()?);
            i = end;
            if closed {
                break;
            }
        }
        for p in pending {
            p.wait()?;
        }
        if closed {
            anyhow::bail!("stdin closed");
        }
        let tokens = buf.view(0, &[count])?.to_u32()?;
        state.pending = tokens.last().copied();
        state.logits.clear();
        sample::mark_seen(&mut state.seen, &tokens);
        Ok(tokens)
    }

    /// Consumes `prompt` with the reference chunk partition and returns the
    /// size of the last chunk (whose final row predicts the next token). With
    /// `split_at` = Some(q), 0 < q < prompt.len(), the submission containing
    /// position q is cut there and `at_split` sees the state after the first
    /// q tokens; the rest continues with the partition of the remaining
    /// tokens. A cut at `prompt.len() - 1` matches the reference partition,
    /// whose last chunk is always the final token alone.
    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        ctx: &MetalContext,
        state: &mut LfmState,
        scratch: &mut Scratch,
        prompt: &[u32],
        stop: &AtomicBool,
        split_at: Option<usize>,
        at_split: &mut dyn FnMut(&LfmState) -> Result<()>,
    ) -> Result<usize> {
        let mut off = 0usize;
        let mut remaining = prompt.len();
        let mut last_chunk = 0usize;
        while remaining > 0 {
            if stop.load(Ordering::SeqCst) {
                anyhow::bail!("stdin closed");
            }
            let mut m = next_group(remaining);
            if let Some(q) = split_at
                && off < q
                && q < off + m
            {
                m = q - off;
            }
            self.forward_chunk(ctx, state, scratch, &prompt[off..off + m])?;
            off += m;
            remaining -= m;
            last_chunk = m;
            if remaining > 0 && split_at == Some(off) {
                at_split(state)?;
            }
        }
        Ok(last_chunk)
    }

    /// Streaming generation: the decode loop of [`Self::generate`] with no
    /// per-call token limit and no host stall. The host keeps up to
    /// `STREAM_AHEAD` steps committed ahead; whenever the oldest command
    /// buffer completes it reads back that buffer's tokens and hands them to
    /// `on_tokens`, then tops the queue up again. Every step runs exactly the
    /// kernels of [`Self::generate`], so the tokens are bitwise those of the
    /// non-streaming path.
    ///
    /// `on_tokens` returns [`Flow::Stop`] with the number of the batch's
    /// tokens to keep (at least one token is always kept). Steps that were
    /// already committed past the last kept token are discarded: the
    /// position and the conv state are rewound to just after the last kept
    /// token (their KV rows lie beyond the position and are rewritten before
    /// any later read), so the state is exactly that of a non-streaming
    /// generation of the kept tokens and a continuation stays correct.
    ///
    /// `split_at` (a position within `prompt`, see [`Self::prefill`]) takes a
    /// [`PrefixSnapshot`] of the state after `prompt[..split_at]`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_stream(
        &self,
        ctx: &MetalContext,
        state: &mut LfmState,
        scratch: &mut Scratch,
        prompt: &[u32],
        max_tokens: usize,
        sampling: Option<&Sampling>,
        split_at: Option<usize>,
        stop: &AtomicBool,
        on_tokens: &mut dyn FnMut(&[u32]) -> Flow,
    ) -> Result<StreamResult> {
        ensure!(state.mode == Mode::Greedy, "generate requires greedy mode");
        ensure!(max_tokens > 0, "max_tokens must be positive");
        let buf = scratch.gen_tokens.clone();
        let started_with_prompt = !prompt.is_empty();
        let seen_before = split_at.map(|_| state.seen.clone());
        sample::mark_seen(&mut state.seen, prompt);
        if sampling.is_some_and(Sampling::uses_penalty) {
            scratch.sample.upload_seen(&state.seen)?;
        }
        let mut snapshot = None;
        if started_with_prompt {
            let mut take = |st: &LfmState| -> Result<()> {
                let q = split_at.expect("split position");
                let mut seen = seen_before.clone().expect("seen set");
                sample::mark_seen(&mut seen, &prompt[..q]);
                snapshot = Some(self.prefix_snapshot(st, seen)?);
                Ok(())
            };
            let last_chunk = self.prefill(ctx, state, scratch, prompt, stop, split_at, &mut take)?;
            let first = buf.view(0, &[1])?;
            match sampling {
                None => self.next_from_final_to(ctx, state, scratch, &first, last_chunk - 1)?,
                Some(s) => {
                    self.sample_from_final_to(ctx, state, scratch, &first, last_chunk - 1, s)?
                }
            }
        } else {
            let pending = state
                .pending
                .ok_or_else(|| anyhow!("continuation without a pending token"))?;
            buf.view(0, &[1])?.write_bytes(&pending.to_le_bytes())?;
        }
        // Token i (i >= 1, or i >= 0 for a continuation) is produced by the
        // step at position `pos + i - first_step`, which must fit the cache.
        let first_step = usize::from(started_with_prompt);
        let room = state.max_seq - state.pos + first_step;
        ensure!(room > 0, "context length exceeded");
        let limit = max_tokens.min(room);

        let mut inflight: VecDeque<(crate::metal::PendingPass<'_>, usize, usize)> = VecDeque::new();
        let mut next = first_step; // next token index to encode
        let mut delivered = 0usize; // tokens handed to `on_tokens`
        let mut kept: Option<usize> = None;
        let mut tokens: Vec<u32> = Vec::new();
        let mut batch = FIRST_DECODE_BATCH;
        let result = (|| -> Result<()> {
            loop {
                // Keep the GPU queue full: encode until STREAM_AHEAD steps are
                // committed past the delivered tokens. Before the prompt's
                // first token is delivered only the first (one-step) batch is
                // encoded, so the first token is not delayed.
                while next < limit && next < delivered + STREAM_AHEAD {
                    if stop.load(Ordering::SeqCst) {
                        anyhow::bail!("stopped");
                    }
                    let end = (next + batch).min(limit).min(delivered + STREAM_AHEAD);
                    batch = DECODE_BATCH;
                    let pass = ctx.begin()?;
                    for j in next..end {
                        let in_idx = if j == 0 { 0 } else { (j - 1) % TOKEN_RING };
                        let input = buf.view(in_idx, &[1])?;
                        let output = buf.view(j % TOKEN_RING, &[1])?;
                        self.decode_step_into(
                            ctx,
                            state,
                            scratch,
                            &pass,
                            &input,
                            &output,
                            j == first_step,
                            j + 1 < limit,
                            sampling,
                        )?;
                        state.pos += 1;
                        state.advance_conv();
                    }
                    inflight.push_back((pass.commit()?, next, end));
                    next = end;
                    if started_with_prompt && delivered == 0 {
                        break;
                    }
                }
                let (s, e) = if started_with_prompt && delivered == 0 {
                    (0, 1)
                } else if let Some((pending, s, e)) = inflight.pop_front() {
                    pending.wait()?;
                    (s, e)
                } else {
                    return Ok(());
                };
                let ring = buf.to_u32()?;
                let chunk: Vec<u32> = (s..e).map(|j| ring[j % TOKEN_RING]).collect();
                tokens.extend_from_slice(&chunk);
                delivered = e;
                if let Flow::Stop(n) = on_tokens(&chunk) {
                    kept = Some(s + n.clamp(usize::from(s == 0), chunk.len()));
                    return Ok(());
                }
                if delivered >= limit {
                    return Ok(());
                }
            }
        })();
        for (pending, _, _) in inflight {
            pending.wait()?;
        }
        result?;
        let total = kept.unwrap_or(delivered);
        tokens.truncate(total);
        // Steps encoded past the last kept token (index total - 1) are
        // discarded.
        let extra = next - total;
        ensure!(extra < CONV_SLOTS, "streaming rewind beyond the conv-state ring");
        state.pos -= extra;
        state.rewind_conv(extra);
        state.pending = tokens.last().copied();
        state.logits.clear();
        sample::mark_seen(&mut state.seen, &tokens);
        Ok(StreamResult { tokens, snapshot })
    }

    fn next_from_final_to(
        &self,
        ctx: &MetalContext,
        _state: &mut LfmState,
        scratch: &mut Scratch,
        out: &Tensor,
        last_row: usize,
    ) -> Result<()> {
        let pass = ctx.begin()?;
        let row = scratch.final_h.view(last_row * HIDDEN, &[1, HIDDEN])?;
        klfm::gemv_quant(ctx, &pass, &self.weights.embed, &row, &scratch.logits1)?;
        klfm::argmax_bf16(
            ctx,
            &pass,
            &scratch.logits1,
            &scratch.argmax_partials,
            out,
        )?;
        pass.commit_wait()
    }

    /// The sampled counterpart of [`Self::next_from_final_to`]: draws the
    /// token at position `state.pos` from the prompt's last logits row.
    fn sample_from_final_to(
        &self,
        ctx: &MetalContext,
        state: &mut LfmState,
        scratch: &mut Scratch,
        out: &Tensor,
        last_row: usize,
        s: &Sampling,
    ) -> Result<()> {
        let pass = ctx.begin()?;
        let row = scratch.final_h.view(last_row * HIDDEN, &[1, HIDDEN])?;
        klfm::gemv_quant(ctx, &pass, &self.weights.embed, &row, &scratch.logits1)?;
        let key = &scratch.argmax_key;
        scratch.sample.encode_draw(ctx, &pass, &scratch.logits1, key, s, state.pos)?;
        klfm::token_tail(ctx, &pass, &scratch.argmax_key, out, None, self.cfg.norm_eps)?;
        if s.uses_penalty() {
            scratch.sample.encode_mark(ctx, &pass, out)?;
        }
        pass.commit_wait()
    }
}

/// The decision of a [`LfmModel::generate_stream`] callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    Continue,
    /// Stop after the first `n` tokens of the batch just delivered.
    Stop(usize),
}

pub struct StreamResult {
    /// The kept tokens.
    pub tokens: Vec<u32>,
    /// The state after `prompt[..split_at]`, when a split was requested.
    pub snapshot: Option<PrefixSnapshot>,
}

// ------------------------------------------------------------------ snapshots

/// The recurrent part of the state after a prompt prefix: the position, the
/// current conv state of every conv layer and the seen-token set. It does not
/// copy the KV cache: it is valid only while the KV rows `[0, pos)` are those
/// written when it was taken, i.e. until a prefill or decode step writes a row
/// below `pos` (after a `reset`, or after restoring an earlier position).
pub struct PrefixSnapshot {
    pos: usize,
    conv: Vec<Option<Vec<u8>>>,
    seen: Vec<u32>,
}

impl PrefixSnapshot {
    pub fn pos(&self) -> usize {
        self.pos
    }
}

pub struct Snapshot {
    mode: Mode,
    pos: usize,
    pending: Option<u32>,
    conv_slot: usize,
    /// The current conv state of each conv layer (the next token overwrites
    /// every other slot before reading it).
    conv: Vec<Option<Vec<u8>>>,
    kv: Vec<Option<(Vec<u8>, Vec<u8>)>>,
    logits: BTreeMap<usize, Vec<f32>>,
    seen: Vec<u32>,
}

impl LfmModel {
    pub fn scratch(&self, ctx: &MetalContext, cap: usize) -> Result<Scratch> {
        Scratch::new(ctx, cap)
    }

    pub fn checkpoint(&self, state: &LfmState) -> Result<Snapshot> {
        let conv = current_conv(state);
        let mut kv = Vec::with_capacity(N_LAYERS);
        for cache in &state.kv {
            kv.push(match cache {
                Some((k, v)) => Some((read_kv_used(k, state.pos)?, read_kv_used(v, state.pos)?)),
                None => None,
            });
        }
        Ok(Snapshot {
            mode: state.mode,
            pos: state.pos,
            pending: state.pending,
            conv_slot: state.conv_slot,
            conv,
            kv,
            logits: state.logits.clone(),
            seen: state.seen.clone(),
        })
    }

    pub fn restore(&self, state: &mut LfmState, snap: &Snapshot) -> Result<()> {
        ensure!(snap.pos <= state.max_seq, "checkpoint position exceeds context");
        for (s, snap_s) in state.conv.iter_mut().zip(&snap.conv) {
            if let (Some(bufs), Some(data)) = (s.as_mut(), snap_s.as_ref()) {
                bufs[snap.conv_slot].write_bytes(data)?;
            }
        }
        for (cache, snap_c) in state.kv.iter_mut().zip(&snap.kv) {
            if let (Some((k, v)), Some((kd, vd))) = (cache.as_mut(), snap_c.as_ref()) {
                write_kv_used(k, snap.pos, kd)?;
                write_kv_used(v, snap.pos, vd)?;
            }
        }
        state.mode = snap.mode;
        state.pos = snap.pos;
        state.pending = snap.pending;
        state.conv_slot = snap.conv_slot;
        state.logits = snap.logits.clone();
        state.seen.clone_from(&snap.seen);
        Ok(())
    }

    fn prefix_snapshot(&self, state: &LfmState, seen: Vec<u32>) -> Result<PrefixSnapshot> {
        Ok(PrefixSnapshot { pos: state.pos, conv: current_conv(state), seen })
    }

    /// Returns the sequence to a [`PrefixSnapshot`] in greedy mode, with no
    /// pending token: the next `generate` must carry a prompt (the tokens
    /// after the prefix). The caller guarantees the KV rows below the
    /// snapshot's position are unchanged since it was taken.
    pub fn restore_prefix(&self, state: &mut LfmState, snap: &PrefixSnapshot) -> Result<()> {
        ensure!(snap.pos <= state.max_seq, "snapshot position exceeds context");
        let slot = state.conv_slot;
        for (s, data) in state.conv.iter_mut().zip(&snap.conv) {
            if let (Some(bufs), Some(data)) = (s.as_mut(), data.as_ref()) {
                bufs[slot].write_bytes(data)?;
            }
        }
        state.mode = Mode::Greedy;
        state.pos = snap.pos;
        state.pending = None;
        state.logits.clear();
        state.seen.clone_from(&snap.seen);
        Ok(())
    }
}

fn current_conv(state: &LfmState) -> Vec<Option<Vec<u8>>> {
    state
        .conv
        .iter()
        .map(|s| s.as_ref().map(|bufs| bufs[state.conv_slot].raw_bytes().to_vec()))
        .collect()
}

fn read_kv_used(t: &Tensor, pos: usize) -> Result<Vec<u8>> {
    let heads = t.shape()[0];
    let max_seq = t.shape()[1];
    let d = t.shape()[2];
    let mut out = vec![0u8; heads * pos * d * 2];
    let raw = t.raw_bytes();
    for h in 0..heads {
        let src = h * max_seq * d * 2;
        let dst = h * pos * d * 2;
        out[dst..dst + pos * d * 2].copy_from_slice(&raw[src..src + pos * d * 2]);
    }
    Ok(out)
}

fn write_kv_used(t: &Tensor, pos: usize, data: &[u8]) -> Result<()> {
    let heads = t.shape()[0];
    let max_seq = t.shape()[1];
    let d = t.shape()[2];
    ensure!(data.len() == heads * pos * d * 2, "kv snapshot size");
    let mut raw = vec![0u8; t.byte_len()];
    // Preserve untouched tail rows? They are never read past `pos`, so zeroing
    // them is harmless and keeps the write simple.
    for h in 0..heads {
        let dst = h * max_seq * d * 2;
        let src = h * pos * d * 2;
        raw[dst..dst + pos * d * 2].copy_from_slice(&data[src..src + pos * d * 2]);
    }
    t.write_bytes(&raw)
}
