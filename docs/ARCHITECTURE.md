# How Molten works

Technical notes on the engine: the model graph, prefill, the decode loop and its fused kernels, sampling, streaming and
serving, and where the time goes. See [BENCHMARKS.md](BENCHMARKS.md) for the measurements.

- **Model graph** (`src/lfm.rs`):
  - 30 double-gated short-convolution layers and 10 GQA attention layers (Q/K RMSNorm, NeoX RoPE).
  - Two dense lead layers, then 38 MoE layers: 64 experts, top-4 sigmoid routing with a selection bias.
  - A 6-bit tied embedding/LM head.
  - Weights are read from the safetensors shards into GPU buffers once at load, unconverted, in MLX's affine
    4-bit/64 layout.
- **Prefill.** It uses the reference's chunk partition. The grouped expert GEMMs and causal flash attention are
  adapted from MLX's steel/NAX kernels. The four query heads of a KV head share K/V tiles.
- **Decode** is one GPU-resident loop. Tokens never leave the GPU between steps: the argmax of step t feeds the
  embedding gather of step t+1. The host encodes steps ahead in batches and waits once per request.
  One token takes about 252 dependent dispatches below 8,192 tokens of context and 272 from there on (down from
  about 355). The main fusions:
  - The residual add + RMSNorm is fused into the F32 router GEMV. The top-4 routing runs in the router's
    last-finishing threadgroup, using an atomic counter and device-scope fences with no inter-threadgroup waiting.
  - The routed experts' gate/up/SiLU and down/combine each run in one gather kernel.
  - The convolution's B/C/X projections and the conv itself are one kernel. The attention Q/K/V projections write V
    straight into the cache.
  - The Q/K norm and RoPE run in the decode attention kernel's prologue.
  - The LM head GEMV carries an argmax epilogue, so logits are not stored during generation. One small kernel then
    writes the token, the next embedding row and its RMSNorm.
  - All decode GEMVs have compile-time shapes.
- **Sampling** (`src/lfm/sample.rs`, `src/kernels/metal/lfm_sample.metal`) stays inside the GPU-resident loop. The
  head GEMV stores the logits; the token is drawn with the Gumbel-max trick from counter-based Philox noise (seed,
  position), folded into the same key the greedy argmax uses, and the greedy token tail continues from it.
  - Temperature only: one extra kernel.
  - Top-k / top-p: exact threshold search: count/mass levels with integer accumulators, refinement rounds, a band
    gather and a one-threadgroup sort.
  - Measured decode cost against greedy at 256 / 1K / 4K context (median of paired rounds, split request on nonce
    prompts): temperature 0.6 alone is within noise (-0.6% / +0.1% / 0.0%). Temperature 0.6 with top_k 50 and
    repetition_penalty 1.05 costs +0.6% / +1.0% / +1.2%. The greedy path is unchanged (same tokens, same speed).
- **Streaming and serving** (`src/lfm.rs`, `src/session.rs`, `src/server.rs`). `run`, `chat` and `serve` use a
  streaming form of the same decode loop: the host keeps 16 steps committed ahead and reads back each completed
  command buffer's tokens while the GPU runs the next ones, so streaming costs no GPU time; the tokens are those of
  the protocol path bit for bit. Stop conditions (EOS, stop strings, `max_tokens`) are checked on every completed
  batch; steps already committed past the stop are discarded, and the state is rewound exactly to the last returned
  token (each conv layer keeps a ring of 32 states). Prompt-prefix reuse restores a snapshot of the recurrent state
  and keeps the KV cache rows, so a new chat turn prefills only its new text.
- **Text** (`src/text.rs`, `src/chat_template.rs`): the checkpoint's `tokenizer.json` through Hugging Face
  `tokenizers` with the pure-Rust regex backend, and its `chat_template.jinja` through `minijinja`, rendering exactly
  what transformers' `apply_chat_template` renders. No C or C++ is compiled anywhere in the build.
- **Where the time goes** (decode at 256 tokens, measured in Molten):
  - The step streams about 1.46 GB of weights, about 445 GB/s on average. The weight-streaming kernels run at
    430-590 GB/s; a plain streaming read on this machine reaches about 590 GB/s.
  - The expert GEMVs take about 50% of the step, the convolutions about 13%, the head about 11%.
  - The GPU is busy 99% of the time.

