# VENDORED

Third-party code in this tree: where it comes from, its license, and how it was changed. Every file not named
below was written for Molten.

## Lily

- Origin: Lily, Perplexity AI's Rust + Metal inference engine, https://github.com/perplexityai/pplx-garden,
  subdirectory `lily/`, revision `1ed972ed3f0bd5616c997c9507c25616c63394fc`.
- License: Apache-2.0 (SPDX `Apache-2.0`), Copyright 2026 Perplexity AI, Inc.
- `LICENSE` and `NOTICE` at the tree root are Lily's files, byte-identical. `NOTICE` is Lily's own attribution
  file, including its MLX and MLX-LM (MIT) notice; it also names Lily files that are not in this tree
  (`src/kernels/metal/moe.metal`, `src/kernels/metal/gdn.metal`, `src/kernels/gdn.rs`,
  `tests/support/cpu_ref.rs`, and the scaled-dot-product attention kernels of Lily's `attention.metal`). The
  `src/kernels/metal/quant_nax.metal` and `tools/gen_quant_nax.py` in this tree are Molten's own (see MLX below),
  not the Lily files of the same names that `NOTICE` lists.
- Molten started from Lily's Metal runtime, tensor and checkpoint-reading code and a subset of its kernels, and
  was then rewritten for LFM2.5-24B-A2B. Lily's Qwen3.6-35B-A3B model graph, decode loop, weight loader, MoE,
  Gated DeltaNet and scaled-dot-product attention kernels, CLI, HTTP server, tokenizer and benchmark binary are
  not in this tree.

### Files derived from Lily

`upstream sha256` identifies the exact upstream file. All of these files except `LICENSE` and `NOTICE` are
modified, and each modified file starts with a one-line notice saying so; the last column summarizes the changes.

| file | upstream file (`lily/...`) | upstream sha256 | changes |
|---|---|---|---|
| `LICENSE` | `LICENSE` | `b9e2b50a48a09fb39dafa71ab1a11a6e87389addda17faa99d55f5f74a0001ac` | none (byte-identical) |
| `NOTICE` | `NOTICE` | `7a01e666044fb8aca1fd88c636bca43bcc95b8b1fecbe101dce4c7c80fbb5759` | none (byte-identical) |
| `src/lib.rs` | `src/lib.rs` | `74ce0f1e90674982d5bcd0bd177b30975e5f836195cd681939d894b940d0ae19` | module list replaced with this engine's modules; crate documentation rewritten |
| `src/metal.rs` | `src/metal.rs` | `dfa57ae244dd55062e170df77afaef29c66cf3af511c3d3331c6dd781b7f3150` | compiled-library cache (one compile per source and MSL version), `pipeline_fc` (pipelines specialized with boolean function constants), `Kernel::max_total_threads`, command-buffer error check in `commit_wait`; concurrent-encoder passes, GPU pass timing, `new_buffer_with_bytes`, `device` and `static_threadgroup_memory_length` removed; comments edited |
| `src/kernels.rs` | `src/kernels.rs` | `97294f1dee16e0e92d98f3681c8b5c5e7b1282e55bf015bace2721b9fb652970` | module list (`lfm`, `steel` added; `gdn`, `moe`, `skinny` removed); `f32_bytes` removed |
| `src/tensor.rs` | `src/tensor.rs` | `87d765f15fa37ee1a4d20b7c162230f3f3fd273f299b4c9d6772e15237bfccab` | `reshape` added; `from_bytes`, `from_f32`, `from_f32_as_bf16`, `buffer` and `identity` removed |
| `src/safetensors.rs` | `src/safetensors.rs` | `01d98906984c650e553e65f643329eff0da5047065d71a16223aa631a0f7e7c1` | `Checkpoint::read` removed; one comment reworded |
| `src/weights.rs` | `src/weights.rs` | `2d797f9e24a395cec8ce6e96141be18bd869c1e1556fd928875fa135e2f5c7cc` | reduced to the `QuantWeights` type; Lily's Qwen3.6-35B-A3B checkpoint loader removed (the LFM2.5 loader is in `src/lfm.rs`); comments rewritten |
| `src/kernels/attention.rs` | `src/kernels/attention.rs` | `17b9e774325e9857ada11a0c77f169031b4d94a1049855c07e48c6540bfe93c6` | reduced to `rope_neox` and `scatter_kv` |
| `src/kernels/elementwise.rs` | `src/kernels/elementwise.rs` | `af849fc6ee0b7047bde955f85a1b83003c0675e8377a15bc77d3cfc51dae3f57` | reduced to `silu_mul_bf16` and `gather_rows_bf16`; the gather dispatches 8- and 16-element vector kernels when the row width allows |
| `src/kernels/gemm.rs` | `src/kernels/gemm.rs` | `9e338149c28065b1da8424bdb40aa5597ceebdffc90a51d93961baba73c68c85` | 128x128 output tile for row counts of at least 4096 |
| `src/kernels/norm.rs` | `src/kernels/norm.rs` | `094b63c71ed2fbf76232ada70953dad2a6ad203785aecd5cb4770f0d1e13d73a` | dispatches the vectorized `_v8` kernels when every stream is 16-byte aligned |
| `src/kernels/quant.rs` | `src/kernels/quant.rs` | `3fb1ae30caa34ba2916e318abd444d79c617d8829e2b8e54c28ca95c7491a3ad` | keeps the weight checks and the dequantize + BF16 GEMM path; adds bulk Q4 dequantization and the grouped expert GEMM wrappers over MLX's `qmm_t` body (`gemm_q4_grouped_mlx`, `gemm_q4_grouped_mlx_gu`); Lily's GEMV, grouped tensor-op GEMM and MoE routing policy removed |
| `src/kernels/metal/attention.metal` | `src/kernels/metal/attention.metal` | `0252d634654c71237bf59c62fdb493a399330324816bf9b5545e43259380e540` | reduced to `rope_neox_bf16` and `scatter_kv_bf16` (bodies unchanged) |
| `src/kernels/metal/elementwise.metal` | `src/kernels/metal/elementwise.metal` | `1566e7de3d16e39637a524c8e54d706f38501bab0245b8b55f9d16c5fa55250b` | reduced to `silu_mul_bf16` (which now rounds the SiLU result to BF16 before the product, as MLX does) and `gather_rows_bf16`; `gather_rows_bf16_v8` and `gather_rows_bf16_v16` added |
| `src/kernels/metal/gemm.metal` | `src/kernels/metal/gemm.metal` | `f82fe60ada9fec0fdfb9278dbf4802b79350057d11ba3f06cefe784df4a43201` | `gemm_bf16_nt_nax_wide` (128x128 tile, same K order and FP32 accumulation) added |
| `src/kernels/metal/norm.metal` | `src/kernels/metal/norm.metal` | `2a996c5598e6685d32e32d57099fe00b2e398f4ca3df7176b69f9393cdb95d33` | vectorized `add_rmsnorm_bf16_v8` and `rmsnorm_bf16_v8` (H=2048, same reduction order) added |
| `src/kernels/metal/quant.metal` | `src/kernels/metal/quant.metal` | `a8ee3860329d2aefa060dc35ab0e00ebd9c22f8295a489b4ccb6f6cb005e5a70` | reduced to the Q4/Q8 dequantization kernels; `dequant_q4_bf16_bulk` added |

### Files written for Molten that contain adapted Lily code

- `src/kernels/metal/lfm.metal`: `dot_word_q4` and the `GEMV_Q4_BODY` row reduction are Lily's (`quant.metal`,
  a comment added); `lfm_argmax_bf16_partial` and `lfm_argmax_final` are Lily's `argmax_f32_partial` and
  `argmax_f32_final` (`elementwise.metal`) with BF16 input and renamed types.
- `src/kernels/lfm.rs`: `argmax_bf16` follows Lily's `argmax_f32` dispatch (`src/kernels/elementwise.rs`).
- `Cargo.toml` is this tree's own manifest. Its release profile keeps Lily's `codegen-units = 1` and sets
  `lto = false` instead of Lily's `lto = "thin"`.

## MLX

- Origin: Apple MLX, https://github.com/ml-explore/mlx, version 0.32.3. The headers are the
  `mlx/include/mlx/backend/metal/kernels/` files installed by the `mlx-metal` 0.32.3 Python distribution; their
  sha256 values equal the ones in that distribution's `RECORD`.
- License: MIT (SPDX `MIT`), Copyright 2023 Apple Inc. MLX's `LICENSE` is kept byte-identical at
  `third_party/mlx-0.32.3/LICENSE`.

### `third_party/mlx-0.32.3/` (unmodified)

`third_party/mlx-0.32.3/` corresponds to MLX's `include/` directory, so the files keep the relative paths their
`#include "mlx/backend/metal/kernels/..."` lines use. These are the headers that `tools/gen_steel_attn.py` and
`tools/gen_quant_nax.py` read (the two entry headers and their includes). Nothing under `third_party/` is compiled
or embedded by `cargo build`. 16 headers, 190344 bytes, plus `LICENSE` (1066 bytes).

| file in this tree | sha256 | bytes |
|---|---|---|
| `third_party/mlx-0.32.3/LICENSE` | `ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01` | 1066 |
| `third_party/mlx-0.32.3/mlx/backend/metal/kernels/bf16.h` | `abd87446a310b77ac530ef52a324feae5cb285d03ec9613e3a88ebb71410fdcb` | 306 |
| `third_party/mlx-0.32.3/mlx/backend/metal/kernels/bf16_math.h` | `1f374f8380f756eb89acf6a847741cb8fecbe642945e159fb6208d804cc06496` | 26024 |
| `third_party/mlx-0.32.3/mlx/backend/metal/kernels/complex.h` | `16e8a815b2cbdb6070e0824e64fe33fccb6e918f1b84ea5c792bd89d33e57bf1` | 9062 |
| `third_party/mlx-0.32.3/mlx/backend/metal/kernels/defines.h` | `a6c2c824e855078cd092fd9592457616f995755f0e865475c3f006a3f5d6bf1d` | 888 |
| `third_party/mlx-0.32.3/mlx/backend/metal/kernels/logging.h` | `197d9d81944146bea274ebcbc17b2b97b39c7478c0316a3166a08b3fb2467e4b` | 511 |
| `third_party/mlx-0.32.3/mlx/backend/metal/kernels/quantized_nax.h` | `1a9197c4afaec8dddc869165a4328f712c7b69d3a3a946e9cafe6ddebc456027` | 48933 |
| `third_party/mlx-0.32.3/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h` | `e576cbbe13a8fe5fda7b630422f6083faac6114fbf2aad3970323932b1e6cc44` | 27486 |
| `third_party/mlx-0.32.3/mlx/backend/metal/kernels/steel/attn/nax.h` | `fb10bcae44095e9fed5c81213d03527dd5fe5ac11be11ec7784bd357c6f317ec` | 25621 |
| `third_party/mlx-0.32.3/mlx/backend/metal/kernels/steel/attn/params.h` | `f03c6ed73d7298fa69b75a49e7eec54954b6fafd59530a1f175934215412ed80` | 1225 |
| `third_party/mlx-0.32.3/mlx/backend/metal/kernels/steel/attn/transforms.h` | `6479114663c5923a7ee2f9fc1cf9d5e99e680d47d793251749989056a8a63141` | 1714 |
| `third_party/mlx-0.32.3/mlx/backend/metal/kernels/steel/defines.h` | `b03cea6a7d5cfe814e6838d8536e2aacdcafd2744cb408e954fda336ce21759a` | 231 |
| `third_party/mlx-0.32.3/mlx/backend/metal/kernels/steel/gemm/nax.h` | `fb10bcae44095e9fed5c81213d03527dd5fe5ac11be11ec7784bd357c6f317ec` | 25621 |
| `third_party/mlx-0.32.3/mlx/backend/metal/kernels/steel/utils.h` | `d4c36298145d1c5617a94f07dbfabf6ac932afe2fd7ea0c9291f1fb385c5404a` | 1073 |
| `third_party/mlx-0.32.3/mlx/backend/metal/kernels/steel/utils/integral_constant.h` | `886eccaac4f8f00eab5a10551fdb5315df7ab7462aba13d69bb95765c1a4735e` | 3700 |
| `third_party/mlx-0.32.3/mlx/backend/metal/kernels/steel/utils/type_traits.h` | `4766c5a3809d3cf7740e63acc5703ef76b6b4e85d8baf16602c9d517a2fd426c` | 1145 |
| `third_party/mlx-0.32.3/mlx/backend/metal/kernels/utils.h` | `e40473c36e2f2e7ba250b29e8dbbe3c150ce207379e57a2818e916b343e9f2cf` | 16804 |

### Generated from the MLX headers (committed; regenerate with the scripts in `tools/`)

- `src/kernels/metal/steel_attn_nax.metal`, by `tools/gen_steel_attn.py` from
  `steel/attn/kernels/steel_attention_nax.h` and `utils.h` with their includes. Modified: the header tree is
  flattened into one source; the attention templates read their Q/O head and query block from a
  (head within the GQA group, query block, KV head) grid instead of (query block, query head), which renames
  threadgroup axes only; one instantiation, `lfm_steel_nax_bq64k32` (BF16, head dim 64, BQ=64, BK=32, WM=WN=2).
- `src/kernels/metal/quant_nax.metal`, by `tools/gen_quant_nax.py` from `quantized_nax.h`, `utils.h`,
  `steel/utils.h` and `steel/gemm/nax.h` with their includes. Modified: flattened into one source;
  `qmm_t_nax_tgp_impl` is copied as `lfm_qmm_t_body` with `K`, `N` and `M` passed by value, and as
  `lfm_qmm_t_shared_body`, whose per-simdgroup MMA loop and store are skipped for rows beyond the expert's
  segment; entry points for this engine's grouped block map are added (`lfm_q4_grouped_mlx_t32`,
  `lfm_q4_grouped_mlx_s128w1`, and the fused gate/up `lfm_q4_grouped_mlx_gu_t32`).
- `tools/gen_steel_attn.py` and `tools/gen_quant_nax.py` were written for Molten.

### Other code following MLX

- `src/kernels/steel.rs`: `AttnParams` mirrors `mlx::steel::AttnParams`; the launcher dispatches the generated
  steel attention kernel.
- `src/kernels/metal/lfm.metal`: the single-token GEMV follows MLX's affine `qmv`/`qdot` accumulation, 6-bit codes
  are unpacked in MLX's packing, and single-token attention follows MLX 0.32.3 `sdpa_vector` and
  `sdpa_vector_2pass_1`/`sdpa_vector_2pass_2` for head dimension 64.
- `src/kernels/metal/quant.metal` (from Lily) uses MLX's affine quantization format; Lily's `NOTICE` covers it.

## mlx-lm

No mlx-lm file is copied into this tree. `src/lfm.rs` implements the `lfm2_moe` model and the prefill chunk
partition as mlx-lm (MIT, Copyright 2023 Apple Inc.; https://github.com/ml-explore/mlx-lm) defines them, so that
Molten computes the same results; `bench/bench_mlx_lm.py` runs mlx-lm itself for comparison.

## Crates

No crate is vendored into this tree; all come from crates.io at the exact versions pinned in `Cargo.toml` and
resolved in `Cargo.lock`. The engine core uses `anyhow` 1.0.104, `bytemuck` 1.25.2, `half` 2.7.1, `serde` 1.0.229,
`serde_json` 1.0.151, `objc2` 0.6.4, `objc2-foundation` 0.3.2 and `objc2-metal` 0.3.2. The text layer
(`molten run`, `chat`, `serve`, `text`) uses `tokenizers` 0.23.2 with default features off and the pure-Rust
`fancy-regex` backend (no Oniguruma, no esaxx C++), `minijinja` 2.24.0 and `minijinja-contrib` 2.24.0
(`pycompat`). Nothing in the build compiles C or C++.
