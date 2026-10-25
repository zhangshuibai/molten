# Benchmarks

Measured on 2026-10-25 on one Apple M5 Max (128 GB, macOS 26). Each number is the mean of two passes run in opposite
order (Molten, mlx-lm, llama.cpp, then llama.cpp, mlx-lm, Molten), with 2-minute cool-downs between backends. Within
each pass, every value is the median of 5 runs after 1 warm-up. Passes agreed within 1-2%; the one exception is
llama.cpp decode at 4K, at 4%. Molten was built with `cargo build --release` from fresh clones of this repository
shortly before the v0.1.0 release; every later change to the engine was checked to give the same tokens at the same
speed.

The 2K and 8K rows come from a later run with the same two-pass method. The 32K-128K rows and the end-to-end
measurements come from another run on the same day and machine: one pass (Molten, mlx-lm, llama.cpp) with 2-minute
cool-downs, 3 runs per length after 1 warm-up. "128K" is 130,816 tokens, which leaves room for the generated tokens in
the 131072-token context.

## Setup

| backend | version | weights |
|---|---|---|
| Molten | 0.1.0 (pre-release builds) | `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` @ 667408d4 (MLX affine 4-bit/64, 6-bit head) |
| mlx-lm | 0.32.0 on MLX 0.32.3, unmodified, its own `generate_step` loop | the same MLX checkpoint |
| llama.cpp | 926862e (2026-10-02), Metal, all layers on the GPU, flash attention auto | `LiquidAI/LFM2.5-24B-A2B-GGUF` `LFM2.5-24B-A2B-QAD-Q4_0.gguf` (official QAD Q4_0) |

The weights differ in format between backends. llama.cpp runs LiquidAI's official Q4_0 GGUF (blocks of 32 with a
scale). The other two run the MLX checkpoint (groups of 64 with a scale and a bias). Both are about 13.5 GB.

## Method

**Molten and mlx-lm** use the same deterministic nonce token prompts:
- TTFT is the time from the request to the first generated token, including prefill: `generate(P, 1)` for Molten,
  the first yield of `generate_step` for mlx-lm.
- Decode is the time for the next 128 tokens, with greedy sampling.

**llama.cpp** uses `llama-bench`:
- Decode is `-d P -n 128 -p 0` (128 tokens generated after a context of P tokens).
- Prefill is `-p P -n 0`.
- llama-bench does not report TTFT. It is estimated as prefill time + one decode step and marked *est.*

Scripts: `bench/bench_engine.py`, `bench/bench_mlx_lm.py`, `bench/bench_llama_cpp.sh`, `bench/compare.py`.

## Decode (tokens/s, higher is better)

| context | Molten | mlx-lm | llama.cpp | vs mlx-lm | vs llama.cpp |
|---:|---:|---:|---:|---:|---:|
| 256 | **304.5** | 190.0 | 198.5 | 1.60x | 1.53x |
| 1K | **300.2** | 188.5 | 195.6 | 1.59x | 1.53x |
| 2K | **291.4** | 188.8 | 193.5 | 1.54x | 1.51x |
| 4K | **285.1** | 184.0 | 189.0 | 1.55x | 1.51x |
| 8K | **269.1** | 180.4 | 186.6 | 1.49x | 1.44x |
| 16K | **248.2** | 164.6 | 171.7 | 1.51x | 1.45x |
| 32K | **212.7** | 145.6 | 154.4 | 1.46x | 1.38x |
| 64K | **168.1** | 115.8 | 126.1 | 1.45x | 1.33x |
| 128K | **119.1** | 82.2 | 89.8 | 1.45x | 1.33x |

LiquidAI's model card reports 184 tokens/s decode on an M5 Max with llama.cpp (Q4_K_M, 4K-token context). That agrees
with the llama.cpp row here (189 at 4K).

## Time to first token (ms, lower is better)

| prompt | Molten | mlx-lm | llama.cpp (est.) | vs mlx-lm | vs llama.cpp |
|---:|---:|---:|---:|---:|---:|
| 256 | **60.3** | 67.4 | 70.2 | 1.12x | 1.16x |
| 1K | **127.5** | 146.7 | 201.4 | 1.15x | 1.58x |
| 2K | **215.0** | 236.4 | 399.0 | 1.10x | 1.86x |
| 4K | **395.3** | 463.9 | 859.4 | 1.17x | 2.17x |
| 8K | **790.7** | 956.4 | 1818 | 1.21x | 2.30x |
| 16K | **1756** | 2168 | 4198 | 1.23x | 2.39x |
| 32K | **3880** | 4600 | 9416 | 1.19x | 2.43x |
| 64K | **10493** | 12229 | 25163 | 1.17x | 2.40x |
| 128K | **28060** | 32448 | 72739 | 1.16x | 2.59x |

## Prefill (tokens/s, higher is better)

| prompt | Molten | mlx-lm | llama.cpp |
|---:|---:|---:|---:|
| 256 | **4243** | 3800 | 3927 |
| 1K | **8032** | 6981 | 5218 |
| 2K | **9524** | 8662 | 5200 |
| 4K | **10361** | 8829 | 4796 |
| 8K | **10360** | 8565 | 4521 |
| 16K | **9330** | 7559 | 3908 |
| 32K | **8445** | 7124 | 3482 |
| 64K | **6246** | 5359 | 2605 |
| 128K | **4662** | 4032 | 1799 |

## End-to-end latency

| | Molten | mlx-lm | llama.cpp |
|---|---:|---:|---:|
| end-to-end, 1024-token prompt + 512 generated tokens | **1.84 s** | 2.77 s | 2.81 s |
| of which time to first token | **133 ms** | 146 ms | 200 ms (est.) |
| decode over the 512 tokens (tok/s) | **300** | 195 | 196 |

Median of 5 runs after 1 warm-up, greedy. Molten runs through its JSON-lines protocol (`generate(P, 1)`, then
continuations of at most 256 tokens) and mlx-lm through `generate_step`. llama.cpp uses `llama-bench -pg 1024,512`
(prompt and generation as one test), with its time to first token estimated from `-p 1024`.

## Paired evaluation against mlx-lm

A separate paired evaluation compared Molten with unmodified mlx-lm 0.32.0 (MLX 0.32.3, the same weights):
- ten context lengths from 256 to 131008 tokens, on fresh nonce prompts;
- five timed rounds per length, alternating Molten and mlx-lm with 120 s cool-downs;
- each request is `generate(P, 1)` (timed as prefill) followed by `generate((), 64)` (timed as decode).

| context | mlx-lm decode tok/s | Molten decode tok/s | decode speedup | prefill speedup |
|---:|---:|---:|---:|---:|
| 256 | 198 | **303** | 1.53x | 1.01x |
| 512 | 196 | **302** | 1.53x | 1.03x |
| 1K | 195 | **299** | 1.53x | 1.12x |
| 2K | 192 | **293** | 1.52x | 1.07x |
| 4K | 186 | **275** | 1.52x | 1.16x |
| 8K | 179 | **262** | 1.49x | 1.23x |
| 16K | 164 | **236** | 1.43x | 1.23x |
| 32K | 136 | **201** | 1.46x | 1.20x |
| 64K | 108 | **160** | 1.49x | 1.19x |
| 128K | 79 | **114** | 1.44x | 1.12x |

The whole-request score, the geometric mean over the ten lengths of the generate-request speedup, is 1.28x.

## Correctness

Molten computes the same model as mlx-lm, in bfloat16 with the same 4-bit weights: no changed quantization, no
speculation, no approximation. Its kernels reproduce MLX's arithmetic (rounding points and reduction orders) closely
enough that greedy tokens agree with mlx-lm except at near-ties.

- **Token-level checks against mlx-lm**, on prompts from 1 to 131008 tokens: teacher-forced token agreement (at
  most 25% mismatches per 65-step stream and 10% overall; the largest stream had 10 of 65), checkpoint/restore and
  continuation identity, and bitwise run-to-run determinism.
- **Held-out prompts** (not used during development, run once): 4.5% token mismatches overall (186 of 4160 compared
  steps), a mean NLL delta of 0.050 on English, Chinese and code text, and a 1.28x whole-request speedup.
- **Every decode optimization** in the last stage of development was verified token for token against the version
  before it: 84 prompts of up to 131008 tokens, in three request shapes each.

## Task accuracy

Two public benchmarks from LiquidAI's model card, run through Molten with the card's settings (temperature 0.6, at
most 32768 output tokens; top_k 50 and repetition_penalty 1.05 from the checkpoint's generation_config.json, which
vLLM applies by default), on fixed random subsets of 200 problems each:

| benchmark | Molten (MLX 4-bit) | LiquidAI model card (vLLM, BF16) |
|---|---:|---:|
| MATH-500 (math-verify) | 95.5% | 96.04% |
| IFEval, prompt-level strict | 96.0% | 96.75% |
| IFEval, mean of the four metrics | 97.1% | |

On MATH-500, 6 of the 9 misses reached the 32768-token limit, and 2 more are equivalent answers that math-verify
does not match ("3 < λ ≤ 4" for "(3,4]", "12th" for "12"). With 200 problems, one standard error is about 1.5
points.
