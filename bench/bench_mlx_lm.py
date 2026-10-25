"""The same measurement with mlx-lm's own generation loop (mlx_lm.generate.generate_step, unmodified).

For each depth P and run: a fresh prompt cache; generate_step(prompt, model, max_tokens=N+1) with the default
greedy sampler and prefill_step_size; the time to the first yielded token is the prefill (as for Molten,
it includes the first token); the time from the first to the (N+1)-th yielded token is N decode steps.
Prompts are the same nonce token ids as bench_engine.py.

usage: python3 bench/bench_mlx_lm.py --model DIR [--depths ...] [--decode 64] [--runs 5] [--warmup 1] [--out ...]
(needs `pip install mlx-lm`; tested with mlx-lm 0.32.0 on MLX 0.32.3)
"""
import argparse
import json
import platform
import statistics as st
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm import load
from mlx_lm.generate import generate_step

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lfm_engine import nonce_prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--depths", default="256,1024,4096,16384")
ap.add_argument("--decode", type=int, default=64)
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--out")
args = ap.parse_args()
depths = [int(x) for x in args.depths.split(",")]

model, _tokenizer = load(args.model)
rows = []
for P in depths:
    samples = []
    for r in range(args.warmup + args.runs):
        prompt = mx.array(nonce_prompt(P, offset=1_000_003 * (r + 1) + 7 * P))
        t0 = time.perf_counter()
        first = last = None
        for n, _ in enumerate(generate_step(prompt, model, max_tokens=args.decode + 1)):
            now = time.perf_counter()
            if n == 0:
                first = now
            last = now
        if r >= args.warmup:
            samples.append((first - t0, last - first))
        mx.clear_cache()
    pf = st.median(s[0] for s in samples)
    dc = st.median(s[1] for s in samples)
    rows.append({"depth": P, "prefill_s": pf, "decode_s": dc, "prefill_tok_s": P / pf,
                 "decode_tok_s": args.decode / dc, "samples": samples})
    print(f"mlx-lm  P={P:6d}  prefill {P / pf:8.1f} tok/s   decode {args.decode / dc:6.1f} tok/s", flush=True)

result = {"backend": f"mlx-lm {mlx_lm.__version__} (MLX {mx.__version__})", "model": str(args.model),
          "decode_tokens": args.decode, "runs": args.runs, "warmup": args.warmup, "host": platform.platform(),
          "time": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
