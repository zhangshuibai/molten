"""Decode (and prefill) throughput of Molten at several context depths.

For each depth P and run: reset; generate(P, 1) is timed as the prefill (it also produces the first token);
generate((), N) is timed as N decode steps. Reported: medians over the runs of P/prefill_s and N/decode_s.
Prompts are deterministic nonce token ids (scripts/lfm_engine.nonce_prompt), fresh for every run.
Greedy decoding unless --temperature is above 0; the sampling flags then apply to both requests of a run.

usage: python3 bench/bench_engine.py --engine target/release/molten --model DIR [--depths 256,1024,4096,16384]
           [--decode 64] [--runs 5] [--warmup 1] [--out results/engine.json]
           [--temperature T] [--top-k K] [--top-p P] [--repetition-penalty R] [--seed S]
"""
import argparse
import json
import platform
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lfm_engine import Engine, nonce_prompt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="target/release/molten")
ap.add_argument("--model", required=True)
ap.add_argument("--depths", default="256,1024,4096,16384")
ap.add_argument("--decode", type=int, default=64)
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--out")
ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default)")
ap.add_argument("--top-k", type=int, default=0)
ap.add_argument("--top-p", type=float, default=1.0)
ap.add_argument("--repetition-penalty", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
depths = [int(x) for x in args.depths.split(",")]
sampling = None
if args.temperature > 0:
    sampling = {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty, "seed": args.seed}

rows = []
with Engine(args.engine, args.model) as eng:
    for P in depths:
        samples = []
        for r in range(args.warmup + args.runs):
            prompt = nonce_prompt(P, offset=1_000_003 * (r + 1) + 7 * P)
            eng.reset()
            _, pf = eng.timed_generate(prompt, 1, sampling)
            _, dc = eng.timed_generate([], args.decode, sampling)
            if r >= args.warmup:
                samples.append((pf, dc))
        pf = st.median(s[0] for s in samples)
        dc = st.median(s[1] for s in samples)
        row = {"depth": P, "prefill_s": pf, "decode_s": dc, "prefill_tok_s": P / pf,
               "decode_tok_s": args.decode / dc, "samples": samples}
        rows.append(row)
        print(f"engine  P={P:6d}  prefill {P / pf:8.1f} tok/s   decode {args.decode / dc:6.1f} tok/s", flush=True)

result = {"backend": "Molten", "engine": str(args.engine), "model": str(args.model),
          "sampling": sampling, "decode_tokens": args.decode, "runs": args.runs, "warmup": args.warmup, "host": platform.platform(),
          "time": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
