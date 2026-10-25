"""Print markdown tables (decode, time to first token, prefill) from the results of bench_engine.py,
bench_mlx_lm.py and bench_llama_cpp.sh. llama-bench reports no TTFT; it is estimated as the prefill time of the
prompt plus one decode step at that depth.
usage: python3 bench/compare.py [results]"""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
eng = {r["depth"]: r for r in json.loads((out / "engine.json").read_text())["rows"]}
mlx = {r["depth"]: r for r in json.loads((out / "mlx_lm.json").read_text())["rows"]}
tg = {r.get("n_depth", 0): r["avg_ts"] for r in json.loads((out / "llama_cpp_tg.json").read_text())}
pp = {r["n_prompt"]: r["avg_ts"] for r in json.loads((out / "llama_cpp_pp.json").read_text())}


def fmt(x, spec):
    return format(x, spec) if x is not None else ""


print("| context | Molten decode | mlx-lm decode | llama.cpp decode | vs mlx-lm | vs llama.cpp |")
print("|---:|---:|---:|---:|---:|---:|")
for d in sorted(eng):
    e, m, l = eng[d]["decode_tok_s"], mlx.get(d, {}).get("decode_tok_s"), tg.get(d)
    print(f"| {d} | {e:.1f} | {fmt(m, '.1f')} | {fmt(l, '.1f')} | {fmt(m and e / m, '.2f')} | {fmt(l and e / l, '.2f')} |")
print()
print("| prompt | Molten TTFT ms | mlx-lm TTFT ms | llama.cpp TTFT ms (est.) |")
print("|---:|---:|---:|---:|")
for d in sorted(eng):
    m = mlx.get(d, {}).get("prefill_s")
    l = d / pp[d] + 1 / tg[d] if d in pp and d in tg else None
    print(f"| {d} | {1000 * eng[d]['prefill_s']:.1f} | {fmt(m and 1000 * m, '.1f')} | {fmt(l and 1000 * l, '.1f')} |")
print()
print("| prompt | Molten prefill tok/s | mlx-lm prefill tok/s | llama.cpp prefill tok/s |")
print("|---:|---:|---:|---:|")
for d in sorted(eng):
    m, l = mlx.get(d, {}).get("prefill_tok_s"), pp.get(d)
    print(f"| {d} | {eng[d]['prefill_tok_s']:.0f} | {fmt(m, '.0f')} | {fmt(l, '.0f')} |")
