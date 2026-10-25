#!/bin/bash
# llama.cpp's own benchmark at the same depths: text generation of N tokens after a context of D tokens
# (llama-bench -d D -n N -p 0; all layers on the GPU), and prompt processing of D tokens (-p D -n 0).
# llama.cpp runs a GGUF (the official LiquidAI LFM2.5-24B-A2B-QAD-Q4_0.gguf), not the MLX 4-bit checkpoint, so
# the weights differ in format (Q4_0 blocks of 32 vs MLX affine groups of 64; both about 13.5 GB).
#
# usage: bench/bench_llama_cpp.sh LLAMA_BENCH GGUF [DEPTHS=256,1024,4096,16384] [N=64] [RUNS=5] [OUT_DIR=results]
set -euo pipefail
BENCH=$1; GGUF=$2; DEPTHS=${3:-256,1024,4096,16384}; N=${4:-64}; RUNS=${5:-5}; OUT=${6:-results}
mkdir -p "$OUT"
"$BENCH" -m "$GGUF" -ngl 99 -p 0 -n "$N" -d "$DEPTHS" -r "$RUNS" -o json > "$OUT/llama_cpp_tg.json"
"$BENCH" -m "$GGUF" -ngl 99 -p "$DEPTHS" -n 0 -r "$RUNS" -o json > "$OUT/llama_cpp_pp.json"
python3 - "$OUT" <<'EOF'
import json, sys
out = sys.argv[1]
for name in ("tg", "pp"):
    for row in json.load(open(f"{out}/llama_cpp_{name}.json")):
        d = row.get("n_depth", 0) if name == "tg" else row["n_prompt"]
        print(f"llama.cpp {name}  depth={d:6d}  {row['avg_ts']:8.1f} tok/s  (+-{row['stddev_ts']:.1f})")
EOF
