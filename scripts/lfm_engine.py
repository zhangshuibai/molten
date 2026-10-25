"""Minimal client for the engine's JSONL protocol (`molten --model DIR --contract-json JSON`, see docs/PROTOCOL.md).
Python standard library only.

    from lfm_engine import Engine
    eng = Engine("target/release/molten", "/path/to/LFM2.5-24B-A2B-MLX-4bit")
    eng.reset()
    tokens = eng.generate(prompt_ids, 64)       # greedy; exactly `count` tokens
    more = eng.generate([], 64)                 # continue from the pending token
    eng.reset()
    s = {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.05, "seed": 7}
    tokens = eng.generate(prompt_ids, 64, sampling=s)   # sampled; reproducible for a seed
    eng.close()
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "contract.json"
DEFAULT_ENGINE = ROOT / "target" / "release" / "molten"


class EngineError(RuntimeError):
    pass


class Engine:
    def __init__(self, engine, model, contract=None, stderr=None):
        """Start the engine process (`engine` None: DEFAULT_ENGINE, this checkout's release build) and wait until it
        has loaded the model (the `hello` reply)."""
        engine = engine or DEFAULT_ENGINE
        text = Path(contract or DEFAULT_CONTRACT).read_text()
        self.contract_json = json.dumps(json.loads(text), separators=(",", ":"))
        self.contract_sha256 = hashlib.sha256(self.contract_json.encode()).hexdigest()
        self.proc = subprocess.Popen(
            [str(Path(engine).resolve()), "--model", str(model), "--contract-json", self.contract_json],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr, bufsize=0)  # stderr: inherited by default (bounded diagnostics only)
        self.id = 0
        self.generation = 0
        t = time.perf_counter()
        self.request("hello")
        self.load_seconds = time.perf_counter() - t

    def request(self, operation, new_generation=False, **fields):
        self.id += 1
        if new_generation:
            self.generation += 1
        req = {"version": 2, "id": self.id, "operation": operation, "generation": self.generation,
               "model_contract_sha256": self.contract_sha256, **fields}
        self.proc.stdin.write((json.dumps(req, separators=(",", ":")) + "\n").encode())
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise EngineError(f"engine exited without replying to {operation!r}")
        reply = json.loads(line)
        if reply.get("ok") is not True:
            raise EngineError(f"{operation} failed: {line[:300]!r}")
        return reply

    def reset(self, mode="greedy"):
        """Clear all state. `greedy` for generate, `teacher_forced` for forward."""
        return self.request("reset", new_generation=True, mode=mode)

    def generate(self, prompt, count, sampling=None):
        """Generation of exactly `count` (1..256) tokens. An empty prompt continues from the pending token.

        Greedy without `sampling`. `sampling` is a dict with any of temperature (0 = greedy), top_k (0 = off),
        top_p (1 = off), repetition_penalty (1 = off) and seed (an integer in [0, 2**64)); see docs/PROTOCOL.md."""
        fields = {"prompt": list(prompt), "count": count}
        if sampling is not None:
            fields["sampling"] = dict(sampling)
        return self.request("generate", new_generation=True, **fields)["tokens"]

    def timed_generate(self, prompt, count, sampling=None):
        t = time.perf_counter()
        tokens = self.generate(prompt, count, sampling)
        return tokens, time.perf_counter() - t

    def close(self):
        """Close stdin; the engine exits cleanly on EOF."""
        try:
            self.proc.stdin.close()
            return self.proc.wait(timeout=60)
        except Exception:
            self.proc.kill()
            return self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def nonce_prompt(length, offset=0, vocab=128000):
    """Deterministic pseudo-random token ids (Knuth multiplicative hash), as used by the benchmarks."""
    return [((i * 2654435761) & 0xFFFFFFFF) % vocab for i in range(offset, offset + length)]
