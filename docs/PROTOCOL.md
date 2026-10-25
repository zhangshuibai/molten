# Engine protocol (JSONL over stdio, version 2)

Molten's engine protocol mode (`molten` without a subcommand) loads the model at start-up and then serves requests: one JSON object per
line on stdin, one reply per line on stdout. Stderr carries only bounded diagnostics. It exits with status 0 when stdin
closes. `scripts/lfm_engine.py` is a complete client using only the Python standard library.

```
molten --model <checkpoint dir> --contract-json '<compact JSON>'
```

- `--model`: the `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` checkpoint directory. It holds three safetensors shards and
  `model.safetensors.index.json`. Every tensor name, shape and dtype is checked at load time.
- `--contract-json`: session limits. `contract.json` in this repository is the configuration Molten was
  evaluated with. `context_tokens` (up to 131072) is required.

## Envelope

Every request carries these fields, and every reply echoes them unchanged:

| field | value |
|---|---|
| `version` | `2` |
| `id` | strictly increasing integer |
| `operation` | `hello`, `reset`, `forward`, `generate`, `checkpoint` or `restore` |
| `generation` | a state epoch chosen by the client; send a new value with `reset`, `forward`, `generate` and `restore` |
| `model_contract_sha256` | 64 lowercase hex digits (the client uses the sha256 of the contract JSON) |

Replies also add `"ok": true`. There are no error replies: a malformed or failing request makes Molten print one
line on stderr and exit with status 1.

## Operations

| operation | request fields | reply fields | meaning |
|---|---|---|---|
| `hello` | none | `ready`, `capabilities` | sent first; the reply comes once the model is loaded |
| `reset` | `mode`: `"greedy"` or `"teacher_forced"` | `state` | clears caches, position, pending token, logits and the seen-token set |
| `generate` | `prompt` (token ids), `count` (1..256), optional `sampling` | `tokens`, `state`, `device_completed` | `greedy` mode (the generation mode, sampled or not). A non-empty prompt starts a sequence; an empty prompt continues from the pending (last) token. Returns exactly `count` tokens, ignoring EOS: the client decides when to stop. Greedy without `sampling`, else sampled (below) |
| `forward` | `tokens`, `logits_positions` | `state`, `logits_positions`, `device_completed` | teacher-forced mode. Consumes tokens and keeps the full logits rows of the requested positions for a later tensor read |
| `checkpoint` | none | `handle`, `state` | deep copy of all state; `handle` is an opaque string |
| `restore` | `handle` | `restored`, `state` | restores a checkpoint exactly |

`state` is `{mode, consumed_tokens, pending_token, ended}`. After a `generate` of N tokens, `consumed_tokens` grows by
the prompt length (or by 1 for a continuation) plus N - 1, and `pending_token` is the last token returned.

Greedy ties go to the lowest token id. Decoding is bitwise deterministic run to run, and a continuation produces
exactly the tokens a single longer `generate` would.

## Sampling

`generate` takes an optional `sampling` object. When it is absent or null, or its `temperature` is 0, the request
runs the greedy path unchanged (the other fields are then ignored).

| field | type | default | meaning |
|---|---|---|---|
| `temperature` | number, 0 or >= 1e-6 | 1 | logits are divided by it; 0 = greedy |
| `top_k` | integer >= 0 | 0 (off) | keep the tokens whose (penalized, scaled) logit is at least the k-th largest (ties kept); k >= 128000 is off |
| `top_p` | number in (0, 1] | 1 (off) | then keep the smallest set of most likely tokens whose probability reaches `top_p` (a group of equal logits at the boundary is kept whole); at least one token is always kept |
| `repetition_penalty` | number in [1e-6, 1e6] | 1 (off) | for each distinct token id already in the sequence: logit / p if the logit > 0, else logit * p |
| `seed` | integer in [0, 2^64) | 0 | the random seed |

Unknown fields and out-of-range values are errors. The steps follow Hugging Face and vLLM: repetition penalty,
temperature, top-k, top-p, renormalization, draw. The logits are the float32 values of the BF16 logits that greedy
decoding compares; the arithmetic is float32.

- The seen-token set holds every distinct token id of the sequence: the prompt tokens and the returned tokens of every
  `generate` since the last `reset`, greedy or sampled. `reset` clears it; `checkpoint` and `restore` carry it.
- The draw samples the renormalized distribution exactly, up to float32 rounding (the Gumbel-max trick): it returns
  the kept token with the largest `logit' + G` (ties to the lowest id), where `logit'` is the penalized logit over the
  temperature and `G` a Gumbel(0, 1) variate made from one 32-bit word of Philox4x32-10 with key (seed mod 2^32,
  seed / 2^32) and counter (token id / 4, position, 0x53414D50, 0), word token id mod 4. `position` is the absolute
  position of the drawn token: the number of tokens before it in the sequence.
- Sampling is bitwise deterministic: the same seed and the same request sequence give the same tokens, and a
  generation split into continuation requests gives the tokens of one longer request (a seed applies per position,
  not per request). Switching between greedy and sampled requests within a sequence is allowed.
- The whole draw runs on the GPU inside the decode loop; tokens do not return to the host between steps.

Tensor reads (`tensor_manifest`, `tensor_chunk`, `tensor_end`; logits rows and the logical KV/conv state as base64
float32) exist for correctness tooling and are not needed for generation. See `src/protocol.rs`.

## Example session

```
> {"version":2,"id":1,"operation":"hello","generation":0,"model_contract_sha256":"<64 hex>"}
< {"version":2,"id":1,"operation":"hello","generation":0,"model_contract_sha256":"<64 hex>","ok":true,"ready":true,"capabilities":[...]}
> {"version":2,"id":2,"operation":"reset","generation":1,"model_contract_sha256":"<64 hex>","mode":"greedy"}
> {"version":2,"id":3,"operation":"generate","generation":2,"model_contract_sha256":"<64 hex>","prompt":[124894,...],"count":64}
< {..."ok":true,"tokens":[...64 ids...],"state":{"mode":"greedy","consumed_tokens":...,"pending_token":...,"ended":false},"device_completed":true}
> {"version":2,"id":4,"operation":"generate","generation":3,"model_contract_sha256":"<64 hex>","prompt":[],"count":64}
> {"version":2,"id":5,"operation":"generate","generation":4,"model_contract_sha256":"<64 hex>","prompt":[],"count":64,"sampling":{"temperature":0.1,"top_k":50,"repetition_penalty":1.05,"seed":7}}
```
