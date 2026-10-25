# Using Molten: command line and OpenAI-compatible server

## Command line: `molten run` and `molten chat`

```bash
molten run --model ~/models/lfm2.5 --prompt "Explain unified memory in two sentences."
```

LFM2.5 is a reasoning model: its chat template opens a `<think>` block. On a terminal the reasoning is printed dimmed
before the answer; when stdout is not a terminal only the answer is printed (`--show-reasoning` and
`--hide-reasoning` override this). A stats line on stderr gives the token counts, the time to the first token and the
decode rate.

Sampling defaults to the model's `generation_config.json` (temperature 0.1, top_k 50, repetition_penalty 1.05) with a
random seed, printed on stderr; the same seed reproduces the same text. `--temperature`, `--top-k`, `--top-p`,
`--repetition-penalty` and `--seed` set single parameters, `--greedy` decodes with argmax:

```bash
molten run --model ~/models/lfm2.5 --prompt "Write a haiku about autumn." --temperature 0.7 --seed 42
molten run --model ~/models/lfm2.5 --prompt "The capital of France is" --raw --greedy --max-tokens 16
```

`--raw` skips the chat template (the text is encoded as is and the raw continuation printed), `--system` adds a
system message, `--max-tokens` limits the output and `--json` prints one JSON object with the prompt and output token
ids, the text split into reasoning and content, and the timings.

```bash
molten chat --model ~/models/lfm2.5
```

is a multi-turn chat with streamed output (`/reset` clears the conversation, `/exit` or Ctrl-D quits). Each turn
reuses the previous prompt's prefix, so only the new text is prefilled.

## `molten serve`: OpenAI-compatible HTTP server

```
molten serve --model DIR [--host 127.0.0.1] [--port 8080] [--context 131072] [--served-model-name NAME]
             [--api-key KEY] [--no-reasoning-split] [--no-prefix-cache]
```

The server is part of the `molten` executable. It loads the model, compiles the Metal pipelines with a short warm-up
generation, then prints `molten: serving NAME on http://HOST:PORT/v1`. It speaks HTTP/1.1 on `std::net` with
keep-alive, chunked request bodies and `Expect: 100-continue`; streamed responses are server-sent events over
chunked transfer encoding.

The engine is single-sequence: requests are served **one at a time, in arrival order**. Connections are accepted
concurrently and the others wait in the queue. Each finished request logs one line on stderr with the prompt, cached
and completion token counts, the queue time, the time to the first token and the decode rate.

| flag | meaning |
|---|---|
| `--host`, `--port` | listen address (default `127.0.0.1:8080`) |
| `--context N` | context length in tokens (default 131072; the KV cache takes 20 KB per token) |
| `--served-model-name NAME` | the id `/v1/models` reports (default: the model directory's name). Requests may name any model |
| `--api-key KEY` | require `Authorization: Bearer KEY` on `/v1/*` (`/health` stays open) |
| `--no-reasoning-split` | return the reasoning inside `content` (see below) |
| `--no-prefix-cache` | prefill every prompt from scratch (see below) |

## Endpoints

| endpoint | |
|---|---|
| `POST /v1/chat/completions` | chat completion, streamed or not |
| `POST /v1/completions` | text completion of a raw prompt (no chat template) |
| `GET /v1/models`, `GET /v1/models/{id}` | the one served model, with `max_model_len` |
| `GET /health` | `{"status": "ok"}` once the model is loaded |

## Request parameters

| field | endpoints | default | notes |
|---|---|---|---|
| `messages` | chat | required | roles `system`, `developer` (rendered as `system`), `user`, `assistant`, `tool`; `content` is a string, `null` or a list of `{"type": "text", "text": ...}` parts. Assistant `tool_calls` (with `arguments` as a JSON string, the OpenAI wire format, or an object) and `reasoning_content` are passed to the template |
| `prompt` | completions | required | a string, a list of one string, or a list of token ids |
| `max_tokens`, `max_completion_tokens` | both | chat: the rest of the context; completions: 16 | prompt + max_tokens must fit the context, else 400 |
| `temperature` | both | 0.1 | below 1e-5: greedy argmax decoding (then `top_k`, `top_p` and `repetition_penalty` do not apply) |
| `top_p` | both | 1.0 | (0, 1] |
| `top_k` | both | 50 | extension; 0 or -1 = off |
| `repetition_penalty` | both | 1.05 | extension; covers the prompt and every generated token |
| `seed` | both | random | the same seed and request give the same output |
| `stop` | both | none | a string or a list of strings; the output ends before the first match (the match is not returned) |
| `stop_token_ids` | both | none | extension; the token is counted in `completion_tokens` |
| `ignore_eos` | both | false | extension: do not stop at `<|im_end|>` |
| `skip_special_tokens` | both | true | extension |
| `stream`, `stream_options.include_usage` | both | false | SSE `data: {...}` chunks, then `data: [DONE]` |
| `tools`, `tool_choice` | chat | none, `auto` | tools are rendered by the chat template; `tool_choice: "none"` keeps them in the prompt but does not parse calls |
| `chat_template_kwargs` | chat | none | extra template variables, e.g. `{"preserve_thinking": true}` |
| `n`, `best_of` | both | 1 | only 1 |
| `response_format` | both | text | only `{"type": "text"}` |

Unspecified sampling fields take the model's `generation_config.json` values individually (temperature 0.1, top_k 50,
top_p 1.0, repetition_penalty 1.05). Sampling is the engine's GPU sampler (Hugging Face/vLLM semantics, see
[PROTOCOL.md](PROTOCOL.md#sampling)). Fields of the OpenAI API that are not listed are ignored (`model`, `user`,
`metadata`, ...), except these, which return 400: `logprobs` / `top_logprobs`, a non-empty `logit_bias`, non-zero
`presence_penalty` / `frequency_penalty`, `echo`, `suffix`, `tool_choice` other than `auto`/`none`, and non-text
content parts.

## Responses

Chat: `{"id": "chatcmpl-...", "object": "chat.completion", "created", "model", "choices": [{"index": 0, "message":
{"role": "assistant", "content", "reasoning_content", "tool_calls"?}, "logprobs": null, "finish_reason"}],
"usage"}`. Completions: `"object": "text_completion"` with `choices[0].text`.

`usage` is `{"prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details": {"cached_tokens"}}`;
`completion_tokens` includes a final EOS or stop token, `cached_tokens` counts the prompt tokens reused from the
prefix cache. `finish_reason` is `stop` (EOS, a stop token or a stop string), `length` (`max_tokens` or the context)
or `tool_calls`.

Streaming chat sends a first chunk with `{"role": "assistant", "content": ""}`, then deltas with `reasoning_content`,
`content` or `tool_calls`, then an empty delta with the `finish_reason`, then (with `include_usage`) a chunk with
`"choices": []` and the `usage`, then `data: [DONE]`. Streaming completions send `text` deltas the same way.

Errors are OpenAI-style: `{"error": {"message", "type", "param", "code"}}` with status 400 (`invalid_request_error`:
malformed JSON, invalid fields, a prompt beyond the context), 401 (`authentication_error`), 404, 405, 413 (body over
64 MiB) or 500 (`server_error`). An error during a stream is sent as a `data: {"error": ...}` event before `[DONE]`.

## Reasoning

The chat template's generation prompt opens a reasoning block (`<|im_start|>assistant\n<think>`), so generation
starts inside it. As vLLM's reasoning parsers do, the text up to `</think>` is returned as `reasoning_content` and
the text after it as `content` (`null` when empty); an output without `</think>` (for example cut by `max_tokens`) is
all reasoning. A `<think>` at the very start of the output is also accepted. Streamed deltas are split the same way.
With `--no-reasoning-split` the whole output is returned in `content` (it then contains `</think>`, but not the
opening tag, which is part of the prompt).

When a conversation is sent back, the template drops the reasoning of assistant turns before the last user message;
`reasoning_content` on the latest assistant turns (or `chat_template_kwargs: {"preserve_thinking": true}`) keeps it.

## Tool calls

The model writes calls as a Pythonic list between `<|tool_call_start|>` and `<|tool_call_end|>`:
`<|tool_call_start|>[get_weather(city="Paris", days=3)]<|tool_call_end|>`. When the request has `tools` (and
`tool_choice` is not `none`), each such block in `content` is parsed: function names, keyword arguments, and literal
values (strings with Python escapes, integers, floats, `True`/`False`/`None` or `true`/`false`/`null`, lists, tuples
and dicts). They are returned as OpenAI `tool_calls` with `arguments` as a JSON object string
(`{"city": "Paris", "days": 3}`), the text outside the blocks stays in `content` (`null` when only whitespace), and
`finish_reason` becomes `tool_calls`. A block that does not parse is left in `content` as raw text. When streaming,
each call arrives whole in one `tool_calls` delta (with `index`, `id`, `type`, `function.name` and the complete
`function.arguments`).

## Prompt-prefix reuse

The server keeps one snapshot of the recurrent state (the conv states, the position and the seen-token set; about
0.25 MB) at a split point of the last prompt, and the KV cache rows below it stay in place. The split point is the
last `<|im_start|>` of the prompt: the next chat turn's prompt is the same text up to there and then re-renders this
turn's assistant message (the template drops its reasoning). Without `<|im_start|>` after the reused part it is the
prompt's last token. When a new prompt's token ids start with the snapshot's ids, the snapshot is restored and only
the rest of the prompt is prefilled; `usage.prompt_tokens_details.cached_tokens` reports the reused length. Any other
prompt starts from scratch and replaces the snapshot.

To take the snapshot, the prefill is cut at the split point (one extra GPU submission). Outputs are equally valid
computations of the model, but the chunking of a reused or cut prefill differs from a single prefill of the whole
prompt, so greedy outputs can differ from `--no-prefix-cache` in rare near-ties, as with any prefix cache. A cut at
the prompt's last token is the reference partition itself, so raw prompts without `<|im_start|>` give exactly the
protocol path's tokens either way. `--no-prefix-cache` never cuts.

Measured on an M5 Max (128 GB) while another engine process ran a dataset evaluation on the same GPU, so absolute
times are about twice the idle ones: a six-turn conversation with a 3.6K-token system prompt, greedy, fixed assistant
replies (identical prompts in both runs), two runs each, client-side time to the first streamed token:

| turn | prompt tokens | cached tokens | TTFT with prefix reuse | TTFT with `--no-prefix-cache` |
|---:|---:|---:|---:|---:|
| 1 | 3632 | 0 | 783 / 639 ms | 773 / 527 ms |
| 2 | 3655 | 3628 | 103 / 103 ms | 569 / 534 ms |
| 3 | 3685 | 3651 | 105 / 106 ms | 547 / 539 ms |
| 4 | 3723 | 3681 | 107 / 108 ms | 541 / 547 ms |
| 5 | 3756 | 3719 | 108 / 106 ms | 544 / 555 ms |
| 6 | 3783 | 3752 | 105 / 106 ms | 549 / 550 ms |

`molten chat` uses the same reuse between turns.

## Example

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
}'
```

returns (abridged)

```json
{"choices": [{"finish_reason": "tool_calls", "index": 0, "logprobs": null,
  "message": {"role": "assistant", "content": null,
              "reasoning_content": "The user wants the weather in Paris. I should call get_weather ...",
              "tool_calls": [{"id": "call_...", "type": "function",
                              "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}}],
 "usage": {"prompt_tokens": 109, "completion_tokens": 81, "total_tokens": 190, "prompt_tokens_details": {"cached_tokens": 0}}}
```

## Limitations

- One sequence: requests queue; there is no batching. A client that disconnects during a streamed response stops its
  generation; a non-streamed request runs to the end.
- No logprobs, `n` > 1, logit bias, presence/frequency penalties, structured output or images.
- `tool_choice` `required` or a named function is not supported; tool calls are parsed after the model writes them.
- One prefix snapshot (the last prompt's); a request that does not extend it evicts it.
