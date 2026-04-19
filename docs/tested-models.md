# Tested models

Live probes performed against a running Ollama instance
(`http://localhost:11434` during development). Results are reproducible
with the smoke suites in `tests/`.

This document captures **what we actually ran**, not a feature matrix of
models denkwerk theoretically supports. Entries are added after a live test
session produces green results.

## Flow orchestrator suite (`tests/flows_smoke.rs`)

Six flows — sequential, concurrent, handoff, group_chat, magentic, dispatch
— run end-to-end with `think=off`, `num_ctx=8192`, two agents per flow where
applicable. Reported column: tests passed / total.

| model                   | family | flows  | notes                                                  |
| ----------------------- | ------ | ------ | ------------------------------------------------------ |
| `qwen3.6:35b-a3b`       | qwen3  | 6 / 6  | default; needs prefill compensation (applied).         |
| `qwen3.5:35b`           | qwen3  | 6 / 6  | same prefill semantics as qwen3.6.                     |
| `gemma4:e4b`            | gemma4 | 6 / 6  | ~8B params; compensation not needed but harmless.      |
| `gemma4:31b`            | gemma4 | 6 / 6  | 31B; behavior identical to e4b.                        |
| `gpt-oss:120b`          | gpt-oss| 1 / 6  | reasoning-first model; leaks chain-of-thought into `content` on short-instruction prompts. Would likely work with `think=on` + revised prompts; not retuned yet. |

Reproduce:

```sh
OLLAMA_SMOKE_URL=http://localhost:11434 \
OLLAMA_SMOKE_MODEL=qwen3.6:35b-a3b \
  cargo test --test flows_smoke -- --ignored --test-threads=1 --nocapture
```

Swap `OLLAMA_SMOKE_MODEL` to any tag you want to probe.

## Ollama provider suite (`tests/ollama_smoke.rs`)

Provider-level probes covering text + reasoning, think on/off, preserve
thinking, streaming, vision, max context, **uncapped generation**, and
**reasoning effort mapping**. Some tests assert on qwen-specific values
(e.g. 262,144 max ctx) and are intentionally model-coupled.

| model                   | text/reasoning | think off | preserve_thinking | streaming | vision | max ctx | uncapped | effort→Auto |
| ----------------------- | -------------- | --------- | ----------------- | --------- | ------ | ------- | -------- | ----------- |
| `qwen3.6:35b-a3b`       | yes            | yes       | yes               | yes       | —      | 262,144 | yes      | yes         |
| `qwen3-vl:8b-instruct`  | —              | —         | —                 | —         | yes    | 262,144 | —        | —           |

- **uncapped**: `CompletionRequest::without_max_tokens()` produces a
  natural-stop completion well past any default token cap (observed ~1.7k
  completion tokens with reasoning on).
- **effort→Auto**: `ThinkMode::Auto` + `with_reasoning_effort(High)` emits
  `think=true`; Auto without an effort emits `think=false` (deterministic,
  not server-default).

Reproduce:

```sh
OLLAMA_SMOKE_URL=http://localhost:11434 \
  cargo test --test ollama_smoke -- --ignored --test-threads=1
```

## Model-specific interactions we compensate for

### Qwen3 / Qwen2 prefill on trailing assistant (`src/flows/prefill.rs`)

Ollama's `qwen3.5` renderer treats a trailing `role="assistant"` message as a
prefill cue — it does not append `<|im_end|>\n<|im_start|>assistant\n` and
leaves the prompt mid-turn. The model's first generated token is then
`<|im_end|>`: empty content, `eval_count: 1`.

This breaks naive multi-agent chaining. Every orchestrator that appends an
agent's reply to a shared transcript and then calls the next agent hits it.

Compensation: `flows::prefill::history_for_llm` injects a synthetic user
turn into the per-call `Vec<ChatMessage>` (never the durable transcript) for
model names that match the Qwen2/Qwen2.5/Qwen3 family. Non-matching models
pass through unchanged via `Cow::Borrowed`.

Confirmed affected: `qwen3.6`, `qwen3.5` (both probed live).
Confirmed unaffected: `gemma4:e4b`, `gemma4:31b`, `gpt-oss:120b`.

Add a model family to `needs_user_alternation` only after live-probing it —
the right signal is a trailing-assistant request returning `eval_count: 1`
and empty `content`.

## Adding a new model to this list

1. Run the flow suite: `OLLAMA_SMOKE_MODEL=<tag> cargo test --test flows_smoke -- --ignored --test-threads=1`.
2. If flows fail, inspect whether the failure is the prefill issue (empty
   content from the second agent in sequential). If so, extend
   `needs_user_alternation` in `src/flows/prefill.rs` and re-run.
3. If failures are instruction-following or chain-of-thought leakage
   (gpt-oss pattern), note it as a caveat rather than a framework bug.
4. Add a row to the table above.
