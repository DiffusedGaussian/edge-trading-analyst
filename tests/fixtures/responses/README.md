# Recorded model responses

Raw output captured verbatim from a real `llama-server`, replayed in CI by
[`tests/test_recorded_responses.py`](../../test_recorded_responses.py).

Every other parser test in this repo uses hand-written strings, which encode what
we *imagine* a 1B model emits. These files are what one actually emitted. They
make parser robustness a regression gate against reality rather than against
imagination, and they cost nothing to run: no GPU, no model, no network.

## This directory is intentionally empty

It has to be populated from a real device run — the captures cannot be written by
hand without defeating their entire purpose. Until then
`tests/test_recorded_responses.py` **skips**, and the skip shows up in the pytest
summary (`addopts = -ra`) rather than passing quietly.

## Populating it

On the Jetson, or anywhere with a `llama-server` on an OpenAI-compatible port:

```bash
./llama-server -hf ggml-org/gemma-3-1b-it-GGUF -ngl 99   # quick tier, :8080

uv run python -m eval.run_fixtures \
    --base-url http://localhost:8080 \
    --model-name gemma-3-1b-it \
    --k 1 \
    --export-fixtures tests/fixtures/responses/
```

That writes one `<model>__<fixture_id>__<sample_idx>.txt` per sample plus an
`expected.yaml` sidecar recording, per file, the expected fallbacks set and each
check's expected pass/fail. Commit both.

Re-running merges rather than overwrites, so the directory accumulates captures
across models and runs — each is a distinct regression case.

## What to commit

- **Include at least one genuinely malformed response.** Those are the valuable
  ones; a directory of only well-formed output proves the parser handles the easy
  case, which the hand-written tests already cover. There is a test asserting
  this once the directory is non-empty.
- Prefer captures from the model tier you actually deploy. A 32B model's output
  tells you nothing about how the forgiving parsers behave under a 1B model.

## After an intentional parser change

`expected.yaml` is a golden file: it records what the current parsers and checks
produce. If a parser change is deliberate, re-export and review the sidecar diff —
that diff *is* the behavioural change, stated in terms of real model output.
