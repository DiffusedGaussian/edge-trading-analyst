# Tier 0 run results

Each `make eval-fixtures` run writes `<run_id>.json` here, mirroring exactly what
went into the `eval_runs` / `eval_samples` tables: SQLite for querying, JSON for
committing a baseline and diffing it in review.

Ad-hoc run files are gitignored. One file is not: `baseline.json`.

## baseline.json — the comparison floor

`baseline.json` is the committed reference run every candidate model is measured
against, and it must be a **real** run. Produce it on the device you actually
deploy to:

```bash
./llama-server -hf ggml-org/gemma-3-1b-it-GGUF -ngl 99        # quick tier, :8080

make eval-fixtures MODEL=gemma-3-1b-it BASE_URL=http://localhost:8080 K=5
cp eval/results/<run_id>.json eval/results/baseline.json
git add -f eval/results/baseline.json
```

It is deliberately absent from this commit rather than filled in with plausible
numbers: an invented baseline is worse than none, because every later comparison
would be measured against a fiction and would look rigorous while doing it.

Once it exists, compare a candidate against it with:

```bash
make eval-compare A=baseline B=<new_run_id>
```

## Reading a baseline

The headline numbers, in the order they matter:

1. **fallback rate per sentinel field** — if a model can't emit a parseable
   `LABEL`, nothing downstream of it means anything.
2. **truncation rate** — a `length` finish_reason invalidates the sample rather
   than scoring it.
3. **per-check pass rate**, worst first.
4. **label-flip rate** — instability at fixed temperature and seed bounds how
   much of any A-vs-B difference could be noise.
5. **prefill and decode tok/s**, separately, each with its spread.

A run whose `git_sha` differs from the code you're comparing against is a
different experiment. The field is recorded for exactly that reason.
