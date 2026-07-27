# Edge Trading Analyst

A multi-agent, LLM-driven market analyst running entirely on a single NVIDIA Jetson
Orin Nano Super (8GB), against a small watchlist, on an intraday cadence, driving a
**paper-trading** loop.

**This is a research / paper-trading project. It does not produce financial advice,
and no path in this system can place a real-money order.**

## Core idea

Determinism first; LLM only where language reasoning is genuinely irreducible.
Every capability falls into one of three buckets:

1. **Pure computation — no LLM, ever.** Technical indicators, gate thresholds, risk
   limits. Runs for the whole watchlist essentially for free, every cycle.
2. **Gating logic — no LLM.** Deterministic rules deciding whether a cycle even
   deserves an LLM call. The single biggest cost lever in the system.
3. **Language reasoning — LLM, rationed.** News/sentiment synthesis, the bull/bear
   debate, the trader's final judgment. What the scarce tokens are actually for.

The Jetson's real constraint (~11.5 tok/s on the 4B "deep" tier at 15W) means the
interesting engineering is deciding when *not* to run the model, not building a
bigger agent graph.

## Architecture / status

| Phase | What it does | Status |
|---|---|---|
| 1. Deterministic core | Market data → indicators (SMA/EMA/MACD/RSI) + fundamentals → persist to SQLite | ✅ done |
| 2. Materiality gate | Crossing/level rules decide if a cycle proceeds to reasoning | ✅ done |
| 3. News/Sentiment analyst (quick ~1B tier) | Prompt anchored on real indicators, not model recall; forgiving sentinel-format parser | ✅ done, live-validated on Jetson |
| 4. Bull/bear debate + Trader (deep ~4B tier) | Fixed-size debate state (no quadratic context resend); trader synthesizes final call | ✅ built, not yet live-validated |
| 5. Paper-trade state + reflection | Persist positions/P&L, feed outcomes back via similarity retrieval | ⬜ not started |
| 6. Scheduling | Pull-based deploy + systemd timer on the Jetson | ✅ deploy infra ready (`deploy/`) |
| 7. Measure & tune | Benchmark real token spend, tune gate thresholds | ⬜ not started |

No agent framework (LangChain/LangGraph) is used for Phases 3–4 — direct HTTP calls
to `llama-server`'s OpenAI-compatible endpoint, so prompt shape and turn count stay
fully under our control. TauricResearch's TradingAgents is used only as a design
reference (role decomposition, and its proven sentinel-text + lenient-regex parsing
pattern), not as a dependency.

## Project layout

```
src/edge_analyst/
  config.py         watchlist/config loading
  data_source.py    yfinance wrapper (OHLCV + fundamentals + news snapshot)
  indicators.py     SMA / EMA / MACD / RSI — pure functions, no LLM
  gate.py           materiality gate: crossing + level rules, OR-combined
  snapshot.py       glues fetch -> indicators -> gate into one TickerSnapshot
  store.py          SQLite persistence (bars, fundamentals, decisions + eval tables)
  pipeline.py       the orchestrator: snapshot -> persist -> gate -> LLM cascade
  llm_client.py     thin HTTP client for llama-server's OpenAI-compatible API
  llm_parsing.py    shared forgiving sentinel-field parser
  news_analyst.py   Phase 3: quick-tier news/sentiment analyst
  debate.py         Phase 4: bull/bear debate + trader synthesis
eval/
  checks.py         Tier 0: deterministic per-response checks, no LLM
  fixtures/         synthetic adversarial fixtures (one YAML per failure mode)
  run_fixtures.py   Tier 0 runner: fixtures x k samples -> eval_runs/eval_samples
  report.py         scorecards, A-vs-B comparison, judge + pairwise summaries
  rubric.py         Tier 1: per-criterion and pairwise judge prompts/parsers
  modal_app.py      Tier 1 on Modal: batched vLLM judging on one fp8 L40S
  calibrate.py      Tier 2: hand-labelling + Cohen's kappa against the judge
tests/              unit + smoke tests (design-decision regressions)
deploy/             pull-based Jetson deploy: systemd service + timer, deploy.sh
config/watchlist.yaml
```

## Running it

Phases 1–2 need no LLM and run anywhere:

```bash
make install   # uv sync --frozen --dev
make check     # lint + format-check + tests (what CI runs)
make run       # one cycle across the watchlist -> SQLite + gate outcome per ticker
```

Phases 3–4 need a running `llama-server` (OpenAI-compatible endpoint) reachable at
some `base_url`, e.g. on the Jetson:

```bash
./llama-server -hf ggml-org/gemma-3-1b-it-GGUF -ngl 99   # quick tier, default :8080

uv run python -m edge_analyst.pipeline AAPL http://localhost:8080          # single ticker, live cascade
uv run python -m edge_analyst.pipeline AAPL http://localhost:8080 --force  # demo the cascade even on a quiet gate

# The optional third arg labels which model produced the decision. Without it the
# row records `unknown`, and an unattributed decision can't be compared to another
# model's later — see "Evaluating a model" below.
uv run python -m edge_analyst.pipeline NVDA http://localhost:8081 olmoe-1b-7b
```

News is auto-fetched (bounded, pre-summarized via yfinance) when the gate is
material and a `base_url` is given. `run_cycle` prints the gate outcome plus, when
the cascade runs, the parsed `SentimentSignal`, `DebateState`, and `TraderDecision`
— and persists the news digest and every LLM output to SQLite (`decisions` /
`debate_turns` tables) alongside the deterministic bars/fundamentals.

## Evaluating a model

The question the harness exists to answer is "is model A better than model B on
this cascade", and answering it needs ground truth, attribution, and a measured
noise floor. Three tiers, cheapest first:

| Tier | What | Where | Cost | Cadence |
|---|---|---|---|---|
| 0 | Deterministic checks, no judge | Jetson / CI | free | every run |
| 1 | LLM-as-judge: per-criterion + pairwise | Modal L40S (fp8) | GPU-minutes | per bake-off |
| 2 | Human calibration set + Cohen's kappa | local, manual | your time | on judge change |

```bash
make eval-fixtures MODEL=gemma-3-1b-it BASE_URL=http://localhost:8080   # Tier 0
make eval-report RUN=latest
make eval-compare A=baseline B=<run_id>

modal secret create huggingface-secret HF_TOKEN=hf_...   # Tier 1, once
make eval-prewarm                                  # Tier 1, cache weights (no GPU)
make eval-judge                                    # Tier 1, per-criterion
make eval-pairwise A=gemma-3-1b-it B=olmoe-1b-7b   # Tier 1, A-vs-B

make eval-calibrate CMD="label --n 40"             # Tier 2, hand-label (resumable)
make eval-calibrate CMD=score                      # Tier 2, kappa vs the judge
```

The eval targets are not in CI: it has no GPU and no `llama-server`. Everything
model-independent — the checks, the fixture loader, the aggregation, and the
recorded-response replay — does run in `make check`.

### What a Tier 1 run costs, and what keeps it cheap

Judging 20 decisions is ~80 short prompts: tens of seconds of generation behind a
multi-minute model load. The load *is* the bill, so `eval/modal_app.py` is built
to pay it as rarely as possible.

- **fp8 weights on one L40S.** Qwen2.5-32B in bf16 is ~65GB against ~44GB usable
  and simply OOMs; fp8 halves it and Ada runs fp8 natively, so the cheaper card is
  also the faster one. `--quantization ""` with `GPU = "A100-80GB"` goes back to
  bf16. Quantization moves verdicts at the margin — re-run Tier 2 across a change.
- **`make eval-prewarm`** downloads weights on a CPU-only container, so the first
  run of a new judge never holds a GPU idle behind Hugging Face.
- **One warm container per judge.** The model loads in `@modal.enter()`, not per
  call, and lives for `scaledown_window` (5 min — long enough that back-to-back
  runs are free, short enough that an abandoned session is not billed for an idle
  GPU). `MODAL_JUDGE_GPU_SNAPSHOT=1` additionally restores a loaded engine
  instead of rebuilding it, where the account has that beta enabled.
- **Already-judged work is skipped.** Re-running `make eval-judge` sends nothing
  and re-prints the same scorecard from SQLite; `FORCE=1` re-judges, which is only
  worth it when the prompts or the judge changed.
- **The record comes before the question** in every prompt, so the four criteria
  for one decision share one cached prefix instead of prefilling that record four
  times. `tests/test_rubric.py` pins that ordering.
- **Prompts go in chunks**, so verdicts persist as they land and a retry re-runs
  one chunk rather than the batch. `run_pairwise` takes `LIMIT=` / `SINCE=`,
  because pairs × 4 criteria × 2 orders over a full shared history is unbounded.

### Two input sets, never averaged together

- **Synthetic fixtures** (`eval/fixtures/synthetic/*.yaml`) are hand-built and
  adversarial, with a known-correct answer. They give hard pass/fail regression
  gates. Their tickers are always synthetic (`TESTA`…`TESTD`) and they never write
  to the `decisions` table.
- **Replayed real days** (the `decisions` table) are real yfinance indicators and
  news. They have **no** ground truth, so they are only used for Tier 1 pairwise
  model-vs-model comparison.

Averaging the two produces a number that answers neither question: a synthetic
score measures "does this model handle this specific adversarial situation", a
replayed score measures "which model is better on real data". Keep them apart.

### Reading the output honestly

Three numbers decide whether a result is a result at all:

- **fallback rate per sentinel field.** The forgiving parsers return
  `neutral / 5.0 / low` on unparseable output, so a broken model scores as
  *mediocre* unless you look here first. `SentimentSignal.fallbacks` and friends
  record which fields defaulted.
- **`judge_parse_failure_rate`** (Tier 1). A high rate means the run is
  **invalid**, not low-scoring. Nothing is ever imputed — an unparseable judge
  response is `None`, and `derived_score` is `yes / answered`.
- **`order_flip_rate`** (Tier 1 pairwise). Every pair is judged in both display
  orders; the fraction whose winner changes is the judge's own noise floor. A
  win-rate margin smaller than it is not a result, and `eval.report` refuses to
  name a winner inside it.

Tier 1 also warns when the judge and the model under test share a family
(`rubric.same_family`) — Qwen2.5-32B scoring Qwen3-30B-A3B output is
self-preference risk. `--second-judge` runs a different family and reports
inter-judge agreement.

Tier 2 is what makes Tiers 0–1 falsifiable: without it, a judge that answers
"yes" to everything and a genuinely good cascade produce the same scorecard.
`make eval-calibrate CMD=score` reports **Cohen's kappa** per criterion between
your labels and the judge's — kappa rather than raw agreement, because raw
agreement is inflated by the base rate (on a criterion where 90% of records are
truly "yes", a yes-to-everything judge scores 90% and has learned nothing).

Kappa below 0.4 on a criterion means **the rubric wording is broken, not the
cascade**: fix `CRITERIA[criterion]` in `eval/rubric.py` and re-judge before
drawing any conclusion about a model. Re-run `score` whenever the judge model or
any criterion prompt changes — a rubric edit invalidates the previous kappa.

The committed comparison floor is `eval/results/baseline.json`; see
[`eval/results/README.md`](eval/results/README.md) for how to produce it and
[`tests/fixtures/responses/README.md`](tests/fixtures/responses/README.md) for
capturing real model output as CI regression fixtures. Both must come from a real
device run — invented numbers would make every later comparison look rigorous
while measuring a fiction.

## Deploying to the Jetson

Pull-based: GitHub CI proves a commit is green, then the device pulls and runs it
on a systemd timer. See [`deploy/README.md`](deploy/README.md) for one-time setup
and `make deploy ENV=staging`.

## Hardware target

NVIDIA Jetson Orin Nano Super, 8GB unified memory, JetPack 7.2 / L4T r39.2, CUDA
13.2, GPU compute capability sm_87. Two GGUF Q4_K_M model tiers via `llama.cpp`:
~1B quick tier, ~4B deep tier. Currently running at 15W (MAXN_SUPER not yet
enabled on this unit).
