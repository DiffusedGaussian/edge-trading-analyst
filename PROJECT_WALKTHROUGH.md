# Edge Trading Analyst — project walkthrough

---

## 1. The one-sentence pitch

A multi-agent LLM trading analyst that runs entirely on a single 8GB Jetson Orin
Nano, producing paper-trade decisions on an intraday cadence — where the central
engineering problem isn't the agent graph, it's deciding when the graph is allowed
to run at all, because the hardware can only afford a handful of LLM calls per day
per ticker.

If they ask "why is this hard," the honest answer is: it isn't hard because of the
LLM reasoning — it's hard because of the 8GB/15W constraint forcing every design
decision to ask "does this need a model call, and can I afford it."

## 2. The core design philosophy — lead with this

Three buckets, and the placement of every piece of logic into one of them is the
single most important decision in the codebase:

1. **Pure computation, no LLM, ever.** Indicators (SMA/EMA/MACD/RSI), the
   materiality gate, risk limits. Runs for the whole watchlist, every cycle, for
   free.
2. **Gating logic, no LLM.** Deterministic rules deciding whether a given cycle
   even deserves an LLM call. This is the actual cost lever in the system — more
   important than anything about the models themselves.
3. **Language reasoning, LLM, rationed.** News synthesis, the bull/bear debate,
   the trader's final call. The only place tokens get spent, and only when bucket
   2 says it's warranted.

The framing to give a senior dev: this is "determinism first, LLM only where
language reasoning is genuinely irreducible" — not "wrap an LLM around some
indicators." The gate existing at all is what makes an 8GB/15W box viable for this
task; without it, the system would either run the model constantly and starve, or
skip reasoning that actually matters.

## 3. Walking the pipeline, phase by phase

The repo's own phase table is the map. Talk through it in this order — it's also
the order things were built:

**Phase 1 — deterministic core (`data_source.py`, `indicators.py`, `store.py`).**
yfinance wrapper → SMA/EMA/MACD/RSI as pure functions → SQLite. No LLM anywhere in
this layer. Worth calling out: `data_source.py` is explicitly a swappable
plumbing layer — the docstring says it's there to be replaced by Alpaca later
(Phase 5, real paper-broker fills) without touching anything downstream, because
callers only depend on the DataFrame shape it returns, not on yfinance itself.

**Phase 2 — the materiality gate (`gate.py`).** Deterministic crossing/level rules
(MACD histogram crossing zero, RSI crossing 30/70, a >3% price move), OR-combined —
any one firing is reason enough to spend an LLM call. Two decisions worth
explaining if asked:
- *Crossing* (`crossed_above`/`crossed_below`) vs *level* (`price_move_exceeds`) are
  deliberately different shapes. RSI/MACD have persistent state, so a crossing
  check fires once per event, not once per day the level holds — a level check on
  those would refire every day RSI sits under 30. Price change has no such state,
  so a level check is correct there.
- OR, not AND: requiring every indicator to agree would miss fast-moving events
  the slower, smoothed indicators haven't caught up to yet. The cost of a false
  positive (one wasted LLM call) is much cheaper than the cost of a false negative
  (missing a real move).

**Phase 3 — news/sentiment analyst, quick tier (`news_analyst.py`).** Live-validated
on the Jetson already. The one design decision to lead with: the model is *given*
the deterministic RSI/MACD labels (`interpret_rsi`, `interpret_macd_hist` in
`indicators.py`) and explicitly told to treat them as ground truth, not recompute
them. Small models are unreliable at the numeric judgment (a 1B model will
mislabel a neutral 55 RSI as "bearish") but fine at the language synthesis around
a label it's handed. This is the load-bearing idea behind every prompt in the
system — indicators.py's `format_market_context` is the single shared function all
three agent prompts (analyst, debate, trader) build from, specifically so this
convention can't drift apart between them.

**Phase 4 — bull/bear debate + trader, deep tier (`debate.py`).** Built, not yet
live-validated (flag this explicitly — it's real status, not embarrassment). Two
decisions:
- `DebateState` is overwritten each round, not appended to — avoids the
  TradingAgents reference pattern's quadratic context-resend, while still letting
  each side rebut the other's *current* strongest point (`opposing_key_point`
  threaded in).
- `should_continue_debate` only escalates to a second round on a genuine
  buy-vs-sell standoff; if either side has already moved to Hold, that's
  convergence, not disagreement worth paying for another round.

**Phase 5 — paper-trade state + reflection.** Not started. Be upfront about this.

**Phase 6 — deployment (`deploy/`).** Built and working: pull-based, not push-based
— GitHub CI proves a commit green, then the Jetson pulls and runs it on a systemd
timer over Tailscale. Nothing is ever pushed *into* the device. `deploy.sh` runs a
smoke test before touching the timer, so a bad commit can't replace a working
deployment. Templated per-environment (`edge-analyst@.service`/`.timer`), so
staging and a future production are the same unit file with a different env file
— staging tracks `main` continuously, production would only ever be promoted from
a tag already watched running on staging.

**Phase 7 — measure & tune.** Not started — this is where the eval harness work
(section 6) lives.

## 4. The orchestrator (`pipeline.py`) — the one place it all meets

`run_cycle` is deliberately the *only* place these phases are wired together:
snapshot → persist → gate check → (if material) fetch news → analyst → debate →
trader → persist decision. Worth pointing at directly: the gate check is a real
early return, not a flag — a quiet cycle exits before any LLM call, at the line
`if not snapshot.gate_result.material and not force: return CycleResult(...)`.
`base_url: str | None = None` being the *only* parameter that determines whether
the cascade runs at all (batch runs pass none, live runs pass a real
`llama-server` endpoint) is a small detail that turned out to matter a lot later —
see section 6, it's the entire reason the eval harness's model-comparison feature
was cheap to bolt on.

## 5. Deliberate non-decisions — things a senior dev will ask about, so name them first

**No agent framework.** No LangChain/LangGraph. `llm_client.py` is a direct HTTP
client against `llama-server`'s OpenAI-compatible endpoint. The reasoning stated
in the code: this keeps prompt shape and turn count fully under our control. On an
8GB box paying per-token, an abstraction layer that adds its own scaffolding
tokens or hides the actual call count is a real cost, not a convenience — the
project explicitly trades framework productivity for legibility of the exact
prompt/turn budget.

**Sentinel-format parsing, not JSON.** Every LLM output is `LABEL: value` lines,
parsed by one shared, forgiving regex scanner (`llm_parsing.py::extract_field`),
not structured JSON output. This is a borrowed pattern (from TauricResearch's
TradingAgents, credited as a design reference, not a dependency) chosen because
small quantized models are unreliable JSON emitters but reliable at "put this word
after this label" — and the parser is deliberately forgiving: tolerant of
leading whitespace, case, and prose/markdown wrapped around the sentinel line,
with a hard safe default per field on any miss rather than raising. That
per-field-independent fallback design is important: one bad field (a garbled
`SCORE:`) doesn't take down the whole parsed object.

**No structured schema/tier config file.** Model identity currently lives purely
in the `llama-server` shell invocation (which GGUF, which port) — there's no
config file mapping "quick tier" / "deep tier" to a model name. This was
identified as a real gap, not an oversight, when investigating whether new
models could be swapped in for comparison (section 6) — worth being upfront that
it's a known piece of debt, deferred deliberately until there's an actual second
model worth promoting to production, rather than speculatively built now.

**TradingAgents as reference, not dependency.** Used only for two ideas — role
decomposition (bull/bear/trader) and the sentinel-text parsing convention — never
imported or run as code.

## 6. Where the project is *right now* — two active threads

Frame these as "in progress, here's the reasoning so far," not finished work.

### 6a. Testing MoE models as an alternative quick/deep tier

Started from an unrelated read of Colibri (a runtime that streams MoE expert
weights from NVMe with predictive prefetch to run huge models in small RAM). The
decision made explicitly: Colibri itself doesn't port to this project (x86-only,
targets far larger MoE models than this project uses) — but the *underlying idea*
(small resident footprint + NVMe for the rest) is worth testing on real MoE models
now that a 1TB NVMe is mounted on the Jetson.

Two bracketing experiments, chosen specifically so neither requires touching
`edge_analyst` source at all — the entire test surface is `run_cycle`'s
`base_url` parameter pointed at a different `llama-server` port:

- **Path A** — a small MoE model (OLMoE-1B-7B, ~4.2GB Q4) that fits **entirely
  resident** in the 8GB budget. Tests whether MoE's capacity/active-compute
  decoupling (7B "knowledge," ~1B active/token) beats the current dense
  gemma-3-1b quick tier on quality at the same speed class.
- **Path B** — a MoE model too large to fit resident (Qwen3-30B-A3B, ~18.6GB
  Q4), backed by NVMe purely via llama.cpp's own default mmap lazy-loading —
  deliberately the *cheapest possible test* of "does expert sparsity plus
  disk-backing help at all" before anyone considers writing a custom
  prefetch/pinning engine resembling Colibri.

Both are genuinely open experiments, not foregone conclusions — under review, the
plan surfaced real problems worth being upfront about if asked for a status: Qwen3
ships with a thinking mode on by default that would blow through the sentinel
parser; the expert-sparsity premise is weaker than it first looks once you
account for prefill touching most experts anyway; `-ngl 0` was the wrong
mitigation for the GPU-materializes-tensors problem (Orin's GPU shares the same
DRAM, so the right fix is offloading non-expert tensors while keeping expert
tensors mmap'd, not disabling the GPU outright). Good instinct for a senior dev
review: this is exactly the kind of thing worth catching before burning Jetson
time on a run that was going to fail for a boring reason.

### 6b. Rebuilding the eval harness

The existing harness (`eval/rubric.py` + `eval/modal_app.py`) is LLM-as-judge
only — a self-hosted Qwen2.5-32B on Modal scores a persisted decision against a
sentinel-format rubric (bull/bear distinctness, indicator consistency, news
fidelity, trader consistency, 0–10 score). It reuses the same sentinel/forgiving-
parser convention as the production code specifically so the judge's parsing
logic can't drift out of sync with the cascade's own.

Reviewing it against the actual codebase surfaced three structural problems,
worth stating plainly since they're the reason a rebuild is underway rather than
just running the existing harness on the new models:

1. **No ground truth.** Every metric is one model's opinion of another model's
   output. A `no` verdict can't be attributed to the cascade being wrong or the
   judge misreading it.
2. **Fallbacks are invisible, then averaged in.** The forgiving parsers correctly
   return safe defaults on unparseable output (`neutral`/`5.0`/`low`) — right
   behavior at runtime, but it means a model that never produces valid output
   scores as "mediocre" in eval, not "broken." The judge's own parser does the
   same thing to itself, imputing `5.0` on its own parse misses.
3. **No attribution.** The `decisions` table has no column recording which model
   produced a row, and the judge fetches "most recent N" — so there's currently no
   way to cleanly compare two models' outputs at all, which is exactly what the
   OLMoE/Qwen3 experiments need.

The rebuild plan (`EVAL_HARNESS_PLAN.md`, already in the repo) is staged in three
tiers: **Tier 0**, deterministic checks with no LLM at all (does the output
contradict the deterministic RSI/MACD label it was handed, are numbers in the
output traceable to the input, is the parser hijackable by a sentinel-formatted
string embedded in a news headline); **Tier 1**, the LLM judge restructured into
one-criterion-per-call plus pairwise A/B comparison instead of bundled absolute
scoring; **Tier 2**, a small hand-labeled calibration set to measure whether the
judge itself agrees with a human, via Cohen's kappa. Two input sets are kept
deliberately separate throughout and never averaged together: synthetic,
hand-built adversarial fixtures with known-correct answers (used for hard
pass/fail regression gates), and replayed real market days (used only for
model-vs-model pairwise comparison, since there's no ground truth for those).

If a senior dev asks "why not just use the harness you already had" — the honest
answer is that it can currently tell you a model scored 6.5/10, but not whether
that's because the model reasoned poorly, because it couldn't produce parseable
output at all, or because the judge itself got confused. None of those are the
same problem, and only one of them is about the model you're testing.

## 7. Things worth having answers ready for

- **Why paper trading, not backtesting against historical data first?** Not
  addressed yet in the project as reviewed — worth having your own answer ready
  if it's a real gap versus a deliberate phase-ordering choice.
- **What happens on a `llama-server` crash or timeout mid-cascade?** `llm_client.py`
  currently does a bare `requests.post(...).raise_for_status()` with no retry —
  worth knowing whether that's accepted (a paper-trading research loop, missing a
  cycle is cheap) or a real gap before being asked.
- **Reproducibility.** `chat_completion` hardcodes `temperature=0.2` with no seed
  today — the eval rebuild plan addresses this (Step 3), but it's a fair question
  about the *existing*, already-validated Phase 3 results: are they reproducible
  as recorded, or was that one run at whatever temperature happened to be
  hardcoded at the time?
- **Single point of hardware failure.** Everything — inference, storage, gate,
  deploy target — is one Jetson. Worth having a one-line answer on whether that's
  an accepted constraint of the research phase or a known risk being tracked for
  later.

## 8. If you only have five minutes

Determinism does the cheap, frequent work (indicators, gating); the LLM cascade
is rationed by a gate and only ever reasons about numbers it's handed, never
asked to compute; the whole system is validated in stages (Phase 3 live-proven,
Phase 4 built-but-unvalidated, stated as such); and the two things in flight right
now — the MoE tier experiments and the eval harness rebuild — exist because the
project just hit the point where "does a different model actually help" became an
answerable, measurable question instead of a guess, and the tooling needed to
catch up to that before drawing conclusions.
