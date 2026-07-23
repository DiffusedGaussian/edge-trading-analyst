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
  config.py        watchlist/config loading
  data_source.py    yfinance wrapper (OHLCV + fundamentals snapshot)
  indicators.py     SMA / EMA / MACD / RSI — pure functions, no LLM
  gate.py           materiality gate: crossing + level rules, OR-combined
  store.py          SQLite persistence (bars, fundamentals)
  pipeline.py       wires data -> indicators -> gate -> persist (Phases 1-2)
  llm_client.py     thin HTTP client for llama-server's OpenAI-compatible API
  llm_parsing.py    shared forgiving sentinel-field parser
  analyst.py        Phase 3: quick-tier news/sentiment analyst
  debate.py         Phase 4: bull/bear debate + trader synthesis
tests/              indicator/gate smoke tests (design-decision regressions)
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

uv run python -m edge_analyst.analyst AAPL "Some news headline." http://localhost:8080
uv run python -m edge_analyst.debate  AAPL "Some news headline." http://localhost:8080
```

`analyst.py`'s `__main__` prints the prompt, raw model output, and parsed
`SentimentSignal`. `debate.py`'s `__main__` runs the full analyst → debate → trader
chain and prints the final `DebateState` and `TraderDecision`.

## Deploying to the Jetson

Pull-based: GitHub CI proves a commit is green, then the device pulls and runs it
on a systemd timer. See [`deploy/README.md`](deploy/README.md) for one-time setup
and `make deploy ENV=staging`.

## Hardware target

NVIDIA Jetson Orin Nano Super, 8GB unified memory, JetPack 7.2 / L4T r39.2, CUDA
13.2, GPU compute capability sm_87. Two GGUF Q4_K_M model tiers via `llama.cpp`:
~1B quick tier, ~4B deep tier. Currently running at 15W (MAXN_SUPER not yet
enabled on this unit).
