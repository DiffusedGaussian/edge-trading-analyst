"""Modal deployment: a self-hosted judge scoring persisted decisions from
data/edge_analyst.db against eval/rubric.py's criteria.

No data Volume needed — unlike a training job, the judging inputs are small text
records, not large binaries. The local entrypoints read decisions/debate_turns
from SQLite directly and ship them to the remote GPU function as plain dicts;
judge_prompts has no filesystem or DB dependency of its own beyond the model
cache below. vLLM does one batched chat() across every prompt in a single warm
container rather than one serverless invocation per prompt.

Two modes:

    modal run eval/modal_app.py                        # per-criterion, one model
    modal run eval/modal_app.py::run_pairwise \\
        --model-a gemma-3-1b-it --model-b olmoe-1b-7b  # A-vs-B bake-off

Per-criterion judging issues four prompts per record instead of one. That is
nearly free here — they all land in the same batched chat() — and independent
verdicts are worth far more than one four-question answer where the reply to
question 4 is contaminated by what the model said about question 1.
"""

import modal

app = modal.App("edge-analyst-judge")

DEFAULT_JUDGE = "Qwen/Qwen2.5-32B-Instruct"
# A second judge from a different family, for measuring inter-judge agreement.
# The default judge is Qwen; scoring Qwen3-30B-A3B output with it risks
# same-family self-preference, which stays invisible unless another family checks.
DEFAULT_SECOND_JUDGE = "mistralai/Mistral-Small-24B-Instruct-2501"

# A 32B model is ~65GB — without caching it on a Volume, every cold container
# re-downloads the full snapshot from Hugging Face before it can judge a single
# record. This Volume persists the download across runs/containers, so only the
# very first run pays that cost.
MODEL_CACHE_PATH = "/model-cache"
model_cache = modal.Volume.from_name(
    "edge-analyst-judge-model-cache", create_if_missing=True
)

# vLLM resolves its own compatible torch version — pinning torch separately here
# (unlike gaussian_splatting's CUDA-kernel-specific pin) risks a mismatch for no
# benefit, since nothing else in this image needs a specific torch build.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.8.5", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Local code ships with every run, so edits to eval/ or src/edge_analyst
    # never trigger an image rebuild.
    .add_local_python_source("eval", "edge_analyst")
)


@app.function(
    image=image,
    gpu="L40S",
    timeout=1800,
    volumes={MODEL_CACHE_PATH: model_cache},
)
def judge_prompts(prompts: list[list[dict]], judge_model: str) -> list[str]:
    """One warm vLLM load, one batched chat() over every prompt, raw text back.

    Deliberately returns *unparsed* text: parsing lives in eval/rubric.py, which
    is tested locally with no GPU. A remote function that both generated and
    interpreted would make the interpretation untestable without Modal.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(model=judge_model, download_dir=MODEL_CACHE_PATH)
    # Persist whatever huggingface_hub just wrote under MODEL_CACHE_PATH so a
    # cold container next time reuses it instead of re-downloading ~65GB.
    model_cache.commit()

    outputs = llm.chat(prompts, SamplingParams(temperature=0.0, max_tokens=200))
    return [output.outputs[0].text for output in outputs]


@app.local_entrypoint()
def run_judge(
    db_path: str = "data/edge_analyst.db",
    limit: int = 20,
    model: str = "",
    judge_model: str = DEFAULT_JUDGE,
    second_judge: str = "",
):
    """Judge the `limit` most recent decisions on every criterion, writing to
    criterion_verdicts. `model` narrows to one model's decisions; `second_judge`
    adds a different-family judge and reports agreement between the two."""
    import datetime as dt

    from edge_analyst import store
    from eval.rubric import (
        CRITERION_NAMES,
        build_judge_prompt,
        parse_verdict,
        same_family,
    )

    conn = store.get_connection(db_path)
    records = store.fetch_decisions_for_judging(conn, limit, model=model or None)
    if not records:
        print("nothing to judge")
        return

    judges = [judge_model] + ([second_judge] if second_judge else [])
    judged_at = dt.datetime.now().isoformat(timespec="seconds")
    verdicts_by_judge: dict[str, dict[tuple, str | None]] = {}

    for judge in judges:
        # (record, criterion) pairs in a fixed order, so the flat list of
        # completions can be zipped back onto what produced it.
        pairs = [(r, c) for r in records for c in CRITERION_NAMES]
        prompts = [build_judge_prompt(r, c) for r, c in pairs]
        print(
            f"judging {len(records)} decisions x {len(CRITERION_NAMES)} criteria "
            f"= {len(prompts)} prompts with {judge}..."
        )
        texts = judge_prompts.remote(prompts, judge)

        rows, verdicts = [], {}
        for (record, criterion), text in zip(pairs, texts, strict=True):
            verdict = parse_verdict(text, criterion)
            verdicts[(record["ticker"], record["as_of"], criterion)] = verdict.verdict
            rows.append(
                {
                    "ticker": record["ticker"],
                    "as_of": record["as_of"],
                    "model": record.get("model"),
                    "judge_model": judge,
                    "criterion": criterion,
                    "verdict": verdict.verdict,
                    "reason": verdict.reason,
                    "judged_at": judged_at,
                }
            )
        store.save_criterion_verdicts(conn, rows)
        verdicts_by_judge[judge] = verdicts
        _print_verdicts(judge, rows)

        risky = {
            r["model"]
            for r in records
            if r.get("model") and same_family(r["model"], judge)
        }
        for model_name in sorted(risky):
            print(
                f"  WARNING: {model_name} and judge {judge} look like the same "
                "model family — treat these verdicts as self-preference-prone"
            )

    if len(judges) == 2:
        _print_inter_judge_agreement(*(verdicts_by_judge[j] for j in judges))
    conn.close()


def _print_verdicts(judge: str, rows: list[dict]) -> None:
    answered = [r for r in rows if r["verdict"] is not None]
    yes = sum(1 for r in answered if r["verdict"] == "yes")
    print(f"\n{judge}")
    print(f"  {'ticker':<8}{'as_of':<22}{'criterion':<22}verdict  reason")
    for row in rows:
        print(
            f"  {row['ticker']:<8}{row['as_of']:<22}{row['criterion']:<22}"
            f"{str(row['verdict']):<9}{row['reason']}"
        )
    # Reported side by side, never blended: a high parse-failure rate means the
    # run is *invalid*, which is a different statement from a low score.
    parse_failures = len(rows) - len(answered)
    derived = f"{yes / len(answered):.2f}" if answered else "n/a"
    print(
        f"  derived_score {derived} ({yes}/{len(answered)} yes)   "
        f"judge_parse_failure_rate {parse_failures / len(rows):.0%}"
    )


def _print_inter_judge_agreement(first: dict, second: dict) -> None:
    both_answered = [
        key
        for key in first
        if key in second and first[key] is not None and second[key] is not None
    ]
    if not both_answered:
        print("\ninter-judge agreement: n/a (no overlapping parsed verdicts)")
        return
    agree = sum(1 for key in both_answered if first[key] == second[key])
    print(
        f"\ninter-judge raw agreement {agree / len(both_answered):.0%} "
        f"over {len(both_answered)} verdicts both judges parsed"
    )


@app.local_entrypoint()
def run_pairwise(
    model_a: str,
    model_b: str,
    db_path: str = "data/edge_analyst.db",
    judge_model: str = DEFAULT_JUDGE,
):
    """Compare two models' decisions on the same ticker-days, with every pair
    judged in both display orders so the judge's own noise floor is measured."""
    import datetime as dt

    from edge_analyst import store
    from eval.report import render_pairwise
    from eval.rubric import (
        CRITERION_NAMES,
        ORDERS,
        build_pairwise_prompt,
        parse_pairwise,
        resolve_pairwise_winner,
    )

    conn = store.get_connection(db_path)
    pairs = store.fetch_decisions_for_pairwise(conn, model_a, model_b)
    if not pairs:
        print(f"no matched ticker-days for {model_a} vs {model_b}")
        return

    jobs = [
        (record_a, record_b, criterion, order)
        for record_a, record_b in pairs
        for criterion in CRITERION_NAMES
        for order in ORDERS
    ]
    print(
        f"comparing {len(pairs)} matched decision(s) x {len(CRITERION_NAMES)} "
        f"criteria x {len(ORDERS)} orders = {len(jobs)} prompts with {judge_model}..."
    )
    texts = judge_prompts.remote(
        [build_pairwise_prompt(a, b, c, o) for a, b, c, o in jobs], judge_model
    )

    judged_at = dt.datetime.now().isoformat(timespec="seconds")
    rows = []
    for (record_a, record_b, criterion, order), text in zip(jobs, texts, strict=True):
        verdict = parse_pairwise(text, criterion, order)
        rows.append(
            {
                "ticker": record_a["ticker"],
                "as_of_a": record_a["as_of"],
                "as_of_b": record_b["as_of"],
                "model_a": model_a,
                "model_b": model_b,
                "judge_model": judge_model,
                "criterion": criterion,
                "order_shown": order,
                # Stored resolved to the record, not the display position, so the
                # table is readable without re-deriving the order mapping.
                "winner": resolve_pairwise_winner(verdict),
                "reason": verdict.reason,
                "judged_at": judged_at,
            }
        )
    store.save_pairwise_results(conn, rows)
    conn.close()

    print(render_pairwise(rows, model_a, model_b))
