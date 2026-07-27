"""Modal deployment: a self-hosted judge scoring persisted decisions from
data/edge_analyst.db against eval/rubric.py's criteria.

No data Volume needed — unlike a training job, the judging inputs are small text
records, not large binaries. The local entrypoints read decisions/debate_turns
from SQLite directly and ship them to the remote GPU function as plain dicts;
the Judge class has no filesystem or DB dependency of its own beyond the model
cache below. vLLM does one batched chat() per chunk in a single warm container
rather than one serverless invocation per prompt.

Two modes:

    modal run eval/modal_app.py                        # per-criterion, one model
    modal run eval/modal_app.py::run_pairwise \\
        --model-a gemma-3-1b-it --model-b olmoe-1b-7b  # A-vs-B bake-off

Per-criterion judging issues four prompts per record instead of one. That is
nearly free here — they all land in the same batched chat(), and rubric.py orders
the record before the question so those four share one cached prefix — and
independent verdicts are worth far more than one four-question answer where the
reply to question 4 is contaminated by what the model said about question 1.

Where the money goes
--------------------
A judge run is ~80 short prompts at limit=20: tens of seconds of generation
behind a multi-minute model load. Loading, not judging, is the bill. Four things
here exist only to stop paying it repeatedly:

1. `Judge` is a class, not a function, so the load happens in @modal.enter()
   once per container instead of once per .remote() call. Two judges no longer
   mean two loads inside one run, and consecutive `make eval-judge` invocations
   within `scaledown_window` mean none at all.
2. `prewarm` downloads weights on a CPU-only container, so the first run of a
   new judge model does not hold a GPU idle while tens of GB come off Hugging
   Face.
3. The entrypoints skip (decision, criterion) pairs already in the DB, so a
   re-run costs nothing rather than re-deriving identical rows.
4. Optional GPU memory snapshotting (see GPU_SNAPSHOT) restores a loaded engine
   instead of rebuilding it.
"""

import os

import modal

app = modal.App("edge-analyst-judge")

DEFAULT_JUDGE = "Qwen/Qwen2.5-32B-Instruct"
# A second judge from a different family, for measuring inter-judge agreement.
# The default judge is Qwen; scoring Qwen3-30B-A3B output with it risks
# same-family self-preference, which stays invisible unless another family checks.
DEFAULT_SECOND_JUDGE = "mistralai/Mistral-Small-24B-Instruct-2501"

# Weight quantization, applied by vLLM at load time to the bf16 checkpoints named
# above -- the reason a 32B judge runs on one 48GB card at all.
#
# bf16 was never going to fit: Qwen2.5-32B is ~65GB of weights against an L40S's
# ~44GB usable, and vLLM died on exactly that --
# "torch.OutOfMemoryError: CUDA out of memory... total capacity of 44.39 GiB".
# FP8 halves the weights to ~33GB, leaving ~11GB for KV cache, and Ada (sm89)
# runs FP8 natively, so it is faster than bf16 rather than a compromise for
# space. Chosen over an A100-80GB, which also fits bf16, because an L40S is both
# cheaper per hour and quicker through the batch; and over int4 AWQ, cheaper
# still, because W8A8 stays far closer to the bf16 verdicts already recorded.
#
# It is not free of consequence: quantization changes the judge's numerics, so
# verdicts shift at the margin. Treat a change here like a change of judge --
# re-run eval-calibrate and check Cohen's kappa before comparing scores across
# it. To go back to bf16: --quantization "" with GPU = "A100-80GB". Pass
# --quantization "" for an already-quantized repo too (e.g. ...-Instruct-AWQ):
# vLLM reads the method from the checkpoint's own config, and naming a conflicting
# one here is an error rather than an override.
DEFAULT_QUANTIZATION = "fp8"

GPU = "L40S"

# Prompts are one record (or two, pairwise) plus a criterion: ~1-2k tokens, ~4k
# pairwise. Left unset, vLLM sizes and validates KV cache against Qwen2.5's full
# 32k context and refuses to start when it cannot fit that. 8192 covers the
# longest pairwise prompt with room to spare and buys back concurrency within the
# batch, which is what actually sets how long a run takes.
MAX_MODEL_LEN = 8192

# Prompts per remote call. Chunking buys two things one giant call cannot:
# verdicts are persisted as each chunk lands, so a timeout in chunk 5 keeps
# chunks 1-4, and `retries` re-runs one chunk instead of the whole batch. Every
# chunk of a run goes to the same warm container, so this costs no extra loads.
CHUNK_SIZE = 200

# A 32B model is tens of GB -- without caching it on a Volume, every cold
# container re-downloads the full snapshot from Hugging Face before it can judge
# a single record. This Volume persists the download across runs/containers, so
# only the very first run (or `prewarm`) pays that cost. HF_HOME and
# VLLM_CACHE_ROOT both live under it, so tokenizer/config files and vLLM's
# torch.compile artifacts are cached too -- not only the weights.
MODEL_CACHE_PATH = "/model-cache"
model_cache = modal.Volume.from_name(
    "edge-analyst-judge-model-cache", create_if_missing=True
)

# Required, not optional: mistralai/Mistral-Small-24B-Instruct-2501 is a gated
# repo and 401s without a token, and anonymous downloads of a 65GB snapshot get
# rate-limited. Create it once with
#   modal secret create huggingface-secret HF_TOKEN=hf_...
hf_secret = modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])

# GPU memory snapshotting restores an already-loaded vLLM engine instead of
# rebuilding it, which is the only thing that makes a *cold* start cheap. It is a
# Modal beta that has to be enabled for the account, and it moves weight loading
# into the snapshot phase, so it ships off by default rather than breaking
# `make eval-judge` on an account without it:
#   MODAL_JUDGE_GPU_SNAPSHOT=1 make eval-judge
# The value below must be the *string* "true": experimental_options is a
# map<string,string> on the wire, and a Python bool raises a protobuf TypeError.
GPU_SNAPSHOT = os.environ.get("MODAL_JUDGE_GPU_SNAPSHOT") == "1"

# vLLM resolves its own compatible torch version — pinning torch separately here
# (unlike gaussian_splatting's CUDA-kernel-specific pin) risks a mismatch for no
# benefit, since nothing else in this image needs a specific torch build.
#
# transformers IS pinned, unlike torch, because vllm==0.8.5 only declares
# "transformers >= 4.51.1" with no upper bound (see its requirements/common.txt
# for that version) -- so an unpinned build reproducibly resolves whatever is
# newest on PyPI at build time. Reproduced live: that resolved transformers
# 5.14.1, whose PreTrainedTokenizerBase no longer has `all_special_tokens_extended`
# (confirmed present in transformers v4.57.0's tokenization_utils_base.py, absent
# in v5.0.0's) -- a property vllm==0.8.5's own get_cached_tokenizer() still calls
# unconditionally, crashing every judge run with
# "AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended"
# before a single prompt is judged. <5 stays within the major version this
# vLLM release was built against; it is not an exact pin, so routine 4.x
# bugfix releases keep resolving without a manual bump.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "vllm==0.8.5", "transformers>=4.51.1,<5", "huggingface_hub[hf_transfer]"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": f"{MODEL_CACHE_PATH}/hf",
            "VLLM_CACHE_ROOT": f"{MODEL_CACHE_PATH}/vllm",
        }
    )
    # Local code ships with every run, so edits to eval/ or src/edge_analyst
    # never trigger an image rebuild.
    .add_local_python_source("eval", "edge_analyst")
)


@app.function(
    image=image,
    volumes={MODEL_CACHE_PATH: model_cache},
    secrets=[hf_secret],
    # No GPU on purpose: this is network- and CPU-bound, and holding an idle L40S
    # for the length of a 65GB download is the most expensive way to wait.
    cpu=8,
    memory=16384,
    timeout=3600,
    retries=2,
)
def prewarm(judge_model: str = DEFAULT_JUDGE) -> None:
    """Fetch one judge's weights onto the Volume and commit, on cheap hardware.

    Separated from judging so a download that dies partway does not take a judge
    run's timeout with it: its own generous timeout, its own retries, and the one
    commit that matters. A cold judge container then finds the snapshot already
    there.
    """
    from huggingface_hub import snapshot_download

    print(f"downloading {judge_model} to {MODEL_CACHE_PATH}...")
    snapshot_download(
        judge_model,
        # `original/` holds an upstream-format duplicate of the same weights
        # (Mistral ships a consolidated copy there) and .pth files are the legacy
        # format; vLLM reads neither, so this is pure download saved.
        ignore_patterns=["original/*", "*.pth"],
    )
    model_cache.commit()
    print("done")


@app.cls(
    image=image,
    gpu=GPU,
    volumes={MODEL_CACHE_PATH: model_cache},
    secrets=[hf_secret],
    # Unset, a container sits near Modal's CPU floor, which throttles both the
    # hf_transfer download this image enables and the tokenization of a few
    # thousand prompts before the GPU sees any of them.
    cpu=8,
    memory=32768,
    timeout=1800,
    # Deliberately short. Keeping a container warm only beats reloading while the
    # reload it saves costs more than the idle GPU it burns: ~2-4 minutes of
    # loading against 5 minutes of idle L40S. That makes back-to-back runs free
    # and an abandoned session cheap. A long window — or min_containers — would
    # quietly bill for an idle GPU instead.
    scaledown_window=300,
    retries=modal.Retries(max_retries=2, initial_delay=5.0),
    enable_memory_snapshot=GPU_SNAPSHOT,
    experimental_options={"enable_gpu_snapshot": "true"} if GPU_SNAPSHOT else {},
)
class Judge:
    """One loaded judge model, reused across every chunk of a run.

    The model is a class parameter, so Modal keys containers by it: judging with
    two judges gets one warm container each, rather than one container that
    loads, unloads and loads again.
    """

    judge_model: str = modal.parameter(default=DEFAULT_JUDGE)
    quantization: str = modal.parameter(default=DEFAULT_QUANTIZATION)

    @modal.enter(snap=GPU_SNAPSHOT)
    def load(self) -> None:
        from vllm import LLM

        self.llm = LLM(
            model=self.judge_model,
            quantization=self.quantization or None,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.92,
            # The whole point of rubric.py putting the record before the
            # question: the four criteria for one record then share a prefix and
            # only the first of them prefills it. On by default for vLLM's V1
            # engine, set explicitly so a version flip cannot silently drop the
            # saving the prompt layout was designed around.
            enable_prefix_caching=True,
            # An fp8 KV cache doubles how many prompts fit in the ~11GB left
            # after fp8 weights, and concurrency is what sets wall-clock here.
            # Tied to the weight setting: with bf16 weights there is no reason to
            # accept the extra approximation.
            kv_cache_dtype="fp8" if self.quantization == "fp8" else "auto",
        )
        # Once per cold start, not once per call. Weights arrive via `prewarm`,
        # but vLLM's torch.compile artifacts are written during this load and are
        # worth persisting -- rebuilding them is a slice of every cold start.
        model_cache.commit()

    @modal.method()
    def judge(self, prompts: list[list[dict]]) -> list[str]:
        """One batched chat() over a chunk of prompts, raw text back.

        Deliberately returns *unparsed* text: parsing lives in eval/rubric.py,
        which is tested locally with no GPU. A remote function that both
        generated and interpreted would make the interpretation untestable
        without Modal.
        """
        import time

        from vllm import SamplingParams

        started = time.monotonic()
        outputs = self.llm.chat(
            prompts,
            # seed pinned alongside temperature=0: greedy decoding alone is not
            # reproducible under continuous batching, where the numerics depend
            # on which sequences happen to batch together.
            SamplingParams(temperature=0.0, max_tokens=200, seed=0),
        )
        elapsed = time.monotonic() - started
        generated = sum(len(output.outputs[0].token_ids) for output in outputs)
        print(
            f"{len(prompts)} prompts in {elapsed:.1f}s "
            f"({generated / max(elapsed, 1e-6):.0f} output tok/s)"
        )
        return [output.outputs[0].text for output in outputs]


def _chunks(jobs: list, size: int = CHUNK_SIZE):
    for start in range(0, len(jobs), size):
        yield jobs[start : start + size]


@app.local_entrypoint()
def run_judge(
    db_path: str = "data/edge_analyst.db",
    limit: int = 20,
    model: str = "",
    judge_model: str = DEFAULT_JUDGE,
    second_judge: str = "",
    quantization: str = DEFAULT_QUANTIZATION,
    force: bool = False,
):
    """Judge the `limit` most recent decisions on every criterion, writing to
    criterion_verdicts. `model` narrows to one model's decisions; `second_judge`
    adds a different-family judge and reports agreement between the two.
    `--force` re-judges what this judge has already scored, which is only worth
    its GPU time when the prompts or the judge's numerics changed."""
    import datetime as dt

    from edge_analyst import store
    from eval.rubric import (
        CRITERION_NAMES,
        build_judge_prompt,
        judge_key,
        parse_verdict,
        pending_judge_jobs,
        same_family,
    )

    conn = store.get_connection(db_path)
    records = store.fetch_decisions_for_judging(conn, limit, model=model or None)
    if not records:
        print("nothing to judge")
        return

    judges = [judge_model] + ([second_judge] if second_judge else [])
    judged_at = dt.datetime.now().isoformat(timespec="seconds")
    # Every verdict this run is *about*, whether judged now or on an earlier run.
    # Both the scorecard and inter-judge agreement are computed over this scope
    # from the DB, so a run that skipped everything still reports the same
    # numbers as the run that first judged it.
    scope = {judge_key(record, c) for record in records for c in CRITERION_NAMES}
    verdicts_by_judge: dict[str, dict[tuple, str | None]] = {}

    for judge in judges:
        jobs = pending_judge_jobs(
            records,
            CRITERION_NAMES,
            store.fetch_judged_keys(conn, judge),
            force=force,
        )
        skipped = len(scope) - len(jobs)
        if jobs:
            print(
                f"judging {len(jobs)} prompt(s) with {judge} "
                f"({len(records)} decisions x {len(CRITERION_NAMES)} criteria, "
                f"{skipped} already stored)..."
            )
        else:
            print(
                f"{judge}: all {skipped} verdict(s) already stored, nothing to "
                "send (--force to re-judge)"
            )

        remote_judge = Judge(judge_model=judge, quantization=quantization)
        for chunk in _chunks(jobs):
            texts = remote_judge.judge.remote(
                [build_judge_prompt(record, criterion) for record, criterion in chunk]
            )
            rows = []
            for (record, criterion), text in zip(chunk, texts, strict=True):
                verdict = parse_verdict(text, criterion)
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
            # Per chunk, not per run: a later chunk timing out or exhausting its
            # retries must not discard GPU time already spent.
            store.save_criterion_verdicts(conn, rows)
            print(f"  saved {len(rows)} verdict(s)")

        stored = [
            row
            for row in store.fetch_criterion_verdicts(conn, judge_model=judge)
            if (row["ticker"], row["as_of"], row["model"] or "", row["criterion"])
            in scope
        ]
        _print_verdicts(judge, stored)
        verdicts_by_judge[judge] = {
            (row["ticker"], row["as_of"], row["criterion"]): row["verdict"]
            for row in stored
        }

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
    print(f"\n{judge}")
    if not rows:
        print("  no verdicts stored")
        return
    answered = [r for r in rows if r["verdict"] is not None]
    yes = sum(1 for r in answered if r["verdict"] == "yes")
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
    quantization: str = DEFAULT_QUANTIZATION,
    limit: int = 0,
    since: str = "",
    force: bool = False,
):
    """Compare two models' decisions on the same ticker-days, with every pair
    judged in both display orders so the judge's own noise floor is measured.

    `limit` (most recent matched ticker-days) and `since` (an as_of prefix like
    2026-07) bound the batch. Unbounded, this grows as
    pairs x 4 criteria x 2 orders over the entire shared history of both models,
    and the failure mode is a timeout *after* paying for all of it."""
    import datetime as dt

    from edge_analyst import store
    from eval.report import render_pairwise
    from eval.rubric import (
        CRITERION_NAMES,
        ORDERS,
        build_pairwise_prompt,
        pairwise_key,
        parse_pairwise,
        pending_pairwise_jobs,
        resolve_pairwise_winner,
    )

    conn = store.get_connection(db_path)
    pairs = store.fetch_decisions_for_pairwise(conn, model_a, model_b)
    if since:
        pairs = [pair for pair in pairs if (pair[0]["as_of"] or "") >= since]
    # fetch_decisions_for_pairwise returns ticker-days as_of-ascending, so the
    # tail is the most recent window.
    if limit:
        pairs = pairs[-limit:]
    if not pairs:
        print(f"no matched ticker-days for {model_a} vs {model_b}")
        return

    jobs = pending_pairwise_jobs(
        pairs,
        CRITERION_NAMES,
        ORDERS,
        store.fetch_pairwise_keys(conn, model_a, model_b, judge_model),
        force=force,
    )
    scope = {
        pairwise_key(record_a, record_b, criterion, order)
        for record_a, record_b in pairs
        for order in ORDERS
        for criterion in CRITERION_NAMES
    }
    if jobs:
        print(
            f"comparing {len(pairs)} matched decision(s) x "
            f"{len(CRITERION_NAMES)} criteria x {len(ORDERS)} orders "
            f"= {len(jobs)} prompt(s) with {judge_model} "
            f"({len(scope) - len(jobs)} already stored)..."
        )
    else:
        print(
            f"{judge_model}: all {len(scope)} comparison(s) already stored, "
            "nothing to send (--force to re-compare)"
        )

    judged_at = dt.datetime.now().isoformat(timespec="seconds")
    remote_judge = Judge(judge_model=judge_model, quantization=quantization)
    for chunk in _chunks(jobs):
        texts = remote_judge.judge.remote(
            [build_pairwise_prompt(a, b, c, o) for a, b, c, o in chunk]
        )
        rows = []
        for (record_a, record_b, criterion, order), text in zip(
            chunk, texts, strict=True
        ):
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
                    # Stored resolved to the record, not the display position, so
                    # the table is readable without re-deriving the order mapping.
                    "winner": resolve_pairwise_winner(verdict),
                    "reason": verdict.reason,
                    "judged_at": judged_at,
                }
            )
        store.save_pairwise_results(conn, rows)
        print(f"  saved {len(rows)} comparison(s)")

    stored = [
        row
        for row in store.fetch_pairwise_results(conn, model_a, model_b, judge_model)
        if (
            row["ticker"],
            row["as_of_a"],
            row["as_of_b"],
            row["criterion"],
            row["order_shown"],
        )
        in scope
    ]
    conn.close()

    print(render_pairwise(stored, model_a, model_b))
