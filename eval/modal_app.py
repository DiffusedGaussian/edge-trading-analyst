"""Modal deployment: self-hosted Qwen2.5-32B-Instruct judging persisted
decisions from data/edge_analyst.db against eval/rubric.py's rubric.

No data Volume needed — unlike a training job, the judging inputs are small
text records, not large binaries. The local entrypoint reads
decisions/debate_turns from SQLite directly and ships them to the remote GPU
function as plain dicts; judge_batch has no filesystem/DB dependency of its
own beyond the model cache below. vLLM does one batched generate() call
across all records in a single warm container rather than one serverless
invocation per record.
"""

import modal

app = modal.App("edge-analyst-judge")

JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct"

# Qwen2.5-32B is ~65GB — without caching it on a Volume, every cold container
# re-downloads the full snapshot from Hugging Face before it can judge a
# single record. This Volume persists the download across runs/containers, so
# only the very first run pays that cost.
MODEL_CACHE_PATH = "/model-cache"
model_cache = modal.Volume.from_name(
    "edge-analyst-judge-model-cache", create_if_missing=True
)

# vLLM resolves its own compatible torch version — pinning torch separately
# here (unlike gaussian_splatting's CUDA-kernel-specific pin) risks a mismatch
# for no benefit, since nothing else in this image needs a specific torch build.
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
def judge_batch(records: list[dict]) -> list[dict]:
    """One warm vLLM load, one batched generate() over every record."""
    from vllm import LLM, SamplingParams

    from eval.rubric import Judgment, build_judge_prompt, parse_judgment

    llm = LLM(model=JUDGE_MODEL, download_dir=MODEL_CACHE_PATH)
    # Persist whatever huggingface_hub just wrote under MODEL_CACHE_PATH so a
    # cold container next time reuses it instead of re-downloading ~65GB.
    model_cache.commit()

    prompts = [build_judge_prompt(r) for r in records]
    outputs = llm.chat(prompts, SamplingParams(temperature=0.0, max_tokens=300))

    judgments: list[Judgment] = [parse_judgment(o.outputs[0].text) for o in outputs]
    return [j.__dict__ for j in judgments]


@app.local_entrypoint()
def run_judge(db_path: str = "data/edge_analyst.db", limit: int = 20):
    """Judge the `limit` most recent persisted decisions and write the
    results back into the same SQLite DB's `judgments` table."""
    import datetime as dt

    from edge_analyst import store

    conn = store.get_connection(db_path)
    records = store.fetch_decisions_for_judging(conn, limit)
    if not records:
        print("nothing to judge")
        return

    print(f"judging {len(records)} decisions with {JUDGE_MODEL}...")
    results = judge_batch.remote(records)

    judged_at = dt.datetime.now().isoformat(timespec="seconds")
    for record, result in zip(records, results, strict=True):
        store.save_judgment(
            conn, record["ticker"], record["as_of"], JUDGE_MODEL, result, judged_at
        )

    print(
        f"{'ticker':<8}{'as_of':<22}{'distinct':<10}{'indicator':<11}"
        f"{'news':<7}{'trader':<9}{'score':<7}notes"
    )
    for record, result in zip(records, results, strict=True):
        print(
            f"{record['ticker']:<8}{record['as_of']:<22}"
            f"{result['bull_bear_distinct']:<10}{result['indicator_consistent']:<11}"
            f"{result['news_fidelity']:<7}{result['trader_consistent']:<9}"
            f"{result['overall_score']:<7}{result['notes']}"
        )
