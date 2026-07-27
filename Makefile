# Single source of truth for the repeatable commands. `make check` is exactly what
# CI runs, so a green local check means a green PR.
#
# The eval-* targets below are deliberately NOT wired into CI: CI has no GPU and no
# llama-server, so they would fail there for reasons unrelated to the code. Only
# the model-independent parts of the harness (checks, fixtures, aggregation, the
# recorded-response replay) run in `make check`.
.PHONY: install fmt lint test check run deploy \
        eval-fixtures eval-report eval-compare eval-judge eval-pairwise eval-calibrate

install:            ## sync all deps incl. dev tools (ruff, pytest)
	uv sync --frozen --dev

fmt:                ## auto-format
	uv run ruff format .

lint:               ## format-check + lint (no writes) — mirrors CI
	uv run ruff format --check .
	uv run ruff check .

test:               ## run the test suite
	uv run pytest

check: lint test    ## the full CI gate, locally

run:                ## run one analysis cycle locally
	uv run python -m edge_analyst.pipeline

deploy:             ## deploy on the Jetson: make deploy ENV=staging
	./deploy/deploy.sh $(or $(ENV),staging)

# --- evaluation ------------------------------------------------------------
# Three tiers, documented in the README's "Evaluating a model" section. Tier 0 is
# free and deterministic; Tier 1 costs GPU-minutes; Tier 2 costs your attention.

eval-fixtures:      ## Tier 0 on-device: make eval-fixtures BASE_URL=... MODEL=...
	uv run python -m eval.run_fixtures --base-url $(or $(BASE_URL),http://localhost:8080) \
		--model-name $(MODEL) --k $(or $(K),5)

eval-report:        ## summarise one run: make eval-report RUN=<run_id> (or RUN=latest)
	uv run python -m eval.report --run $(or $(RUN),latest)

eval-compare:       ## side-by-side: make eval-compare A=<run_id> B=<run_id>
	uv run python -m eval.report --compare $(A) $(B)

eval-judge:         ## Tier 1 on Modal (needs `uv sync --group eval` + Modal auth)
	uv run modal run eval/modal_app.py::run_judge

eval-pairwise:      ## Tier 1 bake-off: make eval-pairwise A=<model> B=<model>
	uv run modal run eval/modal_app.py::run_pairwise --model-a $(A) --model-b $(B)

eval-calibrate:     ## Tier 2: make eval-calibrate CMD="label --n 40" | CMD=score
	uv run python -m eval.calibrate $(or $(CMD),score)
