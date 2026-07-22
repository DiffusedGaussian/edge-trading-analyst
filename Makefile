# Single source of truth for the repeatable commands. `make check` is exactly what
# CI runs, so a green local check means a green PR.
.PHONY: install fmt lint test check run deploy

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
