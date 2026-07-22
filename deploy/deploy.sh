#!/usr/bin/env bash
#
# Pull-based deploy, run ON the Jetson. Neither GitHub nor a CI runner can reach
# the box (headless, behind Tailscale), so the box pulls the already-tested commit
# rather than being pushed to.
#
#   ./deploy/deploy.sh [environment] [git-ref]
#
#   environment  staging (default) | production
#   git-ref      what to deploy (default: origin/main for staging; pass a tag like
#                v0.2.0 to pin production to a released commit)
#
# Assumes the systemd unit + timer are already installed (one-time; see README.md).
set -euo pipefail

ENVIRONMENT="${1:-staging}"
GIT_REF="${2:-origin/main}"
APP_DIR="${APP_DIR:-/opt/edge-trading-analyst}"
UNIT="edge-analyst@${ENVIRONMENT}"

echo ">> Deploying '${ENVIRONMENT}' from '${GIT_REF}' in ${APP_DIR}"
cd "$APP_DIR"

# 1. Fetch and check out exactly the requested commit. --ff-only refuses to deploy
#    if local history diverged (someone hand-edited on the box) instead of silently
#    merging — a deploy should be a clean fast-forward to a known-good ref.
git fetch --tags --prune origin
git checkout --quiet --detach "$GIT_REF"
echo ">> Now at $(git rev-parse --short HEAD)"

# 2. Install runtime deps exactly as locked. No dev tools on the device.
uv sync --frozen --no-dev

# 3. Smoke test: run one real cycle through the actual unit. `start --wait` blocks
#    until the one-shot exits and returns its exit code, so a crash fails the deploy
#    loudly instead of leaving a broken timer to fire silently later.
echo ">> Smoke test: one cycle via ${UNIT}.service"
if ! sudo systemctl start --wait "${UNIT}.service"; then
    echo "!! Smoke test FAILED — timer left untouched, previous deploy still active." >&2
    sudo systemctl status "${UNIT}.service" --no-pager --lines=20 >&2 || true
    exit 1
fi

# 4. Green: (re)enable the cadence timer. Idempotent — safe to run every deploy.
sudo systemctl enable --now "${UNIT}.timer"
echo ">> Deployed. Next runs scheduled by ${UNIT}.timer:"
systemctl list-timers "${UNIT}.timer" --no-pager || true
