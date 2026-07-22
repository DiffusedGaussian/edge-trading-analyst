# Deploy — Jetson Orin Nano

Pull-based deploy for a single headless Jetson reached over Tailscale. CI on GitHub
proves a commit is green (lint + tests); the box then pulls that commit and runs it
on a timer. Nothing pushes *into* the Jetson.

```
GitHub CI ── lint + test ──▶ green commit on main
                                   │
   on the Jetson:  ./deploy/deploy.sh staging
     git checkout ▸ uv sync --frozen --no-dev ▸ smoke-run ▸ enable timer
```

## One-time setup (per box)

```bash
# 1. Code lives at a fixed path the unit file expects.
sudo git clone https://github.com/DiffusedGaussian/edge-trading-analyst.git /opt/edge-trading-analyst
sudo chown -R "$USER" /opt/edge-trading-analyst

# 2. Per-environment config (data dir etc.) — never committed.
sudo mkdir -p /etc/edge-analyst
sudo tee /etc/edge-analyst/staging.env >/dev/null <<'EOF'
EDGE_ENV=staging
EDGE_DB_PATH=/var/lib/edge-analyst/staging.db
EOF
sudo mkdir -p /var/lib/edge-analyst

# 3. Install the templated unit + timer.
sudo cp /opt/edge-trading-analyst/deploy/edge-analyst@.service /etc/systemd/system/
sudo cp /opt/edge-trading-analyst/deploy/edge-analyst@.timer   /etc/systemd/system/
sudo systemctl daemon-reload

# 4. First deploy.
cd /opt/edge-trading-analyst
./deploy/deploy.sh staging
```

`deploy.sh` uses `sudo systemctl`; add a NOPASSWD sudoers rule for those calls if you
want fully non-interactive deploys.

## Deploying

```bash
./deploy/deploy.sh staging            # deploy latest main to staging
./deploy/deploy.sh staging v0.2.0     # or pin to a tag
```

A failed smoke test aborts **before** the timer is touched, so a bad commit never
replaces a working one.

## Adding production later

There is no production code yet, so only `staging` is wired. The unit is templated on
the environment name, so promoting is a matter of standing up a second instance —
same code, separate data dir — no new unit files:

```bash
sudo tee /etc/edge-analyst/production.env >/dev/null <<'EOF'
EDGE_ENV=production
EDGE_DB_PATH=/var/lib/edge-analyst/production.db
EOF
./deploy/deploy.sh production v0.2.0   # promote a released, staged-verified tag
```

Staging tracks `main` continuously; production is deployed only from tags you've
already watched run on staging — same commit promoted, never a fresh build.
