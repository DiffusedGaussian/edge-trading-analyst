"""Load watchlist/config from YAML. Pure plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    tickers: list[str]
    lookback_days: int


def load_config(path: str | Path = "config/watchlist.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(tickers=raw["tickers"], lookback_days=raw["lookback_days"])
