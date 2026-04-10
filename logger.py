"""
Snapshot logger for backtesting.

Appends one JSON line per refresh cycle to data/snapshots.jsonl.
Each line is a self-contained snapshot of all contracts + vol gaps at a point in time.

Schema per line:
{
  "ts":   "2026-04-10T10:19:53+00:00",   # UTC ISO timestamp
  "spot": 71822.0,                         # BTC/USD
  "contracts": [
    {
      "market_id":     "...",
      "event_id":      "...",
      "contract_type": "european_digital" | "one_touch",
      "question":      "...",
      "strike":        70000.0,
      "direction":     "above",
      "expiry":        "2026-04-10T16:00:00+00:00",
      "T_years":       0.000274,
      "yes_price":     0.96,
      "liquidity":     45118.0,
      "iv_poly":       57.5,        # % or null
      "iv_deribit":    42.6,        # % or null
      "vol_gap":       14.9,        # pp or null
    },
    ...
  ]
}
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

SNAPSHOT_DIR  = Path("data")
SNAPSHOT_FILE = SNAPSHOT_DIR / "snapshots.jsonl"


def _ensure_dir() -> None:
    SNAPSHOT_DIR.mkdir(exist_ok=True)


def log_snapshot(
    ts:        datetime,
    spot:      float,
    contracts: list[dict],          # already-processed dicts from _contracts_to_api
    euro_groups: list[dict],
    ot_groups:   list[dict],
) -> None:
    """
    Flatten all contract rows from euro_groups + ot_groups into a single snapshot
    line and append to SNAPSHOT_FILE.
    """
    _ensure_dir()

    rows = []
    for ct_label, groups in [("european_digital", euro_groups),
                              ("one_touch",        ot_groups)]:
        for g in groups:
            expiry  = g["expiry"]
            T_years = g["T_years"]
            for m in g["markets"]:
                rows.append({
                    "market_id":     m["market_id"],
                    "contract_type": ct_label,
                    "question":      m["question"],
                    "strike":        m["strike"],
                    "direction":     m["direction"],
                    "expiry":        expiry,
                    "T_years":       T_years,
                    "yes_price":     m["yes_price"],
                    "liquidity":     m["liquidity"],
                    "iv_poly":       m["iv_poly"],
                    "iv_deribit":    m["iv_deribit"],
                    "vol_gap":       m["vol_gap"],
                })

    record = {
        "ts":        ts.isoformat(),
        "spot":      spot,
        "contracts": rows,
    }

    try:
        with open(SNAPSHOT_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        log.error("Snapshot write failed: %s", e)


def load_snapshots(path: Optional[Path] = None) -> list[dict]:
    """Load all snapshots from disk. Returns list sorted by ts ascending."""
    p = path or SNAPSHOT_FILE
    if not p.exists():
        return []
    records = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    records.sort(key=lambda r: r["ts"])
    return records
