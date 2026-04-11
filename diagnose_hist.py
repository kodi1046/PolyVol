"""
Diagnose CLOB price-history failures.

Tests CLOB with different fidelity values, with/without time params,
and across markets from different time periods (recent vs old).

Run: .venv/bin/python diagnose_hist.py
"""

import json
import logging
import sys
from datetime import datetime, timezone, timedelta

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

GAMMA   = "https://gamma-api.polymarket.com"
CLOB    = "https://clob.polymarket.com"
HEADERS = {"User-Agent": "PolyVol/1.0"}

session = requests.Session()
session.headers.update(HEADERS)

now = datetime.now(timezone.utc)


def get_token(slug: str) -> tuple[str | None, str | None]:
    """Return (token_id, market_question) for slug's first market."""
    r = session.get(f"{GAMMA}/events", params={"slug": slug}, timeout=15)
    if not r.ok or not r.json():
        return None, None
    event = r.json()[0]
    markets = event.get("markets", [])
    if not markets:
        eid = event.get("id")
        r2 = session.get(f"{GAMMA}/events/{eid}", timeout=15)
        markets = r2.json().get("markets", []) if r2.ok else []
    if not markets:
        return None, None
    m = markets[0]
    raw = m.get("clobTokenIds", "[]")
    tokens = json.loads(raw) if isinstance(raw, str) else raw
    token_id = str(tokens[0]) if tokens else None
    return token_id, m.get("question", "?")[:60]


def clob_try(label: str, token_id: str, **params):
    p = {"market": token_id, **params}
    r = session.get(f"{CLOB}/prices-history", params=p, timeout=15)
    hist = r.json().get("history", []) if r.ok else []
    prices = [item["p"] for item in hist]
    p_min  = min(prices) if prices else None
    p_max  = max(prices) if prices else None
    passable = [p for p in prices if 0 < p < 1]
    print(f"    [{label}] status={r.status_code}  points={len(hist)}  "
          f"in(0,1)={len(passable)}  range=[{p_min},{p_max}]")
    if hist:
        print(f"      first={hist[0]}  last={hist[-1]}")
    return hist


# ── Test across 3 time periods ────────────────────────────────────────────────

for days_ago, label in [(1, "yesterday"), (30, "1 month ago"), (80, "3 months ago")]:
    d = now - timedelta(days=days_ago)
    month = d.strftime("%B").lower()
    slug  = f"bitcoin-above-on-{month}-{d.day}"
    print(f"\n{'='*60}")
    print(f"Period: {label} ({d.date()})  slug={slug}")

    token_id, question = get_token(slug)
    if not token_id:
        print(f"  No event/token found for slug")
        continue
    print(f"  Token: {token_id[:20]}…")
    print(f"  Q:     {question}")

    start_s  = int((d - timedelta(days=7)).timestamp())
    end_s    = int((d + timedelta(days=2)).timestamp())

    print(f"  CLOB tests (startTs/endTs bracket the market's life):")
    clob_try("no params",       token_id, fidelity=60)
    clob_try("fidelity=1000",   token_id, fidelity=1000)
    clob_try("ts_sec w/ fid60", token_id, fidelity=60,   startTs=start_s,        endTs=end_s)
    clob_try("ts_ms  w/ fid60", token_id, fidelity=60,   startTs=start_s * 1000, endTs=end_s * 1000)
    clob_try("ts_ms  fid1000",  token_id, fidelity=1000, startTs=start_s * 1000, endTs=end_s * 1000)

print()
