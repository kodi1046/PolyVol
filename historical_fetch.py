"""
Historical data fetcher for PolyVol backtesting.

Fetches past BTC price markets from Polymarket (Gamma + CLOB APIs) and
combines them with Deribit historical vol (DVOL index) and spot prices to
produce data/historical.jsonl — same format as data/snapshots.jsonl, so
backtest.py processes both without modification.

Usage:
    .venv/bin/python3 historical_fetch.py [--lookback 90] [--output data/historical.jsonl]

TRADEOFFS vs live-collected data (data/snapshots.jsonl)
─────────────────────────────────────────────────────────
Advantages:
  + Months of data immediately — no waiting period
  + Includes contracts that have already resolved
  + Much larger sample size for statistical analysis

Limitations:
  - No vol smile: iv_deribit is the DVOL index (Deribit's 30d ATM vol),
    NOT a strike-interpolated smile. The digital skew correction
    (digital_fair_value via call spread) is NOT applied. For ITM/OTM
    strikes this introduces a systematic error of ~2–5pp depending on
    the prevailing skew — small for near-ATM, growing for strikes >5%
    away from spot.

  - Constant risk-free rate (HIST_R = 5%): the futures-implied r is not
    reconstructed historically. Live data uses ~1.3% from the futures
    curve. This affects the one-touch formula more than the digital
    (r enters via the Laplace exponent ν). Impact on 5-day ED: ~0.05pp.
    Impact on 20-day OT: ~1–2pp.

  - Hourly granularity: CLOB price history has 1h bars vs 4s live.
    Entry/exit signals may be triggered a full hour late.

  - No same-day hourly events: the historical scan uses slug-based
    discovery for daily events only. Hourly intraday events (e.g.
    "Bitcoin above $72,400 on April 10, 8PM ET") are not included
    because scanning historical event IDs at scale is impractical.

  - Liquidity proxy: CLOB does not expose per-timestamp liquidity.
    The market's pool size from the Gamma API is used as a constant
    across the entire history — typically the final pool depth, not the
    contemporaneous one. Early in a market's life the pool is shallower.
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from fetcher import (_MONTHS, _make_session, _fetch_event_by_slug,
                     _fetch_event_by_id, _parse_iso,
                     _parse_euro_market, _parse_ot_market)
from vol_math import implied_vol_european, implied_vol_one_touch

log = logging.getLogger(__name__)

CLOB_API          = "https://clob.polymarket.com"
DERIBIT           = "https://www.deribit.com/api/v2/public"
HEADERS           = {"User-Agent": "PolyVol/1.0"}
TIMEOUT           = 15
REQUEST_DELAY     = 0.15   # seconds between CLOB calls to avoid rate-limiting

# IV parameters — must stay in sync with app.py
HIST_R    = 0.05   # constant r (see module docstring)
PRICE_MIN = 0.03
PRICE_MAX = 0.97
T_MIN     = 2.0 / 365.25 / 24   # 2 hours
MAX_IV    = 2.50


# ── Helpers ───────────────────────────────────────────────────────────────────

def _round_to_hour(ts_s: int) -> int:
    """Round a Unix timestamp (seconds) down to the nearest hour."""
    return (ts_s // 3600) * 3600


def _extract_yes_token_id(market: dict) -> Optional[str]:
    """
    Parse the YES-side CLOB token ID from a Gamma API market object.
    clobTokenIds is stored as a JSON string e.g. '["abc123", "def456"]'.
    Index 0 = YES, index 1 = NO.
    """
    raw = market.get("clobTokenIds", "[]")
    if isinstance(raw, str):
        try:
            tokens = json.loads(raw)
        except Exception:
            return None
    elif isinstance(raw, list):
        tokens = raw
    else:
        return None
    return str(tokens[0]) if tokens else None


# ── Deribit historical data ───────────────────────────────────────────────────

def _fetch_deribit_dvol(start_dt: datetime,
                         end_dt:   datetime,
                         chunk_days: int = 30) -> dict[int, float]:
    """
    Fetch hourly BTC DVOL (30d ATM vol, %) via the dedicated volatility index endpoint.
    Returns {hour_timestamp_s: dvol_value_percent}.
    """
    out: dict[int, float] = {}
    cursor = start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
        try:
            r = requests.get(
                f"{DERIBIT}/get_volatility_index_data",
                params={
                    "currency":        "BTC",
                    "start_timestamp": int(cursor.timestamp() * 1000),
                    "end_timestamp":   int(chunk_end.timestamp() * 1000),
                    "resolution":      "3600",
                },
                headers=HEADERS,
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            # Response: {"result": {"data": [[ts_ms, open, high, low, close], ...], ...}}
            data = r.json()["result"].get("data", [])
            for row in data:
                t_ms, _o, _h, _l, close = row
                if close and close > 0:
                    out[_round_to_hour(int(t_ms) // 1000)] = float(close)
        except Exception as e:
            log.warning("Deribit BTC-DVOL fetch failed [%s→%s]: %s",
                        cursor.date(), chunk_end.date(), e)
        cursor = chunk_end + timedelta(seconds=1)
        time.sleep(0.1)

    return out


def _fetch_deribit_spot_history(start_dt: datetime,
                                 end_dt:   datetime,
                                 chunk_days: int = 30) -> dict[int, float]:
    """
    Fetch hourly BTC spot (perpetual mark price) via get_tradingview_chart_data.
    Returns {hour_timestamp_s: price}.
    """
    out: dict[int, float] = {}
    cursor = start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_dt)
        try:
            r = requests.get(
                f"{DERIBIT}/get_tradingview_chart_data",
                params={
                    "instrument_name": "BTC-PERPETUAL",
                    "start_timestamp": int(cursor.timestamp() * 1000),
                    "end_timestamp":   int(chunk_end.timestamp() * 1000),
                    "resolution":      "60",
                },
                headers=HEADERS,
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            res    = r.json()["result"]
            ticks  = res.get("ticks",  [])
            closes = res.get("close",  [])
            for t_ms, c in zip(ticks, closes):
                if c and c > 0:
                    out[_round_to_hour(int(t_ms) // 1000)] = float(c)
        except Exception as e:
            log.warning("Deribit spot fetch failed [%s→%s]: %s",
                        cursor.date(), chunk_end.date(), e)
        cursor = chunk_end + timedelta(seconds=1)
        time.sleep(0.1)

    return out


def fetch_deribit_history(start_dt: datetime,
                           end_dt:   datetime
                           ) -> tuple[dict[int, float], dict[int, float]]:
    """
    Fetch hourly DVOL (30d ATM vol, %) and BTC spot price over [start_dt, end_dt].
    Returns (dvol_by_hour, spot_by_hour) keyed by Unix timestamp (seconds).
    """
    log.info("Fetching Deribit DVOL history [%s → %s]…",
             start_dt.date(), end_dt.date())
    dvol = _fetch_deribit_dvol(start_dt, end_dt)

    log.info("Fetching Deribit spot history…")
    spot = _fetch_deribit_spot_history(start_dt, end_dt)

    log.info("Got %d DVOL hours, %d spot hours", len(dvol), len(spot))
    return dvol, spot


# ── Polymarket CLOB price history ─────────────────────────────────────────────

def fetch_clob_history(token_id: str,
                        start_ts:  int,
                        end_ts:    int,
                        session:   requests.Session
                        ) -> list[tuple[int, float]]:
    """
    Fetch hourly YES price history from Polymarket CLOB.
    Returns list of (unix_timestamp_seconds, price) sorted ascending.
    """
    try:
        r = session.get(
            f"{CLOB_API}/prices-history",
            params={
                "market":   token_id,
                "fidelity": 60,
                "startTs":  start_ts,
                "endTs":    end_ts,
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        history = r.json().get("history", [])
        result  = []
        for item in history:
            t = int(item["t"])
            p = float(item["p"])
            if 0 < p < 1:
                result.append((_round_to_hour(t), p))
        return sorted(result, key=lambda x: x[0])
    except Exception as e:
        log.debug("CLOB history failed (token %s): %s", token_id, e)
        return []


# ── Event discovery ───────────────────────────────────────────────────────────

def discover_historical_events(session:       requests.Session,
                                lookback_days: int = 90
                                ) -> list[tuple[str, dict]]:
    """
    Find historical BTC price events via slug patterns for past dates.
    Returns list of (contract_type, event_dict).

    Note: same-day hourly events are not included (see module docstring).
    """
    now    = datetime.now(timezone.utc)
    found: dict[str, tuple[str, dict]] = {}   # event_id → (ct, event)

    total_slugs = lookback_days * 2 + 4
    checked     = 0

    # Daily Euro Digital and One-Touch slugs for each past day
    for delta in range(1, lookback_days + 1):
        d     = now - timedelta(days=delta)
        month = _MONTHS[d.month - 1]
        day   = str(d.day)

        for ct, slug in [
            ("european_digital", f"bitcoin-above-on-{month}-{day}"),
            ("one_touch",        f"what-price-will-bitcoin-hit-on-{month}-{day}"),
        ]:
            checked += 1
            if checked % 20 == 0:
                log.info("Discovery: %d/%d slugs checked, %d events found",
                         checked, total_slugs, len(found))

            event = _fetch_event_by_slug(session, slug)
            if event:
                eid = str(event.get("id", ""))
                if eid and eid not in found:
                    if not event.get("markets"):
                        event = _fetch_event_by_id(session, eid) or event
                    found[eid] = (ct, event)
            time.sleep(REQUEST_DELAY)

    # Monthly One-Touch slugs (past 3 months)
    for month_back in range(1, 4):
        m_date = (now.replace(day=1) - timedelta(days=30 * month_back)).replace(day=1)
        slug   = (f"what-price-will-bitcoin-hit-in-"
                  f"{_MONTHS[m_date.month-1]}-{m_date.year}")
        event  = _fetch_event_by_slug(session, slug)
        if event:
            eid = str(event.get("id", ""))
            if eid and eid not in found:
                if not event.get("markets"):
                    event = _fetch_event_by_id(session, eid) or event
                found[eid] = ("one_touch", event)
        time.sleep(REQUEST_DELAY)

    log.info("Discovery complete: %d historical events found", len(found))
    return list(found.values())


# ── Per-market data assembly ──────────────────────────────────────────────────

def _compute_iv(yes_price: float, spot: float, strike: float,
                T: float, contract_type: str) -> Optional[float]:
    """Invert yes_price to IV with the same guards as app.py."""
    if not (PRICE_MIN < yes_price < PRICE_MAX) or T < T_MIN:
        return None
    if contract_type == "european_digital":
        iv = implied_vol_european(yes_price, spot, strike, T, HIST_R)
    else:
        iv = implied_vol_one_touch(yes_price, spot, strike, T, HIST_R)
    return iv if (iv is not None and iv <= MAX_IV) else None


def _nearest(lookup: dict[int, float], ts: int, max_gap_s: int = 3600) -> Optional[float]:
    """Return value from {ts: val} nearest to ts, within max_gap_s."""
    if not lookup:
        return None
    best_ts = min(lookup.keys(), key=lambda t: abs(t - ts))
    if abs(best_ts - ts) <= max_gap_s:
        return lookup[best_ts]
    return None


# ── Main builder ──────────────────────────────────────────────────────────────

def build_historical_snapshots(
    output_path:   Path = Path("data/historical.jsonl"),
    lookback_days: int  = 90,
) -> None:
    """
    Main entry point.  Fetches all historical data and writes JSONL output.
    Runtime: 5–20 min depending on lookback and number of markets found.
    """
    output_path.parent.mkdir(exist_ok=True)
    now       = datetime.now(timezone.utc)
    start_dt  = now - timedelta(days=lookback_days)
    start_ts  = int(start_dt.timestamp())
    end_ts    = int(now.timestamp())

    # ── 1. Deribit historical vol + spot ─────────────────────────────────────
    dvol_by_hour, spot_by_hour = fetch_deribit_history(start_dt, now)
    if not dvol_by_hour:
        log.error("No Deribit DVOL data — aborting.")
        return

    # ── 2. Discover historical events ────────────────────────────────────────
    session = _make_session()
    events  = discover_historical_events(session, lookback_days)
    if not events:
        log.error("No historical events found — check slug patterns.")
        return

    # ── 3. For each market, fetch CLOB price history ──────────────────────────
    # Collect all (timestamp, contract_row) pairs, then group by hour

    # {hour_ts: {market_id: contract_row_dict}}
    by_hour: dict[int, dict[str, dict]] = {}

    parsers = {
        "european_digital": _parse_euro_market,
        "one_touch":        _parse_ot_market,
    }

    n_parsed = n_no_token = n_clob_empty = n_no_deribit = n_rows = 0

    for ct, event in events:
        expiry = _parse_iso(event.get("endDate", ""))
        if expiry is None:
            continue
        event_title = event.get("title", "")

        for m in event.get("markets", []):
            contract = parsers[ct](m, expiry, str(event.get("id", "")), event_title)
            if contract is None:
                continue
            n_parsed += 1

            token_id = _extract_yes_token_id(m)
            if not token_id:
                n_no_token += 1
                log.debug("No CLOB token for market %s", m.get("id"))
                continue

            # Gamma API omits liquidityNum for resolved markets; fall back to
            # volumeNum (total lifetime volume) as a liquidity proxy. Overstates
            # liquidity early in a market's life but is still the best signal
            # available for filtering out illiquid dead markets.
            liquidity = float(
                m.get("liquidityNum") or m.get("liquidity") or
                m.get("volumeNum")    or m.get("volume")    or 0
            )

            # Use per-market time bounds — the global 90-day window exceeds
            # the CLOB API's max range and returns 400.
            m_start = _parse_iso(m.get("startDate") or m.get("startDateIso") or "")
            m_end   = _parse_iso(m.get("endDate") or "")
            if m_start and m_end:
                m_start_ts = int((m_start - timedelta(hours=1)).timestamp())
                m_end_ts   = int((m_end   + timedelta(hours=1)).timestamp())
            else:
                m_start_ts = start_ts
                m_end_ts   = end_ts

            history = fetch_clob_history(token_id, m_start_ts, m_end_ts, session)
            time.sleep(REQUEST_DELAY)

            if not history:
                n_clob_empty += 1
                log.debug("CLOB returned 0 points for %s (token %s)",
                          contract.question[:50], token_id[:12])
                continue

            log.debug("CLOB: %d points for %s", len(history), contract.question[:50])

            for hour_ts, yes_price in history:
                spot = _nearest(spot_by_hour, hour_ts)
                dvol = _nearest(dvol_by_hour, hour_ts)

                if spot is None or dvol is None or spot <= 0:
                    n_no_deribit += 1
                    continue

                ts_dt  = datetime.fromtimestamp(hour_ts, tz=timezone.utc)
                T      = max((expiry - ts_dt).total_seconds() / 86400 / 365.25, 1e-6)

                iv_poly    = _compute_iv(yes_price, spot, contract.strike, T, ct)
                iv_deribit = dvol / 100.0   # DVOL is in %, convert to fraction

                vol_gap = (iv_poly - iv_deribit) if (iv_poly is not None) else None

                row = {
                    "market_id":     contract.market_id,
                    "contract_type": ct,
                    "question":      contract.question,
                    "strike":        contract.strike,
                    "direction":     contract.direction,
                    "expiry":        expiry.isoformat(),
                    "T_years":       round(T, 6),
                    "yes_price":     round(yes_price, 4),
                    "liquidity":     round(liquidity, 0),
                    "iv_poly":       round(iv_poly * 100, 2)    if iv_poly   is not None else None,
                    "iv_deribit":    round(iv_deribit * 100, 2) if iv_deribit is not None else None,
                    "vol_gap":       round(vol_gap * 100, 2)    if vol_gap   is not None else None,
                }

                by_hour.setdefault(hour_ts, {})[contract.market_id] = row
                n_rows += 1

    log.info("Assembly: %d markets parsed, %d no token, %d CLOB empty, "
             "%d no Deribit match, %d rows added",
             n_parsed, n_no_token, n_clob_empty, n_no_deribit, n_rows)

    if not by_hour:
        log.error("No historical data assembled — check counters above.")
        return

    # ── 4. Write one snapshot per hour ────────────────────────────────────────
    n_snaps = 0
    with open(output_path, "w") as f:
        for hour_ts in sorted(by_hour.keys()):
            contracts = list(by_hour[hour_ts].values())
            if not contracts:
                continue
            spot = _nearest(spot_by_hour, hour_ts) or 0.0
            snap = {
                "ts":        datetime.fromtimestamp(hour_ts, tz=timezone.utc).isoformat(),
                "spot":      round(spot, 2),
                "source":    "historical",
                "contracts": contracts,
            }
            f.write(json.dumps(snap) + "\n")
            n_snaps += 1

    log.info("Wrote %d hourly snapshots to %s", n_snaps, output_path)

    # ── 5. Summary ─────────────────────────────────────────────────────────────
    all_contracts = [c for h in by_hour.values() for c in h.values()]
    with_gap   = [c for c in all_contracts if c["vol_gap"] is not None]
    n_markets  = len({c["market_id"] for c in all_contracts})
    span_start = datetime.fromtimestamp(min(by_hour.keys()), tz=timezone.utc)
    span_end   = datetime.fromtimestamp(max(by_hour.keys()), tz=timezone.utc)

    print(f"\n{'='*60}")
    print(f"  Historical dataset built")
    print(f"{'='*60}")
    print(f"  Span:              {span_start.date()}  →  {span_end.date()}")
    print(f"  Hourly snapshots:  {n_snaps:,}")
    print(f"  Unique markets:    {n_markets}")
    print(f"  Contract rows:     {len(all_contracts):,}")
    print(f"  Rows with IV gap:  {len(with_gap):,}")
    print(f"  Output:            {output_path}")
    print(f"\n  Run backtest with:")
    print(f"    python backtest.py --snapshot {output_path}")
    print()
    print("  REMINDER: iv_deribit is DVOL (30d ATM), not smile-interpolated.")
    print("  Vol gaps for ITM/OTM strikes will include ~2–5pp skew error.")
    print("  Use --entry 7 or higher to filter out noise from this effect.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="PolyVol historical data fetcher")
    parser.add_argument("--lookback", type=int, default=90,
                        help="Days of history to fetch (default 90)")
    parser.add_argument("--output", default="data/historical.jsonl",
                        help="Output JSONL path (default data/historical.jsonl)")
    args = parser.parse_args()

    build_historical_snapshots(
        output_path   = Path(args.output),
        lookback_days = args.lookback,
    )
