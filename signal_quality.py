"""
PolyVol Signal Quality Report

Tests whether vol-gap signals predict short-term yes_price movement,
WITHOUT waiting for contract expiry.

Methodology:
  1. Build per-market time series of (ts, yes_price, vol_gap) from snapshots.
  2. Identify signal onsets: first snapshot where |vol_gap| >= entry_threshold
     after gap was below threshold (cooldown prevents same spike being counted twice).
  3. At each forward horizon (0.5h, 1h, 2h, 4h, 8h, 12h), look up yes_price.
  4. Compute fractional P&L:
       buy_yes:  (exit - entry) / entry
       sell_yes: (entry - exit) / (1 - entry)
  5. Report win rate and mean return at each horizon, broken down by direction.

Usage:
    python signal_quality.py [--snapshot data/snapshots.jsonl] [--entry 5.0]
                             [--max-gap 50] [--cooldown 2.0] [--min-data 3]
"""

import argparse
import bisect
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from logger import load_snapshots

# ── Defaults ─────────────────────────────────────────────────────────────────

ENTRY_THRESHOLD = 7.0    # pp — signal fires when |gap| crosses this (raised to match backtest)
MAX_ENTRY_GAP   = 50.0   # pp — ignore spikes above this (IV noise)
COOLDOWN_HOURS  = 2.0    # hours after signal onset before same market can fire again
MIN_DATA_POINTS = 3      # min snapshots a market needs before it's analysed
HORIZONS_H      = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]   # forward look horizons in hours
BUY_ONLY        = True   # only evaluate buy_yes signals
CONTRACT_TYPE   = None   # None = all; "one_touch" or "european_digital" to restrict


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _frac_pnl(side: str, entry: float, exit_p: float) -> float:
    """Fractional P&L (return on capital at risk) for a binary position."""
    if side == "buy_yes":
        return (exit_p - entry) / entry
    else:  # sell_yes = buy NO
        return (entry - exit_p) / (1.0 - entry)


def _lookup_price(times: list[datetime], prices: list[float], target: datetime) -> Optional[float]:
    """Return yes_price at the snapshot closest to but not before target."""
    idx = bisect.bisect_left(times, target)
    if idx >= len(times):
        return None
    return prices[idx]


# ── Core analysis ─────────────────────────────────────────────────────────────

def build_market_series(snapshots: list[dict]) -> dict[str, dict]:
    """
    Build per-market time series.
    Returns: market_id → {
        "times":  [datetime, ...],
        "prices": [float, ...],
        "gaps":   [float, ...],
        "meta":   {contract_type, question, strike, direction, expiry}
    }
    """
    series: dict[str, dict] = {}

    for snap in snapshots:
        ts = _parse_ts(snap["ts"])
        for c in snap["contracts"]:
            mid = c["market_id"]
            gap = c.get("vol_gap")
            if gap is None:
                continue
            if mid not in series:
                series[mid] = {
                    "times":  [],
                    "prices": [],
                    "gaps":   [],
                    "meta": {
                        "contract_type": c.get("contract_type", ""),
                        "question":      c.get("question", ""),
                        "strike":        c.get("strike", 0),
                        "direction":     c.get("direction", ""),
                        "expiry":        c.get("expiry", ""),
                    },
                }
            series[mid]["times"].append(ts)
            series[mid]["prices"].append(c["yes_price"])
            series[mid]["gaps"].append(gap)

    return series


def find_signal_onsets(
    series: dict,
    entry_threshold: float,
    max_entry_gap: float,
    cooldown_hours: float,
    min_data_points: int,
    buy_only: bool = BUY_ONLY,
    contract_type: Optional[str] = None,
) -> list[dict]:
    """
    Scan each market's time series for signal onsets.
    A signal onset is the FIRST snapshot where |gap| >= entry_threshold
    after gap was below threshold (or after cooldown period has elapsed).

    Returns list of signal dicts.
    """
    signals = []
    cooldown = timedelta(hours=cooldown_hours)

    for mid, s in series.items():
        times  = s["times"]
        prices = s["prices"]
        gaps   = s["gaps"]

        if len(times) < min_data_points:
            continue

        meta = s["meta"]

        if contract_type and meta["contract_type"] != contract_type:
            continue
        last_signal_ts: Optional[datetime] = None
        was_below = True   # starts as "no active gap"

        for i, (ts, price, gap) in enumerate(zip(times, prices, gaps)):
            abs_gap = abs(gap)

            # Track whether gap is below threshold (reset for next onset)
            if abs_gap < entry_threshold:
                was_below = True
                continue

            # Gap is above threshold
            if abs_gap > max_entry_gap:
                was_below = False
                continue

            # Price bounds: skip deep ITM/OTM (matches backtest filter)
            if not (0.10 < price < 0.90):
                was_below = False
                continue

            # Only fire on onset (transition from below → above)
            if not was_below:
                continue

            # Cooldown: don't fire again too soon for same market
            if last_signal_ts is not None and ts - last_signal_ts < cooldown:
                was_below = False
                continue

            # Signal fires
            was_below = False
            last_signal_ts = ts
            side = "buy_yes" if gap < 0 else "sell_yes"

            if buy_only and side == "sell_yes":
                continue

            signals.append({
                "market_id":     mid,
                "contract_type": meta["contract_type"],
                "question":      meta["question"],
                "strike":        meta["strike"],
                "direction":     meta["direction"],
                "expiry":        meta["expiry"],
                "onset_ts":      ts,
                "onset_idx":     i,
                "entry_price":   price,
                "entry_gap":     gap,
                "side":          side,
                "times":         times,
                "prices":        prices,
                "gaps":          gaps,
            })

    signals.sort(key=lambda x: x["onset_ts"])
    return signals


def evaluate_signals(
    signals: list[dict],
    horizons_h: list[float],
) -> dict:
    """
    For each signal, look up yes_price at each forward horizon and compute P&L.

    Returns: {
        "by_horizon": {h: {"n": int, "wins": int, "returns": [float], "gap_decays": [float]}},
        "by_horizon_side": {(h, side): same},
        "by_horizon_type": {(h, contract_type): same},
        "signals": [enriched signal dicts with returns at each horizon],
    }
    """
    by_horizon:      dict[float, dict] = {}
    by_horizon_side: dict[tuple, dict] = {}
    by_horizon_type: dict[tuple, dict] = {}

    def _bucket():
        return {"n": 0, "wins": 0, "returns": [], "gap_decays": []}

    for h in horizons_h:
        by_horizon[h] = _bucket()
        for side in ("buy_yes", "sell_yes"):
            by_horizon_side[(h, side)] = _bucket()
        for ct in ("european_digital", "one_touch"):
            by_horizon_type[(h, ct)] = _bucket()

    enriched = []
    for sig in signals:
        times  = sig["times"]
        prices = sig["prices"]
        gaps   = sig["gaps"]
        onset  = sig["onset_ts"]
        entry  = sig["entry_price"]
        side   = sig["side"]
        ct     = sig["contract_type"]
        entry_gap = sig["entry_gap"]

        horizon_results = {}
        for h in horizons_h:
            target = onset + timedelta(hours=h)
            exit_p = _lookup_price(times, prices, target)
            if exit_p is None:
                horizon_results[h] = None
                continue

            # Find gap at that point too
            exit_idx = bisect.bisect_left(times, target)
            if exit_idx < len(gaps):
                exit_gap = gaps[exit_idx]
                gap_decay = abs(entry_gap) - abs(exit_gap)   # positive = gap narrowed (good)
            else:
                exit_gap = None
                gap_decay = None

            ret = _frac_pnl(side, entry, exit_p)
            win = ret > 0
            horizon_results[h] = {"return": ret, "exit_price": exit_p,
                                   "exit_gap": exit_gap, "gap_decay": gap_decay, "win": win}

            for bucket in [
                by_horizon[h],
                by_horizon_side[(h, side)],
                by_horizon_type[(h, ct)],
            ]:
                bucket["n"]    += 1
                bucket["wins"] += int(win)
                bucket["returns"].append(ret)
                if gap_decay is not None:
                    bucket["gap_decays"].append(gap_decay)

        enriched.append({**sig, "horizon_results": horizon_results})

    return {
        "by_horizon":      by_horizon,
        "by_horizon_side": by_horizon_side,
        "by_horizon_type": by_horizon_type,
        "signals":         enriched,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def _mean(lst: list[float]) -> Optional[float]:
    return sum(lst) / len(lst) if lst else None


def print_report(
    result: dict,
    signals: list[dict],
    horizons_h: list[float],
    entry_threshold: float,
    snapshots: list[dict],
) -> None:
    by_h   = result["by_horizon"]
    by_hs  = result["by_horizon_side"]
    by_ht  = result["by_horizon_type"]
    sigs   = result["signals"]

    span = ""
    if snapshots:
        t0 = snapshots[0]["ts"]
        t1 = snapshots[-1]["ts"]
        dt = (_parse_ts(t1) - _parse_ts(t0)).total_seconds() / 3600
        span = f"  [{t0[:16]} → {t1[:16]}  ({dt:.1f}h)]"

    print(f"\n{'='*72}")
    print(f"  SIGNAL QUALITY REPORT   entry threshold: {entry_threshold} pp")
    print(f"  {len(snapshots)} snapshots   {len(signals)} signal onsets{span}")
    print(f"{'='*72}")

    # ── Overall by horizon ────────────────────────────────────────────────────
    print(f"\n  {'Horizon':>8}  {'N':>5}  {'WinRate':>8}  {'MeanRet':>9}  {'MeanGapDecay':>13}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*8}  {'-'*9}  {'-'*13}")
    for h in horizons_h:
        b   = by_h[h]
        n   = b["n"]
        if n == 0:
            print(f"  {h:>7.1f}h  {'—':>5}")
            continue
        wr  = b["wins"] / n * 100
        mr  = _mean(b["returns"])
        mgd = _mean(b["gap_decays"])
        mr_str  = f"{mr*100:+.2f}%"  if mr  is not None else "  —"
        mgd_str = f"{mgd:+.2f}pp"    if mgd is not None else "  —"
        print(f"  {h:>7.1f}h  {n:>5}  {wr:>7.1f}%  {mr_str:>9}  {mgd_str:>13}")

    # ── By direction ──────────────────────────────────────────────────────────
    for side in ("buy_yes", "sell_yes"):
        print(f"\n  [{side}]")
        print(f"  {'Horizon':>8}  {'N':>5}  {'WinRate':>8}  {'MeanRet':>9}")
        print(f"  {'-'*8}  {'-'*5}  {'-'*8}  {'-'*9}")
        for h in horizons_h:
            b = by_hs[(h, side)]
            n = b["n"]
            if n == 0:
                print(f"  {h:>7.1f}h  {'—':>5}")
                continue
            wr = b["wins"] / n * 100
            mr = _mean(b["returns"])
            mr_str = f"{mr*100:+.2f}%" if mr is not None else "  —"
            print(f"  {h:>7.1f}h  {n:>5}  {wr:>7.1f}%  {mr_str:>9}")

    # ── By contract type ──────────────────────────────────────────────────────
    for ct in ("european_digital", "one_touch"):
        ct_sigs = [s for s in sigs if s["contract_type"] == ct]
        if not ct_sigs:
            continue
        print(f"\n  [{ct}]")
        print(f"  {'Horizon':>8}  {'N':>5}  {'WinRate':>8}  {'MeanRet':>9}")
        print(f"  {'-'*8}  {'-'*5}  {'-'*8}  {'-'*9}")
        for h in horizons_h:
            b = by_ht[(h, ct)]
            n = b["n"]
            if n == 0:
                print(f"  {h:>7.1f}h  {'—':>5}")
                continue
            wr = b["wins"] / n * 100
            mr = _mean(b["returns"])
            mr_str = f"{mr*100:+.2f}%" if mr is not None else "  —"
            print(f"  {h:>7.1f}h  {n:>5}  {wr:>7.1f}%  {mr_str:>9}")

    # ── Individual signal log ─────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  SIGNAL LOG  ({len(sigs)} total)")
    print(f"{'='*72}")
    h1, h4 = 1.0, 4.0
    print(f"  {'Time':16s}  {'Side':10s}  {'K':>8}  {'Entry':>6}  {'Gap@E':>7}  "
          f"{'Ret@1h':>7}  {'Ret@4h':>7}  {'Type'}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*20}")
    for sig in sigs:
        ts_str = sig["onset_ts"].strftime("%m-%d %H:%M")
        r1 = sig["horizon_results"].get(h1)
        r4 = sig["horizon_results"].get(h4)
        r1_str = f"{r1['return']*100:+.1f}%" if r1 else "  —"
        r4_str = f"{r4['return']*100:+.1f}%" if r4 else "  —"
        print(f"  {ts_str:16s}  {sig['side']:10s}  {sig['strike']:>8,.0f}  "
              f"{sig['entry_price']:>6.3f}  {sig['entry_gap']:>+7.1f}pp  "
              f"{r1_str:>7}  {r4_str:>7}  {sig['contract_type']}")

    # ── Gap distribution of signals ───────────────────────────────────────────
    if sigs:
        all_gaps = [abs(s["entry_gap"]) for s in sigs]
        buy_n  = sum(1 for s in sigs if s["side"] == "buy_yes")
        sell_n = sum(1 for s in sigs if s["side"] == "sell_yes")
        print(f"\n  Signal gap distribution: min={min(all_gaps):.1f}pp  "
              f"max={max(all_gaps):.1f}pp  mean={_mean(all_gaps):.1f}pp")
        print(f"  Directions: {buy_n} buy_yes  {sell_n} sell_yes")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PolyVol signal quality report")
    parser.add_argument("--snapshot",  default="data/snapshots.jsonl")
    parser.add_argument("--entry",     type=float, default=ENTRY_THRESHOLD,
                        help="Entry threshold in vol pp (default %(default)s)")
    parser.add_argument("--max-gap",   type=float, default=MAX_ENTRY_GAP,
                        help="Max |gap| pp to include; filters IV noise (default %(default)s)")
    parser.add_argument("--cooldown",  type=float, default=COOLDOWN_HOURS,
                        help="Hours between signals for same market (default %(default)s)")
    parser.add_argument("--min-data",  type=int,   default=MIN_DATA_POINTS,
                        help="Min snapshots per market to include (default %(default)s)")
    parser.add_argument("--buy-only",  action=argparse.BooleanOptionalAction, default=BUY_ONLY,
                        help="Only evaluate buy_yes signals; --no-buy-only to include sell_yes (default %(default)s)")
    parser.add_argument("--contract-type", choices=["european_digital", "one_touch", "all"],
                        default="all",
                        help="Restrict analysis to one contract type (default: all)")
    args = parser.parse_args()

    ct_filter = None if args.contract_type == "all" else args.contract_type

    snaps = load_snapshots(Path(args.snapshot))
    if not snaps:
        print(f"No snapshots found at {args.snapshot}. Run app.py first.")
        raise SystemExit(0)

    series  = build_market_series(snaps)
    signals = find_signal_onsets(series, args.entry, args.max_gap,
                                 args.cooldown, args.min_data,
                                 buy_only=args.buy_only,
                                 contract_type=ct_filter)

    if not signals:
        print(f"No signals found with entry threshold={args.entry}pp.")
        raise SystemExit(0)

    result = evaluate_signals(signals, HORIZONS_H)
    print_report(result, signals, HORIZONS_H, args.entry, snaps)
