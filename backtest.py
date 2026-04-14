"""
PolyVol backtester.

Replays snapshot history (data/snapshots.jsonl) and simulates a vol-gap
mean-reversion strategy on Polymarket binary contracts.

Strategy rules:
  Entry:
    - Open a position when |vol_gap| > ENTRY_THRESHOLD (default 5 pp)
    - BUY YES  when vol_gap < -ENTRY_THRESHOLD  (poly IV too low → contract cheap)
    - SELL YES when vol_gap > +ENTRY_THRESHOLD  (poly IV too high → contract dear)
    - Only enter if liquidity >= MIN_LIQUIDITY and yes_price in [PRICE_MIN, PRICE_MAX]

  Exit:
    - Close when |vol_gap| < EXIT_THRESHOLD (default 2 pp)  — gap closed
    - Close when T_years < T_MIN_EXIT (e.g. 30 min to expiry) — time stop
    - Hold to expiry if neither condition met (resolved at 0 or 1)

  Sizing:
    - Fixed $POSITION_SIZE per trade

  Fees (Polymarket):
    - 2% of net winnings on the winning side
    - Applied at exit: fee = 0.02 * profit  if profit > 0 else 0

  Slippage:
    - SLIPPAGE_CENTS added to entry price (buying costs more, selling costs less)

Usage:
    python backtest.py [--snapshot data/snapshots.jsonl] [--entry 5] [--exit 2]
"""

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from logger import load_snapshots
from vol_math import european_digital_price, one_touch_price

# ── Config ────────────────────────────────────────────────────────────────────

ENTRY_THRESHOLD    = 7.0    # vol gap pp to trigger entry
EXIT_THRESHOLD     = 2.0    # vol gap pp to trigger exit
MAX_ENTRY_GAP      = 50.0   # ignore signals with |gap| > 50pp — IV inversion noise
POSITION_SIZE      = 100.0  # USD per trade
MIN_LIQUIDITY      = 1_000   # minimum pool liquidity to enter (matches app.py MIN_IV_LIQ)
PRICE_MIN          = 0.05   # skip deep OTM/ITM
PRICE_MAX          = 0.95
MIN_MONEYNESS      = 0.01   # skip strikes within 1% of spot
T_MIN_EXIT         = 0.5 / 365.25 / 24   # 30 min in years — time stop
MAX_HOLD_HOURS     = 2.0                 # exit after this many hours if gap hasn't closed
SLIPPAGE_REF_LIQ   = 20_000  # liquidity at which base slippage applies
SLIPPAGE_BASE_CENTS = 0.005   # 0.5¢ at SLIPPAGE_REF_LIQ; scales as sqrt(ref/actual)
MAX_POSITION_FRAC  = 0.10    # max position size as fraction of pool liquidity
POLY_FEE_RATE      = 0.02    # 2% of net winnings

# Directional / filtering defaults
BUY_ONLY           = False   # trade both directions
CONTRACT_TYPES     = None    # None = all; or set {"one_touch"} / {"european_digital"}
TREND_HOURS        = 4.0     # lookback window for BTC trend detection
TREND_THRESHOLD    = 0.01    # skip sell_yes if spot rose > 1% over TREND_HOURS
PRICE_GAP_THRESHOLD = 0.03   # min |yes_price - deribit_fair| to enter
RISK_FREE_RATE      = 0.05   # fallback r (snapshots don't store it)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Position:
    market_id:     str
    contract_type: str
    question:      str
    strike:        float
    direction:     str
    expiry:        str
    side:          str          # "buy_yes" | "sell_yes"
    entry_ts:      str
    entry_price:   float        # price paid (after slippage)
    entry_gap:     float        # vol gap at entry (pp)
    size_usd:      float        # $ committed
    # filled at close:
    exit_ts:       Optional[str]  = None
    exit_price:    Optional[float] = None
    exit_gap:      Optional[float] = None
    exit_reason:   Optional[str]  = None
    pnl:           Optional[float] = None


@dataclass
class BacktestResult:
    trades:           list[Position] = field(default_factory=list)
    # Realized only (closed trades) — the only trustworthy P&L figure
    realized_pnl:     float = 0.0
    win_count:        int   = 0
    loss_count:       int   = 0
    total_fees:       float = 0.0
    total_slippage:   float = 0.0
    # Unrealized mark — noisy for binaries, shown for context only
    unrealized_pnl:   float = 0.0
    open_count:       int   = 0
    filter_stats:     dict  = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _pnl(side: str, entry_price: float, exit_price: float, size_usd: float) -> float:
    """
    P&L for a binary position before fees.
    Buying YES at p, selling at q: profit per $ = (q - p) / p * size ...
    No — Polymarket is a binary market, so:
      BUY YES at p: you pay p per share, receive 1 if YES wins.
      Shares = size_usd / entry_price
      Value at exit = shares * exit_price
      PnL = shares * (exit_price - entry_price)  [mark-to-market]
            OR = shares * (1 - entry_price) if held to YES resolution
            OR = shares * (0 - entry_price) if held to NO resolution
    SELL YES at p (i.e. buy NO at 1-p):
      Equivalent: buy NO shares at (1-p), NO pays 1 if price goes to 0.
      PnL = shares_NO * (exit_no_price - entry_no_price)
          = (size_usd / (1 - entry_price)) * ((1 - exit_price) - (1 - entry_price))
          = (size_usd / (1 - entry_price)) * (entry_price - exit_price)
    """
    if side == "buy_yes":
        shares = size_usd / entry_price
        return shares * (exit_price - entry_price)
    else:  # sell_yes = buy NO
        shares = size_usd / (1.0 - entry_price)
        return shares * (entry_price - exit_price)


def _fee(side: str, entry_price: float, exit_price: float, size_usd: float,
         raw_pnl: float) -> float:
    """
    Polymarket charges 2% of net winnings on the winning side only.
    Winnings = profit in dollar terms (>0).
    """
    if raw_pnl <= 0:
        return 0.0
    return POLY_FEE_RATE * raw_pnl


# ── Main backtester ───────────────────────────────────────────────────────────

def run_backtest(
    snapshots:        list[dict],
    entry_threshold:  float = ENTRY_THRESHOLD,
    exit_threshold:   float = EXIT_THRESHOLD,
    position_size:    float = POSITION_SIZE,
    max_entry_gap:    float = MAX_ENTRY_GAP,
    min_moneyness:    float = MIN_MONEYNESS,
    buy_only:         bool  = BUY_ONLY,
    contract_types:   Optional[set] = None,   # None = allow all
    trend_hours:      float = TREND_HOURS,
    trend_threshold:  float = TREND_THRESHOLD,
    max_hold_hours:   float = MAX_HOLD_HOURS,
) -> BacktestResult:
    """
    Replay snapshots chronologically. One position per market_id at a time.

    Filters applied at entry:
      buy_only        — skip all sell_yes signals (default True; they have delta risk)
      contract_types  — restrict to e.g. {"one_touch"}; None = all types
      trend filter    — skip sell_yes when BTC spot rose > trend_threshold over last
                        trend_hours (requires buy_only=False to matter)
    """
    result   = BacktestResult()
    open_pos: dict[str, Position] = {}   # market_id → Position

    # Filter diagnostic counters
    filt = {
        "gap_null": 0, "liq": 0, "price": 0, "time": 0,
        "gap_large": 0, "moneyness": 0, "gap_small": 0,
        "price_gap_small": 0, "buy_only": 0, "trend": 0, "entered": 0,
    }

    # Rolling deque of (datetime, spot) for trend detection
    trend_window: deque[tuple[datetime, float]] = deque()
    trend_cutoff = timedelta(hours=trend_hours)

    for snap in snapshots:
        ts_str = snap["ts"]
        ts     = _parse_ts(ts_str)
        spot   = snap["spot"]

        # Maintain rolling spot window
        trend_window.append((ts, spot))
        while trend_window and ts - trend_window[0][0] > trend_cutoff:
            trend_window.popleft()

        # BTC trend over the window: positive = rising
        def _spot_trending_up() -> bool:
            if len(trend_window) < 2:
                return False
            oldest_spot = trend_window[0][1]
            return oldest_spot > 0 and (spot - oldest_spot) / oldest_spot > trend_threshold

        for c in snap["contracts"]:
            ts = ts_str   # reuse string form for Position fields
            mid       = c["market_id"]
            gap       = c.get("vol_gap")
            yes_price = c["yes_price"]
            T         = c["T_years"]
            liq       = c["liquidity"]

            if gap is None:
                filt["gap_null"] += 1
                continue

            # ── Check exit for open positions ──────────────────────────────
            if mid in open_pos:
                pos = open_pos[mid]
                exit_reason = None

                held_hours = (_parse_ts(ts) - _parse_ts(pos.entry_ts)).total_seconds() / 3600
                if abs(gap) < exit_threshold:
                    exit_reason = "gap_closed"
                elif T < T_MIN_EXIT:
                    exit_reason = "time_stop"
                elif max_hold_hours > 0 and held_hours >= max_hold_hours:
                    exit_reason = "max_hold"

                if exit_reason:
                    pos.exit_ts     = ts
                    pos.exit_price  = yes_price
                    pos.exit_gap    = gap
                    pos.exit_reason = exit_reason
                    raw = _pnl(pos.side, pos.entry_price, yes_price, pos.size_usd)
                    fee = _fee(pos.side, pos.entry_price, yes_price, pos.size_usd, raw)
                    pos.pnl = raw - fee
                    result.realized_pnl += pos.pnl
                    result.total_fees   += fee
                    if pos.pnl >= 0:
                        result.win_count  += 1
                    else:
                        result.loss_count += 1
                    result.trades.append(pos)
                    del open_pos[mid]
                continue  # don't re-enter same tick

            # ── Check entry ────────────────────────────────────────────────
            if mid in open_pos:
                continue
            if liq < MIN_LIQUIDITY:
                filt["liq"] += 1
                continue
            if not (PRICE_MIN < yes_price < PRICE_MAX):
                filt["price"] += 1
                continue
            if T < T_MIN_EXIT:
                filt["time"] += 1
                continue

            if abs(gap) > max_entry_gap:
                filt["gap_large"] += 1
                continue  # IV inversion noise — gap too large to be a real signal

            strike = c.get("strike", 0)
            if spot and strike and abs(spot - strike) / spot < min_moneyness:
                filt["moneyness"] += 1
                continue  # too close to ATM — high delta, not a pure vol bet

            # Contract type filter
            if contract_types and c.get("contract_type") not in contract_types:
                continue

            # Pre-screen: require meaningful IV divergence before computing price
            if abs(gap) < entry_threshold:
                filt["gap_small"] += 1
                continue

            # Direction via price gap — correct for both OTM and ITM digitals.
            # vol_gap sign is unreliable for ITM contracts (negative vega flips it).
            iv_d = c.get("iv_deribit")
            if iv_d is None:
                filt["gap_null"] += 1
                continue
            try:
                sigma_d  = iv_d / 100.0
                ct       = c.get("contract_type", "european_digital")
                raw_fair = (one_touch_price(spot, strike, T, RISK_FREE_RATE, sigma_d)
                            if ct == "one_touch"
                            else european_digital_price(spot, strike, T, RISK_FREE_RATE, sigma_d))
                dirn         = c.get("direction", "above")
                deribit_fair = 1.0 - raw_fair if dirn == "below" else raw_fair
            except Exception:
                filt["gap_null"] += 1
                continue

            price_gap = yes_price - deribit_fair  # >0 → poly dear; <0 → poly cheap

            side = None
            if price_gap < -PRICE_GAP_THRESHOLD:
                side = "buy_yes"    # poly cheaper than Deribit fair → underpriced
            elif price_gap > PRICE_GAP_THRESHOLD:
                side = "sell_yes"   # poly more expensive than Deribit fair → overpriced

            if side is None:
                filt["price_gap_small"] += 1
                continue

            # Directional filters for sell_yes
            if side == "sell_yes":
                if buy_only:
                    filt["buy_only"] += 1
                    continue   # sell_yes disabled
                if _spot_trending_up():
                    filt["trend"] += 1
                    continue   # BTC trending up — short delta exposure, skip

            # Liquidity-scaled slippage: wider spread on thinner pools
            slip = SLIPPAGE_BASE_CENTS * (SLIPPAGE_REF_LIQ / max(liq, 1)) ** 0.5

            # Cap position size at MAX_POSITION_FRAC of pool (avoid moving market)
            capped_size = min(position_size, liq * MAX_POSITION_FRAC)

            # Apply slippage to entry price
            entry_price = (yes_price + slip if side == "buy_yes"
                           else yes_price - slip)
            entry_price = max(0.001, min(0.999, entry_price))

            slippage_cost = slip * (capped_size / entry_price)
            result.total_slippage += slippage_cost

            filt["entered"] += 1
            open_pos[mid] = Position(
                market_id     = mid,
                contract_type = c["contract_type"],
                question      = c["question"],
                strike        = c["strike"],
                direction     = c["direction"],
                expiry        = c["expiry"],
                side          = side,
                entry_ts      = ts,
                entry_price   = entry_price,
                entry_gap     = gap,
                size_usd      = capped_size,
            )

    result.filter_stats = filt

    # Mark still-open positions at last known price — unrealized, not counted in P&L
    last_snap = snapshots[-1] if snapshots else {}
    last_prices = {c["market_id"]: c["yes_price"]
                   for c in last_snap.get("contracts", [])}
    for mid, pos in open_pos.items():
        exit_price = last_prices.get(mid, pos.entry_price)
        pos.exit_ts     = last_snap.get("ts", "")
        pos.exit_price  = exit_price
        pos.exit_reason = "still_open"
        raw = _pnl(pos.side, pos.entry_price, exit_price, pos.size_usd)
        fee = _fee(pos.side, pos.entry_price, exit_price, pos.size_usd, raw)
        pos.pnl = raw - fee
        # Unrealized: tracked separately, NOT added to realized_pnl
        result.unrealized_pnl += pos.pnl
        result.open_count     += 1
        result.trades.append(pos)

    return result


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(result: BacktestResult) -> None:
    trades   = result.trades
    closed   = [t for t in trades if t.exit_reason != "still_open"]
    open_pos = [t for t in trades if t.exit_reason == "still_open"]
    n_closed = len(closed)
    wr = result.win_count / n_closed * 100 if n_closed else 0.0

    print(f"\n{'='*70}")
    print(f"  REALIZED P&L  ({n_closed} closed trades)  ← the only number that matters")
    print(f"{'='*70}")
    print(f"  Realized P&L:           ${result.realized_pnl:+.2f}")
    print(f"  Fees paid:              ${result.total_fees:.2f}")
    print(f"  Slippage cost:          ${result.total_slippage:.2f}")
    print(f"  Net after fees+slip:    ${result.realized_pnl - result.total_slippage:+.2f}")
    print(f"  Win rate:               {wr:.1f}%  ({result.win_count}W / {result.loss_count}L)")
    print()

    by_reason = {}
    for t in closed:
        by_reason.setdefault(t.exit_reason, []).append(t.pnl or 0)
    for reason, pnls in sorted(by_reason.items()):
        avg = sum(pnls) / len(pnls)
        total = sum(pnls)
        print(f"  Exit [{reason:15s}]: {len(pnls):3d} trades  avg ${avg:+.2f}  total ${total:+.2f}")

    print()
    print(f"{'='*70}")
    print(f"  STILL OPEN  ({result.open_count} positions)  ← unreliable mark for binaries")
    print(f"{'='*70}")
    if open_pos:
        print(f"  Unrealized P&L (last snapshot price): ${result.unrealized_pnl:+.2f}")
        print(f"  NOTE: binary options near expiry can mark at 0.50 but resolve 0 or 1.")
        print(f"  Do not add this to realized P&L. Run longer to close these out.")
        print()
        for t in sorted(open_pos, key=lambda x: x.expiry or ""):
            held_h = ((_parse_ts(t.exit_ts) - _parse_ts(t.entry_ts)).total_seconds() / 3600
                      if t.exit_ts else 0)
            print(f"  {t.side:10s}  K=${t.strike:>9,.0f}  entry={t.entry_price:.3f}"
                  f"  last={t.exit_price:.3f}  gap@entry={t.entry_gap:+.1f}pp"
                  f"  held={held_h:.1f}h  exp={t.expiry[:10] if t.expiry else '?'}")
    else:
        print("  None.")

    print()
    print(f"{'='*70}")
    print(f"  CLOSED TRADE LOG")
    print(f"{'='*70}")
    print(f"  {'Side':10s} {'Type':20s} {'Strike':>10s} {'Entry':>7s} {'Exit':>7s} "
          f"{'Gap@E':>7s} {'Gap@X':>7s} {'P&L':>8s}  Reason")
    print(f"  {'-'*10} {'-'*20} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}  ------")
    for t in sorted(closed, key=lambda x: x.entry_ts):
        ep = f"{t.entry_price:.3f}"
        xp = f"{t.exit_price:.3f}" if t.exit_price is not None else "  ?"
        ge = f"{t.entry_gap:+.1f}pp"
        gx = f"{t.exit_gap:+.1f}pp" if t.exit_gap is not None else "    ?"
        print(f"  {t.side:10s} {t.contract_type:20s} {t.strike:>10,.0f} "
              f"{ep:>7s} {xp:>7s} {ge:>7s} {gx:>7s} ${t.pnl:+7.2f}  {t.exit_reason or ''}")

    if result.filter_stats:
        f = result.filter_stats
        total = sum(f.values())
        print(f"\n{'='*70}")
        print(f"  ENTRY FILTER STATS  (contract-ticks evaluated)")
        print(f"{'='*70}")
        print(f"  vol_gap=null (no IV):  {f['gap_null']:>8,}")
        print(f"  liquidity < min:       {f['liq']:>8,}")
        print(f"  price out of range:    {f['price']:>8,}")
        print(f"  near expiry:           {f['time']:>8,}")
        print(f"  |gap| too large:       {f['gap_large']:>8,}")
        print(f"  near ATM (moneyness):  {f['moneyness']:>8,}")
        print(f"  gap too small:         {f['gap_small']:>8,}")
        print(f"  buy_only blocked:      {f['buy_only']:>8,}")
        print(f"  trend filter:          {f['trend']:>8,}")
        print(f"  → ENTERED:             {f['entered']:>8,}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PolyVol backtester")
    parser.add_argument("--snapshot", default="data/snapshots.jsonl",
                        help="Path to snapshots JSONL file")
    parser.add_argument("--entry", type=float, default=ENTRY_THRESHOLD,
                        help="Entry threshold in vol pp (default %(default)s)")
    parser.add_argument("--exit",  type=float, default=EXIT_THRESHOLD,
                        help="Exit threshold in vol pp (default %(default)s)")
    parser.add_argument("--size",  type=float, default=POSITION_SIZE,
                        help="Position size in USD (default %(default)s)")
    parser.add_argument("--max-gap", type=float, default=MAX_ENTRY_GAP,
                        help="Max |vol gap| pp for entry; filters IV noise (default %(default)s)")
    parser.add_argument("--min-moneyness", type=float, default=MIN_MONEYNESS,
                        help="Min |spot-strike|/spot to enter; skips near-ATM high-delta (default %(default)s)")
    parser.add_argument("--buy-only", action=argparse.BooleanOptionalAction, default=BUY_ONLY,
                        help="Only trade buy_yes signals; use --no-buy-only to allow sell_yes (default %(default)s)")
    parser.add_argument("--contract-type", choices=["european_digital", "one_touch", "all"],
                        default="all",
                        help="Restrict to one contract type (default: all)")
    parser.add_argument("--trend-hours", type=float, default=TREND_HOURS,
                        help="Lookback window in hours for BTC trend filter on sell_yes (default %(default)s)")
    parser.add_argument("--trend-threshold", type=float, default=TREND_THRESHOLD,
                        help="Skip sell_yes if BTC rose more than this fraction over trend window (default %(default)s)")
    parser.add_argument("--max-hold", type=float, default=MAX_HOLD_HOURS,
                        help="Max hours to hold a position before force-exit; 0 = disabled (default %(default)s)")
    args = parser.parse_args()

    ct_filter = None if args.contract_type == "all" else {args.contract_type}

    snaps = load_snapshots(Path(args.snapshot))
    if not snaps:
        print(f"No snapshots found at {args.snapshot}.")
        print("Run app.py first to collect data, then re-run this script.")
        sys.exit(0)

    span_start = snaps[0]["ts"]
    span_end   = snaps[-1]["ts"]
    print(f"Loaded {len(snaps)} snapshots  [{span_start}  →  {span_end}]")
    print(f"Entry: {args.entry}pp   Exit: {args.exit}pp   Max gap: {args.max_gap}pp   "
          f"Min moneyness: {args.min_moneyness*100:.0f}%   Size: ${args.size:.0f}")
    print(f"Buy-only: {args.buy_only}   Contract type: {args.contract_type}   "
          f"Trend filter: {args.trend_hours}h / {args.trend_threshold*100:.1f}%   "
          f"Max hold: {args.max_hold}h")

    result = run_backtest(
        snaps,
        entry_threshold = args.entry,
        exit_threshold  = args.exit,
        position_size   = args.size,
        max_entry_gap   = args.max_gap,
        min_moneyness   = args.min_moneyness,
        buy_only        = args.buy_only,
        contract_types  = ct_filter,
        trend_hours     = args.trend_hours,
        trend_threshold = args.trend_threshold,
        max_hold_hours  = args.max_hold,
    )
    print_report(result)
