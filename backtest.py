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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from logger import load_snapshots

# ── Config ────────────────────────────────────────────────────────────────────

ENTRY_THRESHOLD    = 5.0    # vol gap pp to trigger entry
EXIT_THRESHOLD     = 2.0    # vol gap pp to trigger exit
POSITION_SIZE      = 100.0  # USD per trade
MIN_LIQUIDITY      = 5_000  # minimum pool liquidity to enter
PRICE_MIN          = 0.05   # skip deep OTM/ITM
PRICE_MAX          = 0.95
T_MIN_EXIT         = 0.5 / 365.25 / 24   # 30 min in years — time stop
SLIPPAGE_REF_LIQ   = 20_000  # liquidity at which base slippage applies
SLIPPAGE_BASE_CENTS = 0.005   # 0.5¢ at SLIPPAGE_REF_LIQ; scales as sqrt(ref/actual)
MAX_POSITION_FRAC  = 0.10    # max position size as fraction of pool liquidity
POLY_FEE_RATE      = 0.02    # 2% of net winnings


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
    trades:         list[Position] = field(default_factory=list)
    total_pnl:      float = 0.0
    win_count:      int   = 0
    loss_count:     int   = 0
    total_fees:     float = 0.0
    total_slippage: float = 0.0


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
    snapshots:       list[dict],
    entry_threshold: float = ENTRY_THRESHOLD,
    exit_threshold:  float = EXIT_THRESHOLD,
    position_size:   float = POSITION_SIZE,
) -> BacktestResult:
    """
    Replay snapshots chronologically. One position per market_id at a time.
    """
    result   = BacktestResult()
    open_pos: dict[str, Position] = {}   # market_id → Position

    for snap in snapshots:
        ts   = snap["ts"]
        spot = snap["spot"]

        for c in snap["contracts"]:
            mid       = c["market_id"]
            gap       = c.get("vol_gap")
            yes_price = c["yes_price"]
            T         = c["T_years"]
            liq       = c["liquidity"]

            if gap is None:
                continue

            # ── Check exit for open positions ──────────────────────────────
            if mid in open_pos:
                pos = open_pos[mid]
                exit_reason = None

                if abs(gap) < exit_threshold:
                    exit_reason = "gap_closed"
                elif T < T_MIN_EXIT:
                    exit_reason = "time_stop"

                if exit_reason:
                    pos.exit_ts     = ts
                    pos.exit_price  = yes_price
                    pos.exit_gap    = gap
                    pos.exit_reason = exit_reason
                    raw = _pnl(pos.side, pos.entry_price, yes_price, pos.size_usd)
                    fee = _fee(pos.side, pos.entry_price, yes_price, pos.size_usd, raw)
                    pos.pnl = raw - fee
                    result.total_pnl   += pos.pnl
                    result.total_fees  += fee
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
                continue
            if not (PRICE_MIN < yes_price < PRICE_MAX):
                continue
            if T < T_MIN_EXIT:
                continue

            side = None
            if gap < -entry_threshold:
                side = "buy_yes"    # poly IV too low → YES underpriced
            elif gap > entry_threshold:
                side = "sell_yes"   # poly IV too high → YES overpriced

            if side is None:
                continue

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

    # Force-close any positions still open at last snapshot
    # (treat as unresolved — use last known price)
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
        result.total_pnl   += pos.pnl
        result.total_fees  += fee
        result.trades.append(pos)

    return result


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(result: BacktestResult) -> None:
    trades   = result.trades
    closed   = [t for t in trades if t.exit_reason != "still_open"]
    open_pos = [t for t in trades if t.exit_reason == "still_open"]

    print(f"\n{'='*70}")
    print(f"  BACKTEST REPORT")
    print(f"{'='*70}")
    print(f"  Total trades (closed):  {len(closed)}")
    print(f"  Still open:             {len(open_pos)}")
    print(f"  Wins / Losses:          {result.win_count} / {result.loss_count}")
    if result.win_count + result.loss_count > 0:
        wr = result.win_count / (result.win_count + result.loss_count) * 100
        print(f"  Win rate:               {wr:.1f}%")
    print(f"  Total PnL:              ${result.total_pnl:+.2f}")
    print(f"  Total fees paid:        ${result.total_fees:.2f}")
    print(f"  Total slippage cost:    ${result.total_slippage:.2f}")
    print(f"  Net PnL (after fees):   ${result.total_pnl:+.2f}")
    print()

    by_reason = {}
    for t in closed:
        by_reason.setdefault(t.exit_reason, []).append(t.pnl or 0)
    for reason, pnls in sorted(by_reason.items()):
        avg = sum(pnls) / len(pnls)
        print(f"  Exit [{reason:15s}]: {len(pnls):3d} trades  avg PnL ${avg:+.2f}")

    print()
    print(f"  {'Side':10s} {'Type':20s} {'Strike':>10s} {'Entry':>7s} {'Exit':>7s} "
          f"{'Gap@E':>7s} {'Gap@X':>7s} {'PnL':>8s}  Reason")
    print(f"  {'-'*10} {'-'*20} {'-'*10} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}  ------")
    for t in sorted(trades, key=lambda x: x.entry_ts):
        pnl_s = f"${t.pnl:+.2f}" if t.pnl is not None else "  open"
        ep    = f"{t.entry_price:.3f}"
        xp    = f"{t.exit_price:.3f}" if t.exit_price is not None else "  ?"
        ge    = f"{t.entry_gap:+.1f}pp"
        gx    = f"{t.exit_gap:+.1f}pp" if t.exit_gap is not None else "    ?"
        print(f"  {t.side:10s} {t.contract_type:20s} {t.strike:>10,.0f} "
              f"{ep:>7s} {xp:>7s} {ge:>7s} {gx:>7s} {pnl_s:>8s}  {t.exit_reason or ''}")


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
    args = parser.parse_args()

    snaps = load_snapshots(Path(args.snapshot))
    if not snaps:
        print(f"No snapshots found at {args.snapshot}.")
        print("Run app.py first to collect data, then re-run this script.")
        sys.exit(0)

    span_start = snaps[0]["ts"]
    span_end   = snaps[-1]["ts"]
    print(f"Loaded {len(snaps)} snapshots  [{span_start}  →  {span_end}]")
    print(f"Entry threshold: {args.entry} pp   Exit threshold: {args.exit} pp   "
          f"Position size: ${args.size:.0f}")

    result = run_backtest(snaps, args.entry, args.exit, args.size)
    print_report(result)
