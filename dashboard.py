"""
Live Polymarket BTC Price Dashboard
=====================================
Continuously fetches all BTC price markets from Polymarket and plots
YES prices vs strike, grouped by expiry and contract type.

  - Top panel:  European Digital ("Bitcoin above $K on [date]")
  - Bottom panel: One-Touch       ("Will Bitcoin reach/dip to $K")

Run:
    .venv/bin/python3 dashboard.py
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")           # works on most Linux/Mac/Windows desktops

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as mticker
import numpy as np

from fetcher import fetch_all, PolyContract

# ── Config ────────────────────────────────────────────────────────────────────

REFRESH_INTERVAL_S = 10     # how often to re-fetch from Polymarket
MIN_LIQUIDITY      = 0      # filter out markets below this USD liquidity
PLOT_STYLE         = "dark_background"

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# ── Shared state updated by background thread ─────────────────────────────────

_lock    = threading.Lock()
_data: dict[str, list[PolyContract]] = {"european_digital": [], "one_touch": []}
_last_update: Optional[datetime] = None
_fetch_error: Optional[str] = None

def _fetch_loop() -> None:
    global _last_update, _fetch_error
    while True:
        try:
            result = fetch_all()
            with _lock:
                _data["european_digital"] = result["european_digital"]
                _data["one_touch"]        = result["one_touch"]
                _last_update = datetime.now(timezone.utc)
                _fetch_error = None
            log.info("Refreshed: %d euro-digital, %d one-touch",
                     len(result["european_digital"]), len(result["one_touch"]))
        except Exception as e:
            with _lock:
                _fetch_error = str(e)
            log.error("Fetch failed: %s", e)

        time.sleep(REFRESH_INTERVAL_S)


# ── Colour palette for expiry series ─────────────────────────────────────────

_PALETTE = [
    "#00d4ff", "#ff6b35", "#7fff00", "#ff00ff",
    "#ffd700", "#00ffcc", "#ff4444", "#aa88ff",
    "#ff8800", "#00ff88",
]


def _expiry_label(expiry: datetime) -> str:
    """Short human-readable expiry label."""
    now = datetime.now(timezone.utc)
    delta = expiry - now
    hours = delta.total_seconds() / 3600
    if hours < 2:
        return expiry.strftime("%H:%M UTC")
    if hours < 36:
        return expiry.strftime("%b %d %H:%M UTC")
    return expiry.strftime("%b %d UTC")


def _group_by_expiry(contracts: list[PolyContract]
                     ) -> dict[datetime, list[PolyContract]]:
    groups: dict[datetime, list[PolyContract]] = {}
    for c in contracts:
        groups.setdefault(c.expiry, []).append(c)
    return groups


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _draw_panel(ax: plt.Axes,
                contracts: list[PolyContract],
                title: str,
                now: datetime) -> None:
    ax.cla()
    ax.set_facecolor("#0d1117")
    ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.set_xlabel("Strike  (BTC / USD)", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("YES Price  (probability)", color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.grid(True, color="#1e2530", linewidth=0.6, linestyle="--")

    if not contracts:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="#666666", fontsize=14)
        return

    # Filter by liquidity
    contracts = [c for c in contracts if c.liquidity >= MIN_LIQUIDITY]

    groups = _group_by_expiry(contracts)
    # Sort expiries (soonest first → most vivid colour = soonest)
    sorted_expiries = sorted(groups.keys())

    for i, expiry in enumerate(sorted_expiries):
        cs = sorted(groups[expiry], key=lambda c: c.strike)
        if not cs:
            continue

        strikes  = np.array([c.strike    for c in cs])
        yes_prices = np.array([c.yes_price for c in cs])
        color    = _PALETTE[i % len(_PALETTE)]
        label    = _expiry_label(expiry)

        # Separate above / below for one-touch (below is a down-touch)
        has_below = any(c.direction == "below" for c in cs)

        if has_below:
            # Down-touch contracts: plot with dashed style
            below_cs = [c for c in cs if c.direction == "below"]
            above_cs = [c for c in cs if c.direction == "above"]

            if below_cs:
                bs = sorted(below_cs, key=lambda c: c.strike)
                ax.plot(
                    [c.strike for c in bs],
                    [c.yes_price for c in bs],
                    "o--", color=color, linewidth=1.2, markersize=4,
                    label=f"{label} (down-touch)",
                )
            if above_cs:
                ab = sorted(above_cs, key=lambda c: c.strike)
                ax.plot(
                    [c.strike for c in ab],
                    [c.yes_price for c in ab],
                    "o-", color=color, linewidth=1.5, markersize=5,
                    label=f"{label} (up-touch)",
                )
        else:
            ax.plot(strikes, yes_prices, "o-", color=color,
                    linewidth=1.5, markersize=5, label=label)

        # Mark the 50% point (ATM proxy)
        above_50 = [(s, p) for s, p in zip(strikes, yes_prices) if p >= 0.50]
        below_50 = [(s, p) for s, p in zip(strikes, yes_prices) if p < 0.50]
        if above_50 and below_50:
            atm_approx = (above_50[-1][0] + below_50[0][0]) / 2
            ax.axvline(atm_approx, color=color, linewidth=0.5,
                       linestyle=":", alpha=0.4)

    ax.legend(loc="upper right", fontsize=7, framealpha=0.3,
              labelcolor="white", facecolor="#0d1117", edgecolor="#333333")


# ── Main dashboard ────────────────────────────────────────────────────────────

def run_dashboard() -> None:
    plt.style.use(PLOT_STYLE)
    fig, (ax_euro, ax_ot) = plt.subplots(
        2, 1,
        figsize=(13, 9),
        facecolor="#0d1117",
    )
    fig.subplots_adjust(hspace=0.42, left=0.08, right=0.97, top=0.93, bottom=0.07)

    status_ax = fig.add_axes([0.0, 0.0, 1.0, 0.025])
    status_ax.set_axis_off()
    status_text = status_ax.text(
        0.5, 0.5, "Loading…",
        transform=status_ax.transAxes,
        ha="center", va="center",
        color="#666666", fontsize=8,
    )

    def _update(_frame: int) -> None:
        with _lock:
            euro  = list(_data["european_digital"])
            ot    = list(_data["one_touch"])
            ts    = _last_update
            err   = _fetch_error

        now = datetime.now(timezone.utc)

        _draw_panel(
            ax_euro, euro,
            "European Digital  —  Bitcoin above $K on [date]  (Polymarket YES price)",
            now,
        )
        _draw_panel(
            ax_ot, ot,
            "One-Touch  —  Will Bitcoin reach/dip to $K?  (Polymarket YES price)",
            now,
        )

        if err:
            status = f"ERROR: {err}"
            color  = "#ff4444"
        elif ts:
            age = (now - ts).total_seconds()
            status = (f"Last update: {ts.strftime('%H:%M:%S UTC')}  "
                      f"({age:.0f}s ago)  |  "
                      f"{len(euro)} euro-digital  /  {len(ot)} one-touch  "
                      f"|  refreshing every {REFRESH_INTERVAL_S}s")
            color = "#666666"
        else:
            status = "Fetching data…"
            color  = "#888888"

        status_text.set_text(status)
        status_text.set_color(color)
        fig.canvas.draw_idle()

    anim = FuncAnimation(
        fig, _update,
        interval=2_000,       # redraw every 2s (data only refreshes every 10s)
        cache_frame_data=False,
    )

    fig.suptitle(
        "Polymarket BTC Implied Volatility Dashboard",
        color="white", fontsize=13, fontweight="bold",
    )

    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Start background fetch thread
    t = threading.Thread(target=_fetch_loop, daemon=True)
    t.start()

    # Give it a moment to do the first fetch
    log.info("Waiting for initial data fetch…")
    deadline = time.time() + 60
    while time.time() < deadline:
        with _lock:
            if _data["european_digital"] or _data["one_touch"] or _fetch_error:
                break
        time.sleep(0.5)

    run_dashboard()
