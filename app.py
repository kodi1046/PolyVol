"""
PolyVol Flask dashboard.

Runs background threads that refresh Polymarket and Deribit data,
then serves a JSON API and a single-page frontend.

Usage:
    .venv/bin/python3 app.py
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from flask import Flask, jsonify, render_template

from fetcher  import fetch_all, discover, refresh, PolyContract
from deribit  import DeribitSurface, get_spot, get_risk_free_rate
from vol_math import implied_vol_european, implied_vol_one_touch
from logger   import log_snapshot

# ── Config ────────────────────────────────────────────────────────────────────

POLY_REFRESH_S      = 4     # Polymarket price refresh (fast concurrent fetch)
POLY_DISCOVER_S     = 180   # Polymarket re-discovery (slug scan + recent events)
DERIBIT_SPOT_S      = 5     # Spot price refresh — fast, single API call
DERIBIT_SURFACE_S   = 30    # Full vol surface rebuild — slower, ~880 options
RISK_FREE_RATE_FALLBACK = 0.05  # used until first Deribit rate fetch succeeds

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# ── Shared state ──────────────────────────────────────────────────────────────

_lock = threading.Lock()

_state: dict = {
    "spot":          None,          # float
    "deribit_atm30": None,          # float, 30d ATM vol
    "surface":       None,          # DeribitSurface
    "risk_free_rate": RISK_FREE_RATE_FALLBACK,  # float, from futures curve
    "poly_raw":      {"european_digital": [], "one_touch": []},
    "api_payload":   None,          # pre-built JSON dict
    "poly_updated":  None,
    "deribit_updated": None,
    "error":         None,
}


# ── Data builder ──────────────────────────────────────────────────────────────

def _contracts_to_api(contracts: list[PolyContract],
                      contract_type: str,
                      surface: Optional[DeribitSurface],
                      spot: float,
                      now: datetime,
                      r: float = RISK_FREE_RATE_FALLBACK) -> list[dict]:
    """
    Group contracts by expiry, compute IV for each, attach Deribit vol.
    Returns list of expiry-group dicts ready for JSON serialisation.
    """
    # Group by expiry
    by_expiry: dict[datetime, list[PolyContract]] = {}
    for c in contracts:
        by_expiry.setdefault(c.expiry, []).append(c)

    PRICE_MIN   = 0.03
    PRICE_MAX   = 0.97
    T_MIN       = 2.0 / 365.25 / 24   # 2 hours — below this near-ATM prices are AMM noise
    MIN_IV_LIQ  = 1_000               # skip IV if market has < $1k liquidity
    MAX_IV      = 2.50                 # 250% — anything above is not a tradeable signal

    groups = []
    for expiry in sorted(by_expiry.keys()):
        cs = sorted(by_expiry[expiry], key=lambda c: c.strike)
        T  = max((expiry - now).total_seconds() / 86400 / 365.25, 1e-6)

        # Deribit smile for this expiry (may be None if surface not yet loaded)
        deribit_smile = surface.smile_for_expiry(expiry) if surface else None

        markets = []
        for c in cs:
            # Only invert IV for liquid, non-trivial prices with enough time left
            iv_poly = None
            if (PRICE_MIN < c.yes_price < PRICE_MAX
                    and T > T_MIN
                    and c.liquidity >= MIN_IV_LIQ):
                if contract_type == "european_digital":
                    iv_raw = implied_vol_european(
                        c.yes_price, spot, c.strike, T, r
                    )
                else:
                    iv_raw = implied_vol_one_touch(
                        c.yes_price, spot, c.strike, T, r
                    )
                if iv_raw is not None and iv_raw <= MAX_IV:
                    iv_poly = iv_raw

            # Deribit reference vol — vanilla smile interpolation
            iv_deribit_vanilla: Optional[float] = None
            if surface:
                iv_deribit_vanilla = surface.get_vol(c.strike, T, r)

            # Deribit reference for European Digital: synthetic digital via
            # call spread, which incorporates the smile slope (skew correction).
            # For One-Touch we fall back to vanilla (no simple static replication).
            iv_deribit: Optional[float] = None
            if surface:
                if contract_type == "european_digital":
                    p_fair = surface.digital_fair_value(c.strike, T, r)
                    if p_fair is not None and PRICE_MIN < p_fair < PRICE_MAX:
                        iv_deribit = implied_vol_european(p_fair, spot, c.strike, T, r)
                    # fall back to vanilla if inversion fails
                    if iv_deribit is None:
                        iv_deribit = iv_deribit_vanilla
                else:
                    iv_deribit = iv_deribit_vanilla

            vol_gap: Optional[float] = None
            if iv_poly is not None and iv_deribit is not None:
                vol_gap = iv_poly - iv_deribit

            markets.append({
                "strike":              c.strike,
                "yes_price":           round(c.yes_price, 4),
                "direction":           c.direction,
                "liquidity":           round(c.liquidity, 0),
                "iv_poly":             round(iv_poly * 100, 2)           if iv_poly            is not None else None,
                "iv_deribit":          round(iv_deribit * 100, 2)        if iv_deribit         is not None else None,
                "iv_deribit_vanilla":  round(iv_deribit_vanilla * 100, 2) if iv_deribit_vanilla is not None else None,
                "vol_gap":             round(vol_gap * 100, 2)           if vol_gap            is not None else None,
                "market_id":           c.market_id,
                "question":            c.question,
                "url":                 f"https://polymarket.com/event/{c.slug}" if c.slug else "",
            })

        hours_to_expiry = (expiry - now).total_seconds() / 3600
        groups.append({
            "expiry":         expiry.isoformat(),
            "label":          _expiry_label(expiry, now),
            "T_years":        round(T, 6),
            "hours_to_expiry": round(hours_to_expiry, 1),
            "event_title":    cs[0].event_title if cs else "",
            "markets":        markets,
            "deribit_smile":  deribit_smile,
        })

    return groups


def _expiry_label(expiry: datetime, now: datetime) -> str:
    delta_h = (expiry - now).total_seconds() / 3600
    if delta_h < 2:
        return expiry.strftime("%H:%M UTC")
    if delta_h < 36:
        return expiry.strftime("%b %d %H:%M UTC")
    return expiry.strftime("%b %d UTC")


def _build_payload() -> dict:
    """Assemble the full API payload from current state."""
    with _lock:
        spot      = _state["spot"]
        atm30     = _state["deribit_atm30"]
        surface   = _state["surface"]
        r         = _state["risk_free_rate"]
        poly_raw  = dict(_state["poly_raw"])
        p_upd     = _state["poly_updated"]
        d_upd     = _state["deribit_updated"]
        error     = _state["error"]

    spot = spot or 0.0
    now  = datetime.now(timezone.utc)

    euro_groups = _contracts_to_api(
        poly_raw["european_digital"], "european_digital", surface, spot, now, r
    )
    ot_groups = _contracts_to_api(
        poly_raw["one_touch"], "one_touch", surface, spot, now, r
    )

    # Summary stats
    n_euro = sum(len(g["markets"]) for g in euro_groups)
    n_ot   = sum(len(g["markets"]) for g in ot_groups)

    # Vol gap signals — one row per individual contract that has a computed gap
    all_gaps = []
    for ct_label, groups in [("ED", euro_groups), ("OT", ot_groups)]:
        for g in groups:
            for m in g["markets"]:
                if m["vol_gap"] is not None:
                    all_gaps.append({
                        "type":      ct_label,
                        "expiry":    g["label"],
                        "strike":    m["strike"],
                        "direction": m["direction"],
                        "gap_pct":   m["vol_gap"],
                        "iv_poly":   m["iv_poly"],
                        "iv_deribit": m["iv_deribit"],
                        "yes_price": m["yes_price"],
                        "liquidity": m["liquidity"],
                        "question":  m["question"],
                        "url":       m["url"],
                    })
    all_gaps.sort(key=lambda x: abs(x["gap_pct"]), reverse=True)

    return {
        "spot":             spot,
        "deribit_atm30":    round(atm30 * 100, 2) if atm30 else None,
        "deribit_surface":  surface.to_api() if surface else [],
        "european_digital": euro_groups,
        "one_touch":        ot_groups,
        "summary": {
            "n_euro_digital":  n_euro,
            "n_one_touch":     n_ot,
            "n_deribit_expiries": len(surface.expiries) if surface else 0,
        },
        "top_signals":      all_gaps[:20],
        "poly_updated":     p_upd.isoformat() if p_upd else None,
        "deribit_updated":  d_upd.isoformat() if d_upd else None,
        "risk_free_rate":   round(r * 100, 2),
        "poly_age_s":       round((now - p_upd).total_seconds()) if p_upd else None,
        "deribit_age_s":    round((now - d_upd).total_seconds()) if d_upd else None,
        "poly_refresh_s":   POLY_REFRESH_S,
        "deribit_refresh_s": DERIBIT_SPOT_S,
        "error":            error,
        "server_time":      now.isoformat(),
    }


# ── Background threads ────────────────────────────────────────────────────────

def _poly_loop() -> None:
    last_discover = 0.0
    while True:
        try:
            now_mono = time.monotonic()
            # Re-discover event IDs periodically (or on first run)
            if now_mono - last_discover >= POLY_DISCOVER_S:
                discover()
                last_discover = now_mono

            result = refresh()
            ts = datetime.now(timezone.utc)
            with _lock:
                _state["poly_raw"]     = result
                _state["poly_updated"] = ts
                _state["error"]        = None
                snap_spot    = _state["spot"] or 0.0
                snap_surface = _state["surface"]
                snap_r       = _state["risk_free_rate"]
            log.debug("Poly refresh: %d ED + %d OT",
                      len(result["european_digital"]), len(result["one_touch"]))

            # Log snapshot for backtesting
            euro_g = _contracts_to_api(
                result["european_digital"], "european_digital",
                snap_surface, snap_spot, ts, snap_r,
            )
            ot_g = _contracts_to_api(
                result["one_touch"], "one_touch",
                snap_surface, snap_spot, ts, snap_r,
            )
            log_snapshot(ts, snap_spot, [], euro_g, ot_g)
        except Exception as e:
            with _lock:
                _state["error"] = str(e)
            log.error("Poly fetch error: %s", e)
        time.sleep(POLY_REFRESH_S)


def _deribit_spot_loop() -> None:
    """Refresh spot price every DERIBIT_SPOT_S seconds — single lightweight API call."""
    while True:
        try:
            spot = get_spot()
            if spot:
                with _lock:
                    _state["spot"]            = spot
                    _state["deribit_updated"] = datetime.now(timezone.utc)
                log.debug("Deribit spot: $%.0f", spot)
        except Exception as e:
            log.error("Deribit spot fetch error: %s", e)
        time.sleep(DERIBIT_SPOT_S)


def _deribit_surface_loop() -> None:
    """Rebuild full vol surface every DERIBIT_SURFACE_S seconds."""
    while True:
        try:
            r       = get_risk_free_rate(fallback=RISK_FREE_RATE_FALLBACK)
            surface = DeribitSurface.build()
            atm30   = surface.atm_vol_30d(r)
            with _lock:
                _state["spot"]            = surface.spot
                _state["surface"]         = surface
                _state["deribit_atm30"]   = atm30
                _state["risk_free_rate"]  = r
                _state["deribit_updated"] = datetime.now(timezone.utc)
            log.info("Deribit surface: spot=$%.0f, 30d ATM=%.1f%%, r=%.2f%%, %d expiries",
                     surface.spot, (atm30 or 0) * 100, r * 100, len(surface.expiries))
        except Exception as e:
            log.error("Deribit surface fetch error: %s", e)
        time.sleep(DERIBIT_SURFACE_S)


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    payload = _build_payload()
    return jsonify(payload)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Start background fetch threads
    threading.Thread(target=_deribit_surface_loop, daemon=True).start()
    threading.Thread(target=_deribit_spot_loop,    daemon=True).start()
    threading.Thread(target=_poly_loop,            daemon=True).start()

    # Give Deribit a head start (spot price needed for IV inversion)
    log.info("Waiting for initial Deribit fetch…")
    for _ in range(30):
        with _lock:
            if _state["spot"]:
                break
        time.sleep(1)

    log.info("Starting Flask on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
