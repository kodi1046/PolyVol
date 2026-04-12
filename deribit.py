"""
Deribit public API client.

Fetches:
  - BTC index price (spot)
  - Full BTC option chain with mark_iv
  - Builds a per-expiry vol smile for interpolation at arbitrary (K, T)
"""

import re
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
from scipy.stats import norm
import requests

log = logging.getLogger(__name__)

DERIBIT = "https://www.deribit.com/api/v2/public"
HEADERS  = {"User-Agent": "PolyVol/1.0"}
TIMEOUT  = 10

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

_INST_RE = re.compile(
    r"BTC-(\d{1,2})([A-Z]{3})(\d{2})-(\d+)-([CP])"
)


# ── Spot price ────────────────────────────────────────────────────────────────

_FUTURES_RE = re.compile(r"BTC-(\d{1,2})([A-Z]{3})(\d{2})$")


def get_risk_free_rate(fallback: float = 0.05) -> float:
    """
    Estimate the USD risk-free rate from the BTC futures curve.

    Uses the nearest non-perpetual BTC quarterly future:
        r = ln(F / S) / T

    Falls back to `fallback` if the request fails or no suitable future exists.
    """
    try:
        spot = get_spot()
        if not spot:
            return fallback

        r = requests.get(
            f"{DERIBIT}/get_book_summary_by_currency",
            params={"currency": "BTC", "kind": "future"},
            headers=HEADERS, timeout=TIMEOUT,
        )
        r.raise_for_status()
        futures = r.json()["result"]
    except Exception as e:
        log.warning("Futures fetch failed: %s", e)
        return fallback

    now = datetime.now(timezone.utc)
    best_rate, best_T = None, None

    for f in futures:
        name = f.get("instrument_name", "")
        m = _FUTURES_RE.match(name)
        if not m:
            continue   # skip perpetuals and weekly expiries
        day, mon, yr = m.groups()
        expiry = _parse_expiry(day, mon, yr)
        T = (expiry - now).total_seconds() / 86400 / 365.25
        if T <= 0.02:  # skip if less than ~1 week out
            continue

        mid = f.get("mid_price") or f.get("mark_price")
        if not mid or mid <= 0:
            continue

        # mid is quoted as index price (USD), not as a fraction
        implied_F = mid
        implied_r = np.log(implied_F / spot) / T

        # Take the nearest expiry to minimise noise
        if best_T is None or T < best_T:
            best_T, best_rate = T, implied_r

    if best_rate is None or not np.isfinite(best_rate):
        return fallback

    # Clamp to a plausible range (avoid aberrations during market stress)
    return float(np.clip(best_rate, 0.0, 0.15))


def get_spot() -> Optional[float]:
    try:
        r = requests.get(
            f"{DERIBIT}/get_index_price",
            params={"index_name": "btc_usd"},
            headers=HEADERS, timeout=TIMEOUT,
        )
        r.raise_for_status()
        return float(r.json()["result"]["index_price"])
    except Exception as e:
        log.warning("Spot fetch failed: %s", e)
        return None


# ── Option chain ──────────────────────────────────────────────────────────────

def _parse_expiry(day: str, mon: str, yr: str) -> datetime:
    """Deribit options expire at 08:00 UTC on the expiry date."""
    return datetime(
        2000 + int(yr), _MONTH_MAP[mon], int(day),
        8, 0, 0, tzinfo=timezone.utc,
    )


def get_option_chain() -> list[dict]:
    """
    Returns list of dicts:
        expiry      datetime UTC
        strike      float
        opt_type    'C' or 'P'
        mark_iv     float  (annualised, e.g. 0.65 for 65%)
        underlying  float  (BTC/USD at time of quote)
        instrument  str
    """
    try:
        r = requests.get(
            f"{DERIBIT}/get_book_summary_by_currency",
            params={"currency": "BTC", "kind": "option"},
            headers=HEADERS, timeout=TIMEOUT,
        )
        r.raise_for_status()
        raw = r.json()["result"]
    except Exception as e:
        log.warning("Option chain fetch failed: %s", e)
        return []

    now = datetime.now(timezone.utc)
    chain = []
    for item in raw:
        name = item.get("instrument_name", "")
        m = _INST_RE.match(name)
        if not m:
            continue
        day, mon, yr, strike_str, opt_type = m.groups()
        expiry = _parse_expiry(day, mon, yr)
        if expiry <= now:
            continue   # already expired

        iv_raw = item.get("mark_iv")
        if iv_raw is None or iv_raw <= 0:
            continue

        chain.append({
            "expiry":     expiry,
            "strike":     float(strike_str),
            "opt_type":   opt_type,
            "mark_iv":    float(iv_raw) / 100.0,   # convert % → fraction
            "underlying": float(item.get("underlying_price") or 0),
            "instrument": name,
        })

    return chain


# ── Vol surface ───────────────────────────────────────────────────────────────

class DeribitSurface:
    """
    Per-expiry vol smile interpolator built from the live Deribit option chain.

    Usage:
        surface = DeribitSurface.build()
        iv = surface.get_vol(K=72000, T=6/365)   # annualised IV at (K, T)
        iv = surface.atm_vol(T=30/365)
    """

    def __init__(self, smiles: dict[datetime, dict],
                 spot: float, built_at: datetime):
        """
        smiles: {expiry → {"strikes": [...], "ivs": [...], "underlying": float}}
        """
        self._smiles   = smiles   # keyed by Deribit expiry datetime
        self._spot     = spot
        self._built_at = built_at

    @classmethod
    def build(cls) -> "DeribitSurface":
        spot  = get_spot() or 0.0
        chain = get_option_chain()
        now   = datetime.now(timezone.utc)

        # Group by expiry, keep calls (use calls for OTM upside; puts for downside)
        by_expiry: dict[datetime, list[dict]] = {}
        for opt in chain:
            by_expiry.setdefault(opt["expiry"], []).append(opt)

        smiles: dict[datetime, dict] = {}
        for expiry, opts in by_expiry.items():
            T = (expiry - now).total_seconds() / 86400 / 365.25
            if T <= 0:
                continue

            # Separate calls and puts; for each strike take the one with better IV
            by_strike: dict[float, list[dict]] = {}
            for opt in opts:
                by_strike.setdefault(opt["strike"], []).append(opt)

            strikes, ivs = [], []
            und = spot  # fallback
            for strike, opts_at_k in sorted(by_strike.items()):
                # prefer call IV above spot, put IV below spot (OTM has tighter spread)
                candidates = opts_at_k
                if len(candidates) > 1:
                    use_type = "C" if strike >= (und or spot) else "P"
                    preferred = [o for o in candidates if o["opt_type"] == use_type]
                    candidates = preferred or candidates
                best = min(candidates, key=lambda o: abs(o["mark_iv"]))
                if best["mark_iv"] > 0:
                    strikes.append(strike)
                    ivs.append(best["mark_iv"])
                    und = best.get("underlying") or und

            if len(strikes) >= 2:
                smiles[expiry] = {
                    "strikes":    np.array(strikes, dtype=float),
                    "ivs":        np.array(ivs, dtype=float),
                    "underlying": und,
                    "T":          T,
                }

        return cls(smiles, spot, now)

    # ── Public query methods ──────────────────────────────────────────────────

    @property
    def expiries(self) -> list[datetime]:
        return sorted(self._smiles.keys())

    @property
    def spot(self) -> float:
        return self._spot

    def get_vol(self, K: float, T: float,
                r: float = 0.05) -> Optional[float]:
        """
        Interpolate/extrapolate vol at strike K, time-to-expiry T (years).
        Uses the nearest Deribit expiry by T; interpolates the smile linearly.
        """
        now = datetime.now(timezone.utc)
        best_exp = self._nearest_expiry(T, now)
        if best_exp is None:
            return None
        return self._smile_vol(best_exp, K, r)

    def smile_for_expiry(self, expiry: datetime,
                          r: float = 0.05) -> Optional[dict]:
        """Return full smile data for the Deribit expiry closest to `expiry`."""
        now = datetime.now(timezone.utc)
        T = max((expiry - now).total_seconds() / 86400 / 365.25, 1e-6)
        best_exp = self._nearest_expiry(T, now)
        if best_exp is None:
            return None
        sm = self._smiles[best_exp]
        return {
            "deribit_expiry": best_exp,
            "strikes": sm["strikes"].tolist(),
            "ivs":     sm["ivs"].tolist(),
            "T":       sm["T"],
        }

    def atm_vol(self, T: float, r: float = 0.05) -> Optional[float]:
        """Return ATM vol at given T."""
        return self.get_vol(self._spot, T, r)

    def atm_vol_30d(self, r: float = 0.05) -> Optional[float]:
        return self.atm_vol(30.0 / 365.25, r)

    def atm_term_structure(self, r: float = 0.05) -> dict:
        """ATM vol at 1d, 7d, 30d from the smile — no extra API calls."""
        return {
            "1d":  self.atm_vol(1.0  / 365.25, r),
            "7d":  self.atm_vol(7.0  / 365.25, r),
            "30d": self.atm_vol(30.0 / 365.25, r),
        }

    def digital_fair_value(self, K: float, T: float,
                            r: float = 0.05,
                            eps: float = 250.0) -> Optional[float]:
        """
        Synthetic European digital call fair value from the Deribit smile,
        using a call spread:

            p = [C(K - eps) - C(K + eps)] / (2 * eps)

        where C(K') is the Black-Scholes call price evaluated at the Deribit
        smile vol at K'.  This incorporates the vol slope (skew) correction
        that vanilla IV interpolation ignores:

            p_digital = e^{-rT} N(d2) - e^{-rT} n(d2) sqrt(T) * dσ/dK

        Using a finite difference instead of the derivative avoids having to
        differentiate the smile explicitly and is robust to linear interpolation.

        eps = 250 is a good default for BTC (strikes spaced ~500–1000 apart).
        """
        if T <= 1e-8:
            return None
        now = datetime.now(timezone.utc)
        exp = self._nearest_expiry(T, now)
        if exp is None:
            return None

        sigma_lo = self._smile_vol(exp, K - eps)
        sigma_hi = self._smile_vol(exp, K + eps)
        if sigma_lo is None or sigma_hi is None:
            return None

        c_lo = self._bs_call(self._spot, K - eps, T, r, sigma_lo)
        c_hi = self._bs_call(self._spot, K + eps, T, r, sigma_hi)
        fair = (c_lo - c_hi) / (2.0 * eps)
        return float(np.clip(fair, 0.0, 1.0))

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes vanilla call price."""
        if T <= 1e-8:
            return max(S - K, 0.0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    def _nearest_expiry(self, T: float, now: datetime) -> Optional[datetime]:
        if not self._smiles:
            return None
        target = now + timedelta(days=T * 365.25)
        return min(self._smiles.keys(), key=lambda e: abs((e - target).total_seconds()))

    def _smile_vol(self, expiry: datetime, K: float,
                   r: float = 0.05) -> Optional[float]:
        sm = self._smiles.get(expiry)
        if sm is None or len(sm["strikes"]) < 2:
            return None
        # Linear interpolation in strike space; flat extrapolation outside range
        vol = float(np.interp(K, sm["strikes"], sm["ivs"],
                               left=sm["ivs"][0], right=sm["ivs"][-1]))
        return vol if vol > 0 else None

    # ── Serialise for API ─────────────────────────────────────────────────────

    def to_api(self) -> list[dict]:
        """Return list of per-expiry smile dicts for the frontend."""
        now = datetime.now(timezone.utc)
        result = []
        for expiry in self.expiries:
            sm = self._smiles[expiry]
            T  = sm["T"]
            result.append({
                "expiry":  expiry.isoformat(),
                "label":   expiry.strftime("%d %b %H:%M UTC"),
                "T_years": round(T, 6),
                "strikes": sm["strikes"].tolist(),
                "ivs":     [round(v * 100, 2) for v in sm["ivs"]],  # → % for frontend
            })
        return result
