"""
Core option pricing and IV inversion.

All formulas are under log-normal GBM with continuous risk-free rate r.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from typing import Optional


# ── One-Touch (corrected formula) ────────────────────────────────────────────

def one_touch_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Discounted one-touch (up-and-in) binary.
    Returns E^Q[e^{-r*tau} * 1_{tau <= T}] where tau = first time S_t >= K.
    """
    if S >= K:
        return 1.0
    if T <= 1e-8:
        return 0.0
    mu    = r - 0.5 * sigma ** 2
    theta = mu / sigma ** 2
    nu    = np.sqrt(theta ** 2 + 2.0 * r / sigma ** 2)
    log_KS = np.log(K / S)
    sqT    = np.sqrt(T)
    z1 = (log_KS - nu * sigma ** 2 * T) / (sigma * sqT)
    z2 = (log_KS + nu * sigma ** 2 * T) / (sigma * sqT)
    val = (K / S) ** (theta + nu) * norm.cdf(-z1) + \
          (K / S) ** (theta - nu) * norm.cdf(-z2)
    return float(np.clip(val, 0.0, 1.0))


# ── European Digital ──────────────────────────────────────────────────────────

def european_digital_price(S: float, K: float, T: float,
                            r: float, sigma: float) -> float:
    """Cash-or-nothing European digital call: pays 1 if S_T >= K."""
    if T <= 1e-8:
        return 1.0 if S >= K else 0.0
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(np.exp(-r * T) * norm.cdf(d2))


# ── IV inversions ─────────────────────────────────────────────────────────────

def _safe_brentq(f, lo: float, hi: float) -> Optional[float]:
    try:
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            return None          # no sign change → no IV in range
        return brentq(f, lo, hi, xtol=1e-6, maxiter=200)
    except Exception:
        return None


def implied_vol_european(price: float, S: float, K: float,
                          T: float, r: float,
                          sigma_lo: float = 0.01,
                          sigma_hi: float = 20.0) -> Optional[float]:
    """IV for a European digital (cash-or-nothing call)."""
    if T <= 1e-8 or price <= 0 or price >= 1:
        return None
    return _safe_brentq(
        lambda s: european_digital_price(S, K, T, r, s) - price,
        sigma_lo, sigma_hi,
    )


def implied_vol_one_touch(price: float, S: float, K: float,
                           T: float, r: float,
                           sigma_lo: float = 0.01,
                           sigma_hi: float = 20.0) -> Optional[float]:
    """IV for a one-touch up-and-in binary."""
    if T <= 1e-8 or price <= 0 or price >= 1:
        return None
    return _safe_brentq(
        lambda s: one_touch_price(S, K, T, r, s) - price,
        sigma_lo, sigma_hi,
    )
