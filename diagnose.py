"""
Diagnostic script for anomalous vol gap signals.
Prints all contracts where |vol_gap| > 50pp, with full context.

Usage:
    .venv/bin/python3 diagnose.py
"""

import warnings
import json
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

from fetcher   import fetch_all
from deribit   import DeribitSurface
from vol_math  import implied_vol_european, implied_vol_one_touch

RISK_FREE_RATE    = 0.05
ANOMALY_THRESHOLD = 50.0   # pp — anything above this gets flagged

# Must match app.py
PRICE_MIN  = 0.03
PRICE_MAX  = 0.97
T_MIN      = 2.0 / 365.25 / 24   # 2 hours
MIN_IV_LIQ = 1_000
MAX_IV     = 2.50


def main():
    print("Fetching Deribit surface…")
    surface = DeribitSurface.build()
    spot    = surface.spot
    now     = datetime.now(timezone.utc)
    print(f"Spot: ${spot:,.0f}  at {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Deribit 30d ATM: {surface.atm_vol_30d()*100:.2f}%")

    print("\nFetching Polymarket contracts…")
    data = fetch_all()
    print(f"  Euro Digital: {len(data['european_digital'])} contracts")
    print(f"  One-Touch:    {len(data['one_touch'])} contracts")

    pairs = [
        ("european_digital", data["european_digital"], implied_vol_european),
        ("one_touch",        data["one_touch"],        implied_vol_one_touch),
    ]

    anomalies = []
    all_rows  = []

    for ct, contracts, iv_fn in pairs:
        for c in contracts:
            T   = (c.expiry - now).total_seconds() / 86400 / 365.25
            T_h = T * 365.25 * 24

            # Compute IV regardless of filters (to show what the raw inversion gives)
            iv_raw = iv_fn(c.yes_price, spot, c.strike, T, RISK_FREE_RATE) if T > 1e-8 else None
            iv_d   = surface.get_vol(c.strike, max(T, 1e-6)) if T > -1 else None

            # Filtered IV (what app.py computes — must stay in sync with app.py)
            price_ok = PRICE_MIN < c.yes_price < PRICE_MAX
            t_ok     = T > T_MIN
            liq_ok   = c.liquidity >= MIN_IV_LIQ
            iv_filt  = None
            if price_ok and t_ok and liq_ok and T > 1e-8:
                _iv = iv_fn(c.yes_price, spot, c.strike, T, RISK_FREE_RATE)
                if _iv is not None and _iv <= MAX_IV:
                    iv_filt = _iv

            gap_raw  = (iv_raw  - iv_d) * 100 if (iv_raw  and iv_d) else None
            gap_filt = (iv_filt - iv_d) * 100 if (iv_filt and iv_d) else None

            row = {
                "type":      ct,
                "question":  c.question,
                "strike":    c.strike,
                "expiry":    c.expiry.strftime("%Y-%m-%d %H:%M UTC"),
                "T_hours":   round(T_h, 2),
                "yes_price": c.yes_price,
                "liquidity": c.liquidity,
                "iv_raw":    round(iv_raw  * 100, 1) if iv_raw  else None,
                "iv_filt":   round(iv_filt * 100, 1) if iv_filt else None,
                "iv_deribit":round(iv_d    * 100, 1) if iv_d    else None,
                "gap_raw":   round(gap_raw,  1)      if gap_raw  is not None else None,
                "gap_filt":  round(gap_filt, 1)      if gap_filt is not None else None,
                "price_ok":  price_ok,
                "t_ok":      t_ok,
                "liq_ok":    liq_ok,
            }
            all_rows.append(row)

            if gap_filt is not None and abs(gap_filt) > ANOMALY_THRESHOLD:
                anomalies.append(row)

    # ── Anomaly report ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ANOMALIES  (|filtered vol_gap| > {ANOMALY_THRESHOLD}pp)")
    print(f"  {len(anomalies)} found")
    print(f"{'='*70}")

    if not anomalies:
        print("  None — no anomalies with current filters.")
    else:
        for a in sorted(anomalies, key=lambda x: abs(x["gap_filt"] or 0), reverse=True):
            print(f"\n  {a['type'].upper():20s}  K=${a['strike']:>10,.0f}")
            print(f"  Question:   {a['question']}")
            print(f"  Expiry:     {a['expiry']}  (T={a['T_hours']:.1f}h)")
            print(f"  YES price:  {a['yes_price']:.4f}  "
                  f"(price_ok={a['price_ok']}, t_ok={a['t_ok']}, liq_ok={a['liq_ok']})")
            print(f"  Liquidity:  ${a['liquidity']:,.0f}")
            print(f"  IV poly (filtered):  {a['iv_filt']}%")
            print(f"  IV poly (raw/no filter): {a['iv_raw']}%")
            print(f"  IV deribit:  {a['iv_deribit']}%")
            print(f"  Gap (filtered): {a['gap_filt']:+.1f}pp")

    # ── Full table ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  FULL TABLE  (all contracts with valid filtered IV)")
    print(f"{'='*70}")
    valid = [r for r in all_rows if r["gap_filt"] is not None]
    valid.sort(key=lambda x: abs(x["gap_filt"] or 0), reverse=True)

    print(f"\n  {'Type':20s} {'Strike':>10s} {'T(h)':>6s} {'YES':>6s} "
          f"{'IV_poly':>8s} {'IV_drbt':>8s} {'Gap':>8s}  {'Liq':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8}  {'-'*10}")
    for r in valid:
        print(f"  {r['type']:20s} {r['strike']:>10,.0f} {r['T_hours']:>6.1f} "
              f"{r['yes_price']:>6.3f} {r['iv_filt']:>8.1f}% {r['iv_deribit']:>8.1f}% "
              f"{r['gap_filt']:>+8.1f}pp  ${r['liquidity']:>9,.0f}")

    # ── Save to file ──────────────────────────────────────────────────────────
    out = {
        "timestamp":  now.isoformat(),
        "spot":       spot,
        "n_anomalies": len(anomalies),
        "anomalies":  anomalies,
        "all":        all_rows,
    }
    with open("diagnose_dump.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull dump written to diagnose_dump.json")


if __name__ == "__main__":
    main()
