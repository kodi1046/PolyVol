"""
Polymarket BTC price market fetcher.

Architecture:
  Two-phase design for low-latency price updates:

  Phase 1 — Discovery (run every ~3 min):
    Finds all active BTC event IDs using slug patterns and a recent-events scan.
    Results are cached in _event_cache.

  Phase 2 — Price refresh (run every ~3 s):
    Fetches current prices for all known event IDs concurrently via ThreadPoolExecutor.
    Only needs N concurrent HTTP calls where N = number of known events (~13 today).

Discovery strategy:
  Euro Digital (daily, multi-day):
    Slugs: bitcoin-above-on-{month}-{day}  for today + next 14 days
  Euro Digital (hourly, same-day):
    Scan recent high-ID events (created today) for "bitcoin above ___" events
  One-Touch (daily):
    Slug: what-price-will-bitcoin-hit-on-{month}-{day}
  One-Touch (monthly):
    Slug: what-price-will-bitcoin-hit-in-{month}-{year}  (current + next month)

Returns structured PolyContract objects ready for the dashboard.
"""

import re
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

log = logging.getLogger(__name__)

GAMMA_API  = "https://gamma-api.polymarket.com"
HEADERS    = {"User-Agent": "PolyVol/1.0"}
TIMEOUT    = 10
MAX_WORKERS = 20   # concurrent HTTP connections for price refresh

_MONTHS = [
    "january","february","march","april","may","june",
    "july","august","september","october","november","december",
]


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PolyContract:
    event_id:      str
    market_id:     str
    question:      str
    strike:        float
    direction:     str          # "above" | "below"
    contract_type: str          # "european_digital" | "one_touch"
    expiry:        datetime     # UTC
    yes_price:     float
    liquidity:     float = 0.0
    event_title:   str   = ""
    slug:          str   = ""   # market slug → polymarket.com/event/{slug}


# ── Event ID cache ────────────────────────────────────────────────────────────

# {event_id: contract_type}  — populated by discover(), consumed by refresh()
_event_cache: dict[str, str] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_iso(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _parse_yes_price(outcome_prices_str: str) -> Optional[float]:
    try:
        prices = json.loads(outcome_prices_str)
        v = prices[0]
        return float(v) if v is not None else None
    except Exception:
        return None


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    # Increase pool size to match concurrent workers (avoids "pool is full" warnings)
    adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=MAX_WORKERS)
    s.mount("https://", adapter)
    return s


def _slug_date_strs(today: datetime, n_days: int = 14) -> list[tuple[str, str]]:
    pairs = []
    for delta in range(n_days + 1):
        d = today + timedelta(days=delta)
        pairs.append((_MONTHS[d.month - 1], str(d.day)))
    return pairs


# ── Single-event fetchers ─────────────────────────────────────────────────────

def _fetch_event_by_slug(session: requests.Session, slug: str) -> Optional[dict]:
    try:
        r = session.get(
            f"{GAMMA_API}/events",
            params={"slug": slug, "active": "true"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        return data[0] if data else None
    except Exception as e:
        log.debug("Slug fetch failed (%s): %s", slug, e)
        return None


def _fetch_event_by_id(session: requests.Session, event_id: str) -> Optional[dict]:
    try:
        r = session.get(f"{GAMMA_API}/events/{event_id}", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug("Event fetch failed (%s): %s", event_id, e)
        return None


def _fetch_recent_events(session: requests.Session, n_pages: int = 4) -> list[dict]:
    """Scan the last n_pages×200 events for same-day hourly BTC events."""
    try:
        r = session.get(
            f"{GAMMA_API}/events",
            params={"active": "true", "closed": "false", "limit": 1},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        total = int(r.headers.get("x-total-count", 0))
    except Exception:
        total = 0

    if total == 0:
        total = 11_200

    start = max(0, total - n_pages * 200)
    events: list[dict] = []
    offset = start
    while offset < total + 200:
        try:
            r = session.get(
                f"{GAMMA_API}/events",
                params={"active": "true", "closed": "false",
                        "limit": 200, "offset": offset},
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            batch = r.json()
        except Exception as e:
            log.warning("Recent events fetch failed at offset=%d: %s", offset, e)
            break
        if not batch:
            break
        events.extend(batch)
        if len(batch) < 200:
            break
        offset += 200

    return events


# ── Market parsers ────────────────────────────────────────────────────────────

def _parse_euro_market(m: dict, expiry: datetime,
                       event_id: str, event_title: str,
                       event_slug: str = ""
                       ) -> Optional[PolyContract]:
    q = m.get("question", "")
    q_lower = q.lower()

    if "above" in q_lower:
        direction = "above"
        hit = re.search(r"above\s+\$?\s*([\d,]+(?:\.\d+)?)", q, re.IGNORECASE)
    elif "below" in q_lower:
        direction = "below"
        hit = re.search(r"below\s+\$?\s*([\d,]+(?:\.\d+)?)", q, re.IGNORECASE)
    else:
        return None

    if not hit:
        return None
    strike = float(hit.group(1).replace(",", ""))

    yes_price = _parse_yes_price(m.get("outcomePrices", "[]"))
    if yes_price is None:
        return None

    return PolyContract(
        event_id=event_id,
        market_id=str(m.get("id", "")),
        question=q,
        strike=strike,
        direction=direction,
        contract_type="european_digital",
        expiry=expiry,
        yes_price=yes_price,
        liquidity=float(m.get("liquidityNum") or m.get("liquidity") or 0),
        event_title=event_title,
        slug=event_slug,
    )


def _parse_ot_market(m: dict, expiry: datetime,
                     event_id: str, event_title: str,
                     event_slug: str = ""
                     ) -> Optional[PolyContract]:
    q = m.get("question", "")
    q_lower = q.lower()

    if "reach" in q_lower or ("hit" in q_lower and "bitcoin" in q_lower.split("hit")[0]):
        direction = "above"
    elif "dip" in q_lower or "drop" in q_lower or "fall" in q_lower:
        direction = "below"
    else:
        return None

    hit = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", q)
    if not hit:
        return None
    strike = float(hit.group(1).replace(",", ""))

    yes_price = _parse_yes_price(m.get("outcomePrices", "[]"))
    if yes_price is None:
        return None

    return PolyContract(
        event_id=event_id,
        market_id=str(m.get("id", "")),
        question=q,
        strike=strike,
        direction=direction,
        contract_type="one_touch",
        expiry=expiry,
        yes_price=yes_price,
        liquidity=float(m.get("liquidityNum") or m.get("liquidity") or 0),
        event_title=event_title,
        slug=event_slug,
    )


def _parse_event(event: dict, parser) -> list[PolyContract]:
    expiry = _parse_iso(event.get("endDate", ""))
    if expiry is None:
        return []
    event_id    = str(event.get("id", ""))
    event_title = event.get("title", "")
    event_slug  = str(event.get("slug", ""))
    out = []
    for m in event.get("markets", []):
        c = parser(m, expiry, event_id, event_title, event_slug)
        if c:
            out.append(c)
    return out


# ── Phase 1: Discovery ────────────────────────────────────────────────────────

def discover(session: Optional[requests.Session] = None) -> None:
    """
    Scan for all active BTC event IDs and populate _event_cache.
    Run infrequently (every 3–5 min). Thread-safe via dict update atomicity.
    """
    global _event_cache
    s   = session or _make_session()
    now = datetime.now(timezone.utc)
    found: dict[str, str] = {}

    # Euro Digital daily slugs (today + 14 days)
    for month, day in _slug_date_strs(now, n_days=14):
        slug  = f"bitcoin-above-on-{month}-{day}"
        event = _fetch_event_by_slug(s, slug)
        if event:
            found[str(event.get("id", ""))] = "european_digital"

    # Euro Digital same-day hourly (recent events scan)
    for event in _fetch_recent_events(s, n_pages=4):
        if "bitcoin above" not in event.get("title", "").lower():
            continue
        found[str(event.get("id", ""))] = "european_digital"

    # One-Touch daily
    daily_slug = f"what-price-will-bitcoin-hit-on-{_MONTHS[now.month-1]}-{now.day}"
    event = _fetch_event_by_slug(s, daily_slug)
    if event:
        found[str(event.get("id", ""))] = "one_touch"

    # One-Touch monthly (current + next month)
    for offset in [0, 1]:
        m_date = (now.replace(day=1) + timedelta(days=32 * offset)).replace(day=1)
        slug   = f"what-price-will-bitcoin-hit-in-{_MONTHS[m_date.month-1]}-{m_date.year}"
        event  = _fetch_event_by_slug(s, slug)
        if event:
            found[str(event.get("id", ""))] = "one_touch"

    _event_cache = found
    log.info("Discovery: %d events cached (%d ED, %d OT)",
             len(found),
             sum(1 for v in found.values() if v == "european_digital"),
             sum(1 for v in found.values() if v == "one_touch"))


# ── Phase 2: Price refresh ────────────────────────────────────────────────────

def refresh(session: Optional[requests.Session] = None
            ) -> dict[str, list[PolyContract]]:
    """
    Fetch current prices for all events in _event_cache concurrently.
    Should be called every 3–5 seconds. Falls back to full fetch if cache empty.
    """
    if not _event_cache:
        return fetch_all(session)

    s = session or _make_session()

    parsers = {
        "european_digital": _parse_euro_market,
        "one_touch":        _parse_ot_market,
    }

    euro_contracts: list[PolyContract] = []
    ot_contracts:   list[PolyContract] = []

    def _fetch_one(event_id: str, contract_type: str):
        event = _fetch_event_by_id(s, event_id)
        if not event:
            return [], contract_type
        return _parse_event(event, parsers[contract_type]), contract_type

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one, eid, ctype): eid
            for eid, ctype in _event_cache.items()
        }
        for future in as_completed(futures):
            try:
                contracts, ctype = future.result()
                if ctype == "european_digital":
                    euro_contracts.extend(contracts)
                else:
                    ot_contracts.extend(contracts)
            except Exception as e:
                log.warning("Price refresh future failed: %s", e)

    log.debug("Refresh: %d ED + %d OT contracts",
              len(euro_contracts), len(ot_contracts))
    return {"european_digital": euro_contracts, "one_touch": ot_contracts}


# ── Legacy full fetch (used for initial load / cache miss) ────────────────────

def fetch_all(session: Optional[requests.Session] = None
              ) -> dict[str, list[PolyContract]]:
    """
    Full discovery + price fetch in one pass. Populates _event_cache as a side effect.
    Use for initial load; prefer discover() + refresh() for ongoing updates.
    """
    s    = session or _make_session()
    now  = datetime.now(timezone.utc)
    seen_event_ids: set[str] = set()

    euro_contracts: list[PolyContract] = []
    ot_contracts:   list[PolyContract] = []

    # Euro Digital: slug-based for next 14 days
    for month, day in _slug_date_strs(now, n_days=14):
        slug = f"bitcoin-above-on-{month}-{day}"
        event = _fetch_event_by_slug(s, slug)
        if event:
            eid = str(event.get("id", ""))
            if eid not in seen_event_ids:
                seen_event_ids.add(eid)
                if not event.get("markets"):
                    event = _fetch_event_by_id(s, eid) or event
                euro_contracts.extend(_parse_event(event, _parse_euro_market))

    # Euro Digital: recent-events scan for same-day hourly
    recent = _fetch_recent_events(s, n_pages=4)
    for event in recent:
        if "bitcoin above" not in event.get("title", "").lower():
            continue
        eid = str(event.get("id", ""))
        if eid in seen_event_ids:
            continue
        seen_event_ids.add(eid)
        if not event.get("markets"):
            event = _fetch_event_by_id(s, eid) or event
        euro_contracts.extend(_parse_event(event, _parse_euro_market))

    # One-Touch daily
    daily_ot_slug = f"what-price-will-bitcoin-hit-on-{_MONTHS[now.month-1]}-{now.day}"
    event = _fetch_event_by_slug(s, daily_ot_slug)
    if event:
        eid = str(event.get("id", ""))
        if eid not in seen_event_ids:
            seen_event_ids.add(eid)
            if not event.get("markets"):
                event = _fetch_event_by_id(s, eid) or event
            ot_contracts.extend(_parse_event(event, _parse_ot_market))

    # One-Touch monthly
    for month_offset in [0, 1]:
        m_date   = (now.replace(day=1) + timedelta(days=32 * month_offset)).replace(day=1)
        month_slug = f"what-price-will-bitcoin-hit-in-{_MONTHS[m_date.month-1]}-{m_date.year}"
        event = _fetch_event_by_slug(s, month_slug)
        if event:
            eid = str(event.get("id", ""))
            if eid not in seen_event_ids:
                seen_event_ids.add(eid)
                if not event.get("markets"):
                    event = _fetch_event_by_id(s, eid) or event
                ot_contracts.extend(_parse_event(event, _parse_ot_market))

    # Populate cache
    global _event_cache
    for c in euro_contracts:
        _event_cache[c.event_id] = "european_digital"
    for c in ot_contracts:
        _event_cache[c.event_id] = "one_touch"

    log.info("fetch_all: %d ED + %d OT contracts (%d events)",
             len(euro_contracts), len(ot_contracts), len(seen_event_ids))

    return {"european_digital": euro_contracts, "one_touch": ot_contracts}


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    print("Phase 1: initial fetch_all (populates cache)…")
    t0 = time.perf_counter()
    data = fetch_all()
    t1 = time.perf_counter()
    print(f"  fetch_all: {t1-t0:.2f}s  →  {len(data['european_digital'])} ED, {len(data['one_touch'])} OT")

    print("\nPhase 2: concurrent refresh (3 rounds)…")
    for i in range(3):
        t0 = time.perf_counter()
        data = refresh()
        t1 = time.perf_counter()
        print(f"  refresh #{i+1}: {t1-t0:.2f}s  →  {len(data['european_digital'])} ED, {len(data['one_touch'])} OT")
        time.sleep(1)
