"""Quick diagnostic: why are no backtest trades firing?"""
import json
from pathlib import Path
from collections import Counter

path = Path("data/snapshots.jsonl")
records = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except:
                pass

records.sort(key=lambda r: r["ts"])
print(f"Total snapshots: {len(records)}")
if records:
    print(f"Time range: {records[0]['ts']}  ->  {records[-1]['ts']}")

# Flatten all contracts
all_contracts = [(r["ts"], r.get("spot", 0), c) for r in records for c in r.get("contracts", [])]
print(f"Total contract rows: {len(all_contracts)}")

# Check vol_gap distribution
gaps = [c["vol_gap"] for _, _, c in all_contracts if c.get("vol_gap") is not None]
print(f"Rows with vol_gap: {len(gaps)}")
if gaps:
    import statistics
    print(f"  min={min(gaps):.1f}  max={max(gaps):.1f}  mean={statistics.mean(gaps):.1f}  stdev={statistics.stdev(gaps):.1f}")
    buckets = Counter()
    for g in gaps:
        if g < -50:   buckets["< -50pp"] += 1
        elif g < -7:  buckets["-50 to -7pp (buy signal)"] += 1
        elif g < -2:  buckets["-7 to -2pp"] += 1
        elif g < 2:   buckets["-2 to 2pp"] += 1
        elif g < 7:   buckets["2 to 7pp"] += 1
        elif g < 50:  buckets["7 to 50pp (sell signal)"] += 1
        else:         buckets["> 50pp"] += 1
    for k, v in sorted(buckets.items()):
        print(f"    {k}: {v}")

# Check liquidity
liqs = [c["liquidity"] for _, _, c in all_contracts]
print(f"\nLiquidity distribution:")
print(f"  < 1k: {sum(1 for l in liqs if l < 1000)}")
print(f"  1k-20k: {sum(1 for l in liqs if 1000 <= l < 20000)}")
print(f"  >= 20k: {sum(1 for l in liqs if l >= 20000)}")

# Check yes_price range
prices = [c["yes_price"] for _, _, c in all_contracts]
print(f"\nyes_price distribution:")
print(f"  < 0.10: {sum(1 for p in prices if p < 0.10)}")
print(f"  0.10-0.90 (tradeable): {sum(1 for p in prices if 0.10 <= p <= 0.90)}")
print(f"  > 0.90: {sum(1 for p in prices if p > 0.90)}")

# Check combined filter: liq >= 20k AND price in [0.10, 0.90] AND gap is not None
tradeable = [
    (ts, spot, c) for ts, spot, c in all_contracts
    if c.get("vol_gap") is not None
    and c["liquidity"] >= 20000
    and 0.10 < c["yes_price"] < 0.90
]
print(f"\nPassing liq+price filter: {len(tradeable)}")
if tradeable:
    tgaps = [c["vol_gap"] for _, _, c in tradeable]
    print(f"  buy signals (gap < -7): {sum(1 for g in tgaps if g < -7)}")
    print(f"  sell signals (gap > 7): {sum(1 for g in tgaps if g > 7)}")
    print(f"  noise filtered (|gap| > 50): {sum(1 for g in tgaps if abs(g) > 50)}")

# Sample a few rows from the snapshot
print("\nSample contract rows:")
for ts, spot, c in all_contracts[:5]:
    print(f"  {ts} | spot={spot} | gap={c.get('vol_gap')} | liq={c['liquidity']:.0f} | price={c['yes_price']:.3f} | type={c['contract_type']}")
