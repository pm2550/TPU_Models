#!/usr/bin/env python3
import csv
from pathlib import Path
import math

BASE = Path(__file__).resolve().parent.parent
MERGED = BASE / 'results/envelope_delta_merged.csv'
GAP = BASE / 'results/models_local_batch_usbmon/single/combined_pure_gap_seg1-8_summary.csv'
OUT = BASE / 'results/delta_vs_gap_correlation.txt'
OUT_CSV = BASE / 'results/delta_vs_gap_merged.csv'


def read_csv_map(path, key_cols):
    with path.open() as f:
        r = csv.DictReader(f)
        rows = list(r)
    m = {}
    for row in rows:
        key = tuple(row[k] for k in key_cols)
        m[key] = row
    return m, rows


def pearson(x, y):
    n = len(x)
    if n == 0:
        return float('nan')
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx)*(yi - my) for xi, yi in zip(x, y))
    denx = math.sqrt(sum((xi - mx)**2 for xi in x))
    deny = math.sqrt(sum((yi - my)**2 for yi in y))
    if denx == 0 or deny == 0:
        return float('nan')
    return num / (denx * deny)


def main():
    merged_map, merged_rows = read_csv_map(MERGED, ['model','segment'])
    gap_map, _ = read_csv_map(GAP, ['model','segment'])

    out_rows = []
    xs = []  # gap p50
    ys = []  # delta
    for key, mrow in merged_map.items():
        grow = gap_map.get(key)
        if not grow:
            continue
        try:
            delta = float(mrow['pre_delta'])
            gap_p50 = float(grow.get('p50_ms') or grow.get('mean_ms'))
        except Exception:
            continue
        out_rows.append({
            'model': key[0],
            'segment': key[1],
            'delta_ms': f"{delta:.6f}",
            'gap_p50_ms': f"{gap_p50:.6f}",
        })
        xs.append(gap_p50)
        ys.append(delta)

    r = pearson(xs, ys)

    # write merged
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model','segment','delta_ms','gap_p50_ms'])
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    with OUT.open('w') as f:
        f.write('Delta vs gap invoke time correlation\n')
        f.write(f'Rows: {len(out_rows)}\n')
        if out_rows:
            f.write(f"mean(delta_ms): {sum(ys)/len(ys):.6f}, stdev(delta_ms): {math.sqrt(sum((y - sum(ys)/len(ys))**2 for y in ys)/len(ys)):.6f}\n")
            f.write(f"mean(gap_p50_ms): {sum(xs)/len(xs):.6f}, stdev(gap_p50_ms): {math.sqrt(sum((x - sum(xs)/len(xs))**2 for x in xs)/len(xs)):.6f}\n")
        f.write(f'Pearson r (delta, gap_p50_ms): {r:.4f}\n')

if __name__ == '__main__':
    main()
