#!/usr/bin/env python3
import sys, os, json, csv

def main():
    if len(sys.argv) < 3:
        print('Usage: export_bo_summary_csv.py <bo_envelope_summary.json> <out.csv>')
        sys.exit(1)
    src, outp = sys.argv[1:3]
    j = json.load(open(src))
    rows = []
    for seg, m in (j or {}).items():
        rows.append({
            'model': seg,
            'count': int(m.get('count') or 0),
            'avg_span_s': float(m.get('avg_span_s') or 0.0),
            'avg_bytes_out': int(m.get('avg_bytes_out') or 0),
            'speed_MiBps': float(m.get('speed_MiBps') or 0.0),
        })
    with open(outp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model','count','avg_span_s','avg_bytes_out','speed_MiBps'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'Wrote CSV: {outp} ({len(rows)} rows)')

if __name__ == '__main__':
    main()
