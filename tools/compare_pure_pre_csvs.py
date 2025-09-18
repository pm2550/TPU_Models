#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare two versions of single_pure_invoke_times.csv:
- current: five_models/results/single_pure_invoke_times.csv (pre uses gap+envelope adjusted)
- backup:  five_models/results/single_pure_invoke_times.csv.bak (pre uses inside pure invoke only)

Outputs:
- results/pure_pre_compare_detailed.csv with per-row values and deltas/ratios
- results/pure_pre_compare_summary.csv grouped by model and by segment
"""

import csv
from pathlib import Path

BASE = Path('/home/10210/Desktop/OS')
CUR = BASE / 'five_models/results/single_pure_invoke_times.csv'
BAK = BASE / 'five_models/results/single_pure_invoke_times.csv.bak'
OUT_DET = BASE / 'results/pure_pre_compare_detailed.csv'
OUT_SUM = BASE / 'results/pure_pre_compare_summary.csv'


def read_csv(path: Path):
    rows = []
    with path.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows


def to_float(v):
    try:
        return None if v in (None, '') else float(v)
    except Exception:
        return None


def main():
    if not CUR.exists() or not BAK.exists():
        raise SystemExit('Missing current or backup CSV')
    cur = read_csv(CUR)
    bak = read_csv(BAK)

    # index by (model,segment,type)
    idx_bak = {(r.get('model'), r.get('segment'), r.get('type')): r for r in bak}
    detailed = []
    for r in cur:
        key = (r.get('model'), r.get('segment'), r.get('type'))
        rb = idx_bak.get(key) or {}
        pre_c = to_float(r.get('pure_ms_pre'))
        pre_b = to_float(rb.get('pure_ms_pre'))
        post_c = to_float(r.get('pure_ms_post'))
        post_b = to_float(rb.get('pure_ms_post'))
        fin_c = to_float(r.get('pure_ms_final'))
        fin_b = to_float(rb.get('pure_ms_final'))
        # deltas and ratios
        pre_delta = (None if (pre_c is None or pre_b is None) else pre_c - pre_b)
        pre_ratio = (None if (pre_c is None or pre_b in (None, 0)) else pre_c / pre_b)
        post_delta = (None if (post_c is None or post_b is None) else post_c - post_b)
        fin_delta = (None if (fin_c is None or fin_b is None) else fin_c - fin_b)
        detailed.append({
            'model': key[0], 'segment': key[1], 'type': key[2],
            'cur_pre': pre_c, 'bak_pre': pre_b, 'pre_delta': pre_delta, 'pre_ratio': pre_ratio,
            'cur_post': post_c, 'bak_post': post_b, 'post_delta': post_delta,
            'cur_final': fin_c, 'bak_final': fin_b, 'final_delta': fin_delta,
        })

    # write detailed
    OUT_DET.parent.mkdir(parents=True, exist_ok=True)
    with OUT_DET.open('w', newline='') as f:
        fn = ['model','segment','type','cur_pre','bak_pre','pre_delta','pre_ratio','cur_post','bak_post','post_delta','cur_final','bak_final','final_delta']
        wr = csv.DictWriter(f, fieldnames=fn)
        wr.writeheader(); wr.writerows(detailed)

    # simple summaries: by model and by model+segment
    from collections import defaultdict
    agg_model = defaultdict(lambda: {'count':0,'pre_delta_sum':0.0,'pre_ratio_sum':0.0})
    agg_seg = defaultdict(lambda: {'count':0,'pre_delta_sum':0.0,'pre_ratio_sum':0.0})
    for d in detailed:
        if d['pre_delta'] is not None:
            k1 = d['model']
            k2 = (d['model'], d['segment'])
            agg_model[k1]['count'] += 1
            agg_model[k1]['pre_delta_sum'] += d['pre_delta']
            if d['pre_ratio'] is not None:
                agg_model[k1]['pre_ratio_sum'] += d['pre_ratio']
            agg_seg[k2]['count'] += 1
            agg_seg[k2]['pre_delta_sum'] += d['pre_delta']
            if d['pre_ratio'] is not None:
                agg_seg[k2]['pre_ratio_sum'] += d['pre_ratio']

    # write summary (two sections)
    with OUT_SUM.open('w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['by_model'])
        wr.writerow(['model','rows','pre_delta_sum','pre_ratio_avg'])
        for m, v in agg_model.items():
            avg_ratio = ('' if v['count']==0 else v['pre_ratio_sum']/v['count'])
            wr.writerow([m, v['count'], v['pre_delta_sum'], avg_ratio])
        wr.writerow([])
        wr.writerow(['by_model_segment'])
        wr.writerow(['model','segment','rows','pre_delta_sum','pre_ratio_avg'])
        for (m,s), v in agg_seg.items():
            avg_ratio = ('' if v['count']==0 else v['pre_ratio_sum']/v['count'])
            wr.writerow([m, s, v['count'], v['pre_delta_sum'], avg_ratio])

    print(f'Saved: {OUT_DET}\nSaved: {OUT_SUM}')


if __name__ == '__main__':
    main()
