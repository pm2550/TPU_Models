#!/usr/bin/env python3
"""
Compute per-invoke Bo span (first Bo S to last Bo C) and bytes for each invoke window from usbmon.txt + merged_invokes.json.
- Aligns boottime-based merged_invokes to usbmon epoch by estimating epoch_ref from earliest usb event and earliest merged begin.
- For each invoke: span = last Co within window - first Bo S within window; bytes = sum Co len= within window.
- Emits per-model arrays and overall summary. Optionally requires min-bytes.
"""
import argparse, json, re
from typing import List, Tuple, Dict, Any

MiB = 1024.0*1024.0

def parse_bo(usb_path: str) -> Tuple[List[float], List[float], List[Tuple[float,int]]]:
    s: List[float] = []
    c: List[float] = []
    co: List[Tuple[float,int]] = []
    with open(usb_path, 'r', errors='ignore') as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
            except Exception:
                continue
            dir_idx = None
            for i,p in enumerate(parts):
                if p.startswith('Bo:'):
                    dir_idx = i; break
            if dir_idx is None:
                continue
            ev = parts[dir_idx-1] if dir_idx-1>=0 else ''
            if ev == 'S':
                s.append(t)
            elif ev == 'C':
                c.append(t)
                nb = 0
                m = re.search(r"len=(\d+)", ln)
                if m:
                    nb = int(m.group(1))
                else:
                    try:
                        if dir_idx+2 < len(parts):
                            nb = int(parts[dir_idx+2])
                    except Exception:
                        nb = 0
                co.append((t, nb))
    s.sort(); c.sort(); co.sort(key=lambda x: x[0])
    return s, c, co


def percentiles(vals: List[float], ps: List[float]) -> Dict[str, float]:
    if not vals:
        return {f"p{int(p):02d}": 0.0 for p in ps}
    xs = sorted(vals); n = len(xs)
    out: Dict[str,float] = {}
    for p in ps:
        r = p/100.0*(n-1)
        i = int(r)
        f = r - i
        if i >= n-1:
            v = xs[-1]
        else:
            v = xs[i]*(1-f) + xs[i+1]*f
        out[f"p{int(p):02d}"] = v
    return out


def main():
    ap = argparse.ArgumentParser(description='Per-invoke Bo span and bytes from usbmon + merged_invokes')
    ap.add_argument('combo_dir', help='results/... directory containing usbmon.txt, time_map.json, merged_invokes.json')
    ap.add_argument('--min-bytes', type=int, default=0, help='Filter: only include invokes with bytes >= threshold')
    ap.add_argument('--emit', action='store_true', help='Emit per-invoke arrays by model')
    ap.add_argument('--pre-ms', type=float, default=0.0, help='Expand window earlier by this many milliseconds')
    ap.add_argument('--post-ms', type=float, default=0.0, help='Expand window later by this many milliseconds')
    args = ap.parse_args()

    root = args.combo_dir
    usb = root + '/usbmon.txt'
    tm = json.load(open(root + '/time_map.json'))
    merged = json.load(open(root + '/merged_invokes.json'))

    s_times, c_times, co = parse_bo(usb)
    if not s_times and not c_times:
        print(json.dumps({'error': 'no Bo events parsed'})); return

    # build epoch_estimate from earliest usb ts and earliest merged begin
    bt_ref = tm.get('boottime_ref')
    spans = merged.get('spans') or []
    if not spans:
        print(json.dumps({'error': 'no merged spans'})); return
    import math
    min_begin = min(sp['begin'] for sp in spans)
    # earliest usb ts
    with open(usb,'r',errors='ignore') as f:
        first_ts = None
        for ln in f:
            parts = ln.split()
            if not parts: continue
            try:
                first_ts = float(parts[0]); break
            except Exception:
                continue
    if first_ts is None:
        print(json.dumps({'error': 'no usb timestamps'})); return
    epoch_est = first_ts - (min_begin - bt_ref)

    # map each span to epoch timeline
    mapped = []
    for sp in spans:
        b1 = epoch_est + (sp['begin'] - bt_ref)
        e1 = epoch_est + (sp['end'] - bt_ref)
        mapped.append({'begin': b1, 'end': e1, 'label': sp.get('seg_label','seg')})

    # compute per-invoke envelope and bytes
    import bisect
    per = []
    for i,sp in enumerate(mapped):
        # apply optional expansion
        b = sp['begin'] - (args.pre_ms or 0.0)/1000.0
        e = sp['end'] + (args.post_ms or 0.0)/1000.0
        # first Bo S >= b, last Co <= e; if no S inside, fall back to completion-only span
        iS = bisect.bisect_left(s_times, b)
        fs = s_times[iS] if iS < len(s_times) and s_times[iS] <= e else None
        jC = bisect.bisect_right(c_times, e) - 1
        lc = c_times[jC] if jC >= 0 and c_times[jC] >= b else None
        span = 0.0
        if fs is not None and lc is not None and lc > fs:
            span = lc - fs
        else:
            # completion-only fallback: from first C in [b,e] to last C in [b,e]
            first_c = None
            last_c = None
            # find first completion >= b
            k0 = bisect.bisect_left(c_times, b)
            if k0 < len(c_times) and c_times[k0] <= e:
                first_c = c_times[k0]
                # last completion <= e
                k1 = bisect.bisect_right(c_times, e) - 1
                if k1 >= 0 and c_times[k1] >= b:
                    last_c = c_times[k1]
            if first_c is not None and last_c is not None and last_c > first_c:
                span = last_c - first_c
        # bytes from Co in [b,e]
        total = 0
        for t,nb in co:
            if t < b: continue
            if t > e: break
            total += int(nb or 0)
        per.append({'i': i, 'label': sp['label'], 'begin': b, 'end': e, 'span_s': span, 'bytes': total})

    # group per model, apply min-bytes and build speeds
    from collections import defaultdict
    by = defaultdict(lambda: {'bytes': [], 'spans': [], 'speeds': []})
    for r in per:
        if args.min_bytes and r['bytes'] < args.min_bytes:
            continue
        if r['span_s'] > 0 and r['bytes'] > 0:
            by[r['label']]['bytes'].append(r['bytes'])
            by[r['label']]['spans'].append(r['span_s'])
            by[r['label']]['speeds'].append((r['bytes']/MiB)/r['span_s'])
        else:
            # still emit zeros for accounting if emit
            if args.emit:
                by[r['label']]['bytes'].append(r['bytes'])
                by[r['label']]['spans'].append(r['span_s'])
                by[r['label']]['speeds'].append(0.0)

    # summarize
    ps = [50.0, 80.0, 85.0, 90.0, 95.0]
    summary: Dict[str, Any] = {}
    for label, arr in by.items():
        spd = arr['speeds']
        if not spd:
            continue
        pct = percentiles(spd, ps)
        summary[label] = {
            'count': len(spd),
            'min': min(spd),
            'max': max(spd),
            **pct,
        }
    out: Dict[str, Any] = {
        'per_model': summary,
        'total_invokes': len(spans),
    }
    if args.emit:
        out['arrays'] = {k: {'bytes': v['bytes'], 'spans_s': v['spans'], 'speeds_MiBps': v['speeds']} for k,v in by.items()}
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
