#!/usr/bin/env python3
"""
Compute per-window Bo (H2D) span and bytes by splitting usbmon.txt with >gap_ms inter-record gaps.
- Windowing: use consecutive record timestamps (any direction) and start a new window when the gap > gap_ms.
- For each window, find first Bo S and last Bo C within [wb,we], span = lastC - firstS (>=0), bytes = sum of Co len= in [wb,we].
- Emit per-window arrays and summary percentiles.
"""
import argparse, json, re
from typing import List, Tuple, Dict, Any

MiB = 1024.0*1024.0

def parse_all_timestamps(usb_path: str) -> List[float]:
    ts: List[float] = []
    with open(usb_path, 'r', errors='ignore') as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 1:
                continue
            try:
                t0 = float(parts[0])
            except Exception:
                continue
            ts.append(t0)
    return ts

def parse_bo_events(usb_path: str) -> Tuple[List[float], List[float], List[Tuple[float,int]]]:
    s_times: List[float] = []
    c_times: List[float] = []
    co_events: List[Tuple[float,int]] = []
    with open(usb_path, 'r', errors='ignore') as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 3:
                continue
            # timestamp: our capture prepends epoch in col0; use it if present else best-effort
            try:
                t0 = float(parts[0])
            except Exception:
                try:
                    t0 = float(parts[1]); t0 = (t0/1e6) if t0 > 1e6 else t0
                except Exception:
                    continue
            # find Bo token
            dir_idx = None
            for i,p in enumerate(parts):
                if p.startswith('Bo:'):
                    dir_idx = i; break
            if dir_idx is None:
                continue
            ev = parts[dir_idx-1] if dir_idx-1>=0 else ''
            if ev == 'S':
                s_times.append(t0)
            elif ev == 'C':
                c_times.append(t0)
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
                co_events.append((t0, nb))
    s_times.sort(); c_times.sort(); co_events.sort(key=lambda x: x[0])
    return s_times, c_times, co_events

def build_gap_windows(ts: List[float], gap_s: float) -> List[Tuple[float,float]]:
    if not ts: return []
    xs = sorted(ts)
    wb = xs[0]; prev = wb
    wins: List[Tuple[float,float]] = []
    for t in xs[1:]:
        if t - prev > gap_s:
            wins.append((wb, prev))
            wb = t
        prev = t
    wins.append((wb, prev))
    return wins

def envelope_for_window(s_times: List[float], c_times: List[float], co_events: List[Tuple[float,int]], wb: float, we: float) -> Tuple[float,int]:
    # first S >= wb, last C <= we
    import bisect
    fs = None; lc = None
    i = bisect.bisect_left(s_times, wb)
    if i < len(s_times) and s_times[i] <= we:
        fs = s_times[i]
    j = bisect.bisect_right(c_times, we) - 1
    if j >= 0 and c_times[j] >= wb:
        lc = c_times[j]
    span = 0.0
    if fs is not None and lc is not None and lc > fs:
        span = lc - fs
    # bytes: sum Co within window
    total = 0
    for t,nb in co_events:
        if t < wb: continue
        if t > we: break
        total += int(nb or 0)
    return span, total


def percentiles(vals: List[float], ps: List[float]) -> Dict[str,float]:
    if not vals:
        return {f"p{int(p):02d}": 0.0 for p in ps}
    xs = sorted(vals)
    out: Dict[str,float] = {}
    n = len(xs)
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
    ap = argparse.ArgumentParser(description='Bo span by >gap_ms inter-record gaps from usbmon.txt')
    ap.add_argument('usbmon_txt', help='Path to usbmon.txt (with prepend_epoch first column preferred)')
    ap.add_argument('--gap-ms', type=float, default=100.0, help='Gap threshold in milliseconds')
    ap.add_argument('--min-bytes', type=int, default=0, help='Only keep windows with Co-bytes >= threshold')
    ap.add_argument('--emit', action='store_true', help='Emit arrays of speeds/bytes/spans for inspection')
    args = ap.parse_args()

    s_times, c_times, co_events = parse_bo_events(args.usbmon_txt)
    ts_all = parse_all_timestamps(args.usbmon_txt)
    gap_s = (args.gap_ms or 100.0)/1000.0
    wins = build_gap_windows(ts_all, gap_s)

    speeds: List[float] = []
    spans: List[float] = []
    bytes_list: List[int] = []
    for (wb,we) in wins:
        sp, bt = envelope_for_window(s_times, c_times, co_events, wb, we)
        spans.append(sp)
        bytes_list.append(bt)
        if sp > 0 and bt > 0:
            speeds.append((bt/MiB)/sp)

    # Apply min-bytes filter for percentiles
    f_speeds = speeds
    if args.min_bytes and args.min_bytes > 0:
        f_speeds = [s for s,b in zip(speeds, bytes_list) if b >= args.min_bytes]

    ps = [1.0,5.0,10.0,25.0,50.0,75.0,80.0,85.0,90.0,95.0,99.0]
    pct = percentiles(f_speeds, ps)

    total_bytes = sum(bytes_list)
    total_span = sum(spans)
    out: Dict[str,Any] = {
        'count': len(wins),
        'nonzero': len([1 for s,b in zip(spans,bytes_list) if s>0 and b>0]),
        'total_bytes': total_bytes,
        'total_span_s': total_span,
        'avg_speed_MiBps': (total_bytes/MiB)/total_span if total_span>0 else 0.0,
        'min_speed_MiBps': min(f_speeds) if f_speeds else 0.0,
        'max_speed_MiBps': max(f_speeds) if f_speeds else 0.0,
    }
    out.update({k: v for k,v in pct.items()})
    if args.emit:
        out['speeds_MiBps'] = speeds
        out['spans_s'] = spans
        out['bytes'] = bytes_list
        out['gap_ms'] = args.gap_ms
        out['min_bytes'] = args.min_bytes
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
