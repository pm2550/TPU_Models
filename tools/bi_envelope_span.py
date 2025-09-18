#!/usr/bin/env python3
import os, json, sys
from typing import List, Dict, Any

MiB = 1024.0*1024.0

def load_json(p: str) -> dict:
    try:
        with open(p,'r') as f:
            return json.load(f)
    except Exception:
        return {}

def parse_usbmon_envelope_in(usb_path: str, mode: str = 'epoch'):
    # Collect first Bi S and last Ci C timestamps, plus Ci bytes per event
    s_times: List[float] = []
    c_times: List[float] = []
    ci_events: List[tuple] = []  # (ts, bytes)
    try:
        with open(usb_path,'r',errors='ignore') as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 3:
                    continue
                # choose ts column
                try:
                    time_idx = 0 if mode == 'epoch' else 1
                    ts = float(parts[time_idx])
                    if mode == 'usb' and ts > 1e6:
                        ts = ts/1e6
                except Exception:
                    continue
                # find direction token index
                dir_idx = None
                dir_tok = None
                for i, p2 in enumerate(parts):
                    if p2.startswith('Bi:') or p2.startswith('Ci:'):
                        dir_idx = i
                        dir_tok = p2[:2]
                        break
                if dir_idx is None:
                    continue
                ev = parts[dir_idx-1] if dir_idx-1 >= 0 else ''
                if dir_tok=='Bi' and ev=='S':
                    s_times.append(ts)
                elif dir_tok in ('Bi','Ci') and ev=='C':
                    c_times.append(ts)
                    # parse len=
                    import re
                    nb = 0
                    m = re.search(r"len=(\d+)", ln)
                    if m:
                        nb = int(m.group(1))
                    else:
                        try:
                            if dir_idx is not None and dir_idx+2 < len(parts):
                                nb = int(parts[dir_idx+2])
                        except Exception:
                            nb = 0
                    ci_events.append((ts, nb))
    except FileNotFoundError:
        pass
    s_times.sort(); c_times.sort(); ci_events.sort(key=lambda x: x[0])
    return s_times, c_times, ci_events

def envelope_for_window(s_times: List[float], c_times: List[float], ci_events: List[tuple], b: float, e: float) -> Dict[str, Any]:
    import bisect
    # first S >= b and <= e
    i = bisect.bisect_left(s_times, b)
    first_s = None
    if i < len(s_times) and s_times[i] <= e:
        first_s = s_times[i]
    # last C <= e and >= b
    j = bisect.bisect_right(c_times, e)-1
    last_c = None
    if j >=0 and c_times[j] >= b:
        last_c = c_times[j]
    span_s = 0.0
    if first_s is not None and last_c is not None and last_c > first_s:
        fs = max(first_s, b)
        lc = min(last_c, e)
        if lc > fs:
            span_s = lc - fs
    # bytes: sum Ci bytes where completion time within [b,e]
    bytes_sum = 0
    for ts, nb in ci_events:
        if ts < b: continue
        if ts > e: break
        bytes_sum += int(nb or 0)
    return {'span_s': span_s, 'bytes_in': bytes_sum}

def main():
    if len(sys.argv) < 2:
        print('Usage: bi_envelope_span.py <combo_K_dir>')
        sys.exit(1)
    k_dir = sys.argv[1]
    usb = os.path.join(k_dir, 'usbmon.txt')
    tm = os.path.join(k_dir, 'time_map.json')
    merged = os.path.join(k_dir, 'merged_invokes.json')
    T = load_json(tm)
    M = load_json(merged)
    if not M:
        print('missing merged_invokes')
        sys.exit(2)
    usb_ref = (T or {}).get('usbmon_ref')
    epoch_ref = (T or {}).get('epoch_ref')
    bt_ref = (T or {}).get('boottime_ref')
    mode = 'usb' if (usb_ref is not None and bt_ref is not None) else 'epoch'
    s_times, c_times, ci_events = parse_usbmon_envelope_in(usb, mode=mode)
    spans = M.get('spans') or []
    per_seg: Dict[str, List[Dict[str,Any]]] = {}
    import time as _t
    if mode == 'epoch':
        try:
            bt_now = _t.clock_gettime(_t.CLOCK_BOOTTIME)
        except Exception:
            bt_now = float(open('/proc/uptime').read().split()[0])
        epoch_now = _t.time()
        boot_epoch = epoch_now - bt_now
    for sp in spans:
        if mode == 'usb':
            b0 = sp['begin'] - bt_ref + usb_ref
            e0 = sp['end'] - bt_ref + usb_ref
        else:
            b0 = sp['begin'] + boot_epoch
            e0 = sp['end'] + boot_epoch
        seg = sp.get('seg_label','seg')
        env = envelope_for_window(s_times, c_times, ci_events, b0, e0)
        per_seg.setdefault(seg, []).append(env)
    out = {}
    for seg, xs in per_seg.items():
        if not xs:
            continue
        avg_span = sum(x['span_s'] for x in xs)/len(xs)
        avg_bytes = sum(x['bytes_in'] for x in xs)/len(xs)
        speed = (avg_bytes/MiB)/avg_span if avg_span>0 else 0.0
        out[seg] = {
            'count': len(xs),
            'avg_span_s': avg_span,
            'avg_bytes_in': int(avg_bytes),
            'speed_MiBps': speed,
        }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
