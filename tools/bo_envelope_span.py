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

def parse_usbmon_envelope(usb_path: str, mode: str = 'epoch'):
    first_s: List[float] = []
    last_c: List[float] = []
    co_bytes: List[int] = []
    # we will keep all Bo S times and Co C times, and also store each Co bytes tuple (ts, bytes)
    s_times: List[float] = []
    c_times: List[float] = []
    co_events: List[tuple] = []
    try:
        with open(usb_path,'r',errors='ignore') as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 3:
                    continue
                # timestamp may be usec int or sec float; normalize.
                # Our capture may prepend EPOCH then usb_ts, so index 1 is usb_ts; index 0 is epoch.
                try:
                    time_idx = 0 if mode == 'epoch' else 1
                    ts = float(parts[time_idx])
                    if mode == 'usb' and ts > 1e6:
                        ts = ts/1e6
                except Exception:
                    continue
                # find direction token index and deduce event token just before it
                dir_idx = None
                dir_tok = None
                for i, p2 in enumerate(parts):
                    if p2.startswith('Bo:') or p2.startswith('Co:'):
                        dir_idx = i
                        dir_tok = p2[:2]
                        break
                if dir_idx is None:
                    continue
                ev = parts[dir_idx-1] if dir_idx-1 >= 0 else ''
                if dir_tok=='Bo' and ev=='S':
                    s_times.append(ts)
                elif dir_tok in ('Bo','Co') and ev=='C':
                    # treat any C for OUT direction as contributing to envelope end
                    c_times.append(ts)
                    # parse bytes from len= or fallback column +2
                    nb = 0
                    import re
                    m = re.search(r"len=(\d+)", ln)
                    if m:
                        nb = int(m.group(1))
                    else:
                        # fallback: direction token index detection
                        try:
                            if dir_idx is not None and dir_idx+2 < len(parts):
                                nb = int(parts[dir_idx+2])
                        except Exception:
                            nb = 0
                    co_events.append((ts, nb))
    except FileNotFoundError:
        pass
    s_times.sort(); c_times.sort(); co_events.sort(key=lambda x: x[0])
    return s_times, c_times, co_events

def envelope_for_window(s_times: List[float], c_times: List[float], co_events: List[tuple], b: float, e: float) -> Dict[str, Any]:
    # clip to window: first Bo S inside [b,e], last Co inside [b,e]
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
        # clip to window edges if necessary
        fs = max(first_s, b)
        lc = min(last_c, e)
        if lc > fs:
            span_s = lc - fs
    # bytes: sum Co bytes where completion time within [b,e]
    bytes_sum = 0
    import bisect
    # co_events sorted by ts
    # naive scan (short lists expected)
    for ts, nb in co_events:
        if ts < b: continue
        if ts > e: break
        bytes_sum += int(nb or 0)
    return {
        'span_s': span_s,
        'bytes_out': bytes_sum,
    }

def main():
    if len(sys.argv) < 2:
        print('Usage: bo_envelope_span.py <combo_K_dir>')
        sys.exit(1)
    k_dir = sys.argv[1]
    usb = os.path.join(k_dir, 'usbmon.txt')
    tm = os.path.join(k_dir, 'time_map.json')
    merged = os.path.join(k_dir, 'merged_invokes.json')
    T = load_json(tm)
    M = load_json(merged)
    if not T or not M:
        print('missing time_map or merged_invokes')
        sys.exit(2)
    usb_ref = T.get('usbmon_ref') if T else None
    epoch_ref = T.get('epoch_ref') if T else None
    bt_ref = T.get('boottime_ref') if T else None
    # Decide which time base to use for parsing/mapping
    mode = None
    if usb_ref is not None and bt_ref is not None:
        mode = 'usb'
    else:
        # Fallback: use epoch column (index 0) and derive boot_epoch from current time
        mode = 'epoch'
    s_times, c_times, co_events = parse_usbmon_envelope(usb, mode=mode)
    # group spans by seg label in merged
    spans = M.get('spans') or []
    per_seg: Dict[str, List[Dict[str,Any]]] = {}
    for sp in spans:
        # If usb_ref exists, use direct mapping: usb = boottime - bt_ref + usb_ref
        # Else if only epoch_ref exists, fall back to using epoch timeline (index 0) which matches boottime approximately
        if mode == 'usb':
            b0 = sp['begin'] - bt_ref + usb_ref
            e0 = sp['end'] - bt_ref + usb_ref
        else:  # epoch mode (derive boot_epoch at analysis time)
            import time as _t
            try:
                bt_now = _t.clock_gettime(_t.CLOCK_BOOTTIME)
            except Exception:
                bt_now = float(open('/proc/uptime').read().split()[0])
            epoch_now = _t.time()
            boot_epoch = epoch_now - bt_now
            b0 = boot_epoch + sp['begin']
            e0 = boot_epoch + sp['end']
        seg = sp.get('seg_label','seg')
        env = envelope_for_window(s_times, c_times, co_events, b0, e0)
        per_seg.setdefault(seg, []).append(env)
    # summarize
    out = {}
    for seg, xs in per_seg.items():
        if not xs:
            continue
        avg_span = sum(x['span_s'] for x in xs)/len(xs)
        avg_bytes = sum(x['bytes_out'] for x in xs)/len(xs)
        speed = (avg_bytes/MiB)/avg_span if avg_span>0 else 0.0
        # per-span speeds (skip invalid spans)
        speeds = []
        for x in xs:
            s = x.get('span_s') or 0.0
            b = x.get('bytes_out') or 0
            if s > 0:
                speeds.append((b/MiB)/s)
        min_speed = min(speeds) if speeds else 0.0
        max_speed = max(speeds) if speeds else 0.0
        out[seg] = {
            'count': len(xs),
            'avg_span_s': avg_span,
            'avg_bytes_out': int(avg_bytes),
            'speed_MiBps': speed,
            'min_speed_MiBps': min_speed,
            'max_speed_MiBps': max_speed,
        }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
