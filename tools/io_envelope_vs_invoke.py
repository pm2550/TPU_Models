#!/usr/bin/env python3
import os, sys, json, re, time
from typing import List, Tuple, Dict, Any

MiB = 1024.0*1024.0

def load_json(p: str) -> dict:
    try:
        with open(p,'r') as f:
            return json.load(f)
    except Exception:
        return {}

def parse_usbmon_times(usb_path: str, mode: str='epoch') -> Tuple[List[float], List[float], List[Tuple[float,int]], List[Tuple[float,int]]]:
    """
    Parse usbmon.txt and return:
      - s_times_in, c_times_in for IN direction envelope (Bi/Ci) as timestamps
      - ci_events: list of (ts, bytes)
      - co_events: list of (ts, bytes)
    mode: 'epoch' uses column 0 as timestamp; 'usb' uses column 1 and normalizes usec
    """
    s_in: List[float] = []
    c_in: List[float] = []
    s_out: List[float] = []
    c_out: List[float] = []
    ci_events: List[Tuple[float,int]] = []
    co_events: List[Tuple[float,int]] = []
    try:
        with open(usb_path,'r',errors='ignore') as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 3:
                    continue
                try:
                    time_idx = 0 if mode=='epoch' else 1
                    ts = float(parts[time_idx])
                    if mode=='usb' and ts > 1e6:
                        ts = ts/1e6
                except Exception:
                    continue
                dir_idx = None
                dir_tok = None
                for i,p2 in enumerate(parts):
                    if p2.startswith('Bo:') or p2.startswith('Co:') or p2.startswith('Bi:') or p2.startswith('Ci:'):
                        dir_idx = i
                        dir_tok = p2[:2]  # Bo/Bi/Co/Ci
                        break
                if dir_idx is None:
                    continue
                ev = parts[dir_idx-1] if dir_idx-1 >= 0 else ''
                # parse len
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
                if dir_tok == 'Bo':
                    if ev=='S': s_out.append(ts)
                    elif ev=='C':
                        c_out.append(ts)
                        co_events.append((ts, nb))
                elif dir_tok == 'Bi':
                    if ev=='S': s_in.append(ts)
                    elif ev=='C':
                        c_in.append(ts)
                        ci_events.append((ts, nb))
                elif dir_tok == 'Co':
                    # Some formats show Co with event token too
                    if ev=='C':
                        c_out.append(ts)
                        co_events.append((ts, nb))
                elif dir_tok == 'Ci':
                    if ev=='C':
                        c_in.append(ts)
                        ci_events.append((ts, nb))
    except FileNotFoundError:
        pass
    s_in.sort(); c_in.sort(); s_out.sort(); c_out.sort()
    ci_events.sort(key=lambda x:x[0]); co_events.sort(key=lambda x:x[0])
    return s_in, c_in, ci_events, co_events, s_out, c_out

def first_S_last_C_union(s_in: List[float], c_in: List[float], s_out: List[float], c_out: List[float], b: float, e: float) -> Tuple[float, float, float]:
    """
    Compute union envelope: earliest S (Bi or Bo) within [b,e] and latest C (Ci or Co) within [b,e].
    Returns (envelope_span_s, outside_before_s, outside_after_s).
    If no S/C found in window, envelope is 0 and outside is full window.
    """
    import bisect
    # find first S >= b for each list
    def first_in_window(arr):
        i = bisect.bisect_left(arr, b)
        if i < len(arr) and arr[i] <= e:
            return arr[i]
        return None
    s1 = first_in_window(s_in)
    s2 = first_in_window(s_out)
    # last C <= e and >= b
    def last_in_window(arr):
        j = bisect.bisect_right(arr, e) - 1
        if j >= 0 and arr[j] >= b:
            return arr[j]
        return None
    c1 = last_in_window(c_in)
    c2 = last_in_window(c_out)
    candidates_S = [x for x in [s1, s2] if x is not None]
    candidates_C = [x for x in [c1, c2] if x is not None]
    if not candidates_S or not candidates_C:
        total = max(0.0, e - b)
        return 0.0, total, 0.0
    firstS = min(candidates_S)
    lastC = max(candidates_C)
    fs = max(b, firstS)
    lc = min(e, lastC)
    span = max(0.0, lc - fs)
    before = max(0.0, fs - b)
    after = max(0.0, e - lc)
    return span, before, after

def sum_bytes_within(ci_events: List[Tuple[float,int]], co_events: List[Tuple[float,int]], b: float, e: float) -> Tuple[int,int]:
    bi = 0; bo = 0
    for ts, nb in ci_events:
        if ts < b: continue
        if ts > e: break
        bi += int(nb or 0)
    for ts, nb in co_events:
        if ts < b: continue
        if ts > e: break
        bo += int(nb or 0)
    return bi, bo

def main():
    if len(sys.argv) < 2:
        print('Usage: io_envelope_vs_invoke.py <combo_dir>')
        sys.exit(1)
    root = sys.argv[1]
    usb = os.path.join(root, 'usbmon.txt')
    tm = os.path.join(root, 'time_map.json')
    merged = os.path.join(root, 'merged_invokes.json')
    T = load_json(tm)
    M = load_json(merged)
    if not M:
        print('missing merged_invokes.json')
        sys.exit(2)
    usb_ref = (T or {}).get('usbmon_ref')
    bt_ref = (T or {}).get('boottime_ref')
    mode = 'usb' if (usb_ref is not None and bt_ref is not None) else 'epoch'
    s_in, c_in, ci_events, co_events, s_out, c_out = parse_usbmon_times(usb, mode)
    spans = M.get('spans') or []
    # derive boot_epoch if needed
    if mode == 'epoch':
        try:
            bt_now = time.clock_gettime(time.CLOCK_BOOTTIME)
        except Exception:
            bt_now = float(open('/proc/uptime').read().split()[0])
        epoch_now = time.time()
        boot_epoch = epoch_now - bt_now
    per_model: Dict[str, list] = {}
    for sp in spans:
        if mode == 'usb':
            b0 = sp['begin'] - bt_ref + usb_ref
            e0 = sp['end'] - bt_ref + usb_ref
        else:
            b0 = boot_epoch + sp['begin']
            e0 = boot_epoch + sp['end']
        seg = sp.get('seg_label','seg')
        invoke_s = max(0.0, e0 - b0)
        span, before, after = first_S_last_C_union(s_in, c_in, s_out, c_out, b0, e0)
        outside = before + after
        bi_bytes, bo_bytes = sum_bytes_within(ci_events, co_events, b0, e0)
        per_model.setdefault(seg, []).append({
            'invoke_s': invoke_s,
            'envelope_s': span,
            'outside_s': outside,
            'outside_before_s': before,
            'outside_after_s': after,
            'bytes_in': bi_bytes,
            'bytes_out': bo_bytes,
        })
    # aggregate averages
    summary: Dict[str, Any] = {}
    for seg, xs in per_model.items():
        n = len(xs)
        if n == 0: continue
        def avg(key):
            return sum(x[key] for x in xs) / n
        summary[seg] = {
            'count': n,
            'avg_invoke_s': avg('invoke_s'),
            'avg_envelope_s': avg('envelope_s'),
            'avg_outside_s': avg('outside_s'),
            'avg_outside_before_s': avg('outside_before_s'),
            'avg_outside_after_s': avg('outside_after_s'),
            'avg_bytes_in': int(avg('bytes_in')),
            'avg_bytes_out': int(avg('bytes_out')),
            'avg_total_bytes_MiBps_over_envelope': ((avg('bytes_in')+avg('bytes_out'))/MiB)/max(1e-9, avg('envelope_s')),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
