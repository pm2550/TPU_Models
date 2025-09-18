#!/usr/bin/env python3
import json
import os
import re
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# Contract
# Inputs:
# - usbmon.txt: lines like "<ptr> <time> [SC] [Bo/Bi]:bus:dev:endp ... size ..."
# - invokes.json: list of dicts with 'idx', 'begin_ms', 'end_ms', 'duration_ms'
# - time_map.json: mapping {'boottime_to_usbmon': {'offset': float, 'scale': float}} or list of pairs
# Outputs:
# - per-invoke CSV with columns: invoke_idx, invoke_ms, env_ms, host_ms, env_t0, env_t1, note
# - summary JSON with n, mean, stdev, p50

USBMON_RE = re.compile(r"^\s*\S+\s+(?P<ts>\d+)\s+(?P<sc>[SC])\s+(?P<dir>Bo|Bi):(?P<bus>\d+):(?P<dev>\d+):(?P<ep>\d+)\s+(?P<rest>.*)$")

@dataclass
class Urb:
    ts: int  # usbmon time units (same as file)
    sc: str  # 'S' or 'C'
    dir: str # 'Bo' or 'Bi'
    size: int


def load_time_map(path: str):
    """Return a function mapping boottime seconds -> usbmon microsecond ticks (int)."""
    with open(path, 'r') as f:
        data = json.load(f)
    # Preferred format: {"usbmon_ref": <sec>, "boottime_ref": <sec>}
    if isinstance(data, dict) and 'usbmon_ref' in data and 'boottime_ref' in data:
        usb_ref = float(data['usbmon_ref'])
        boot_ref = float(data['boottime_ref'])
        return lambda t_sec: int(round((t_sec - boot_ref + usb_ref) * 1_000_000))
    # Fallback linear mapping (a,b) in usbmon microseconds given milliseconds or seconds; assume seconds
    if isinstance(data, dict) and 'boottime_to_usbmon' in data:
        m = data['boottime_to_usbmon']
        a = float(m.get('scale') or m.get('slope') or 1.0)
        b = float(m.get('offset') or m.get('intercept') or 0.0)
        return lambda t_sec: int(round((a * t_sec + b) * 1_000_000))
    # identity: treat input as seconds to microseconds
    return lambda t_sec: int(round(t_sec * 1_000_000))


def parse_usbmon(path: str) -> List[Urb]:
    urbs: List[Urb] = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            m = USBMON_RE.match(line)
            if not m:
                continue
            ts = int(m.group('ts'))
            sc = m.group('sc')
            dir_ = m.group('dir')
            rest = m.group('rest')
            # size is last int before '=' or '<' or '>'
            size = None
            for tok in rest.split():
                try:
                    size = int(tok)
                except Exception:
                    continue
            if size is None:
                size = 0
            urbs.append(Urb(ts=ts, sc=sc, dir=dir_, size=size))
    return urbs


def env_bo_to_bi_within(urbs: List[Urb], t0: int, t1: int) -> Optional[Tuple[int,int]]:
    """Envelope inside [t0,t1]: earliest Bo(S) to latest Bi(C); fallbacks if missing."""
    # Primary picks: start of bulk out submission to end of bulk in completion
    bo_S = [u.ts for u in urbs if u.sc=='S' and u.dir=='Bo' and t0 <= u.ts <= t1]
    bi_C = [u.ts for u in urbs if u.sc=='C' and u.dir=='Bi' and t0 <= u.ts <= t1]
    if bo_S and bi_C:
        return min(bo_S), max(bi_C)
    # Fallback 1: use Bi(S) if no Bi(C)
    bi_S = [u.ts for u in urbs if u.sc=='S' and u.dir=='Bi' and t0 <= u.ts <= t1]
    if bo_S and bi_S:
        return min(bo_S), max(bi_S)
    # Fallback 2: any usbmon line timestamps within window (node-to-node)
    any_ts = [u.ts for u in urbs if t0 <= u.ts <= t1]
    if any_ts:
        return min(any_ts), max(any_ts)
    return None


def compute_host_times_single(model_dir: str, out_dir: str):
    seg1 = os.path.join(model_dir, 'seg1')
    usb_path = os.path.join(seg1, 'usbmon.txt')
    inv_path = os.path.join(seg1, 'invokes.json')
    map_path = os.path.join(seg1, 'time_map.json')
    if not (os.path.isfile(usb_path) and os.path.isfile(inv_path) and os.path.isfile(map_path)):
        return None
    with open(inv_path, 'r') as f:
        inv_obj = json.load(f)
    # Structure: {"name": ..., "spans": [{"begin": sec, "end": sec, ...}, ...]}
    spans = inv_obj['spans'] if isinstance(inv_obj, dict) and 'spans' in inv_obj else inv_obj
    to_usb = load_time_map(map_path)
    urbs = parse_usbmon(usb_path)
    rows = []
    for idx, inv in enumerate(spans):
        b_sec = inv.get('begin') if isinstance(inv, dict) else None
        e_sec = inv.get('end') if isinstance(inv, dict) else None
        if b_sec is None or e_sec is None:
            continue
        # Use usbmon mapping inside [set_begin, get_end] when available
        set_b = inv.get('set_begin') if isinstance(inv, dict) else None
        get_e = inv.get('get_end') if isinstance(inv, dict) else None
        if set_b is not None and get_e is not None:
            b_map_sec = float(set_b)
            e_map_sec = float(get_e)
            invoke_ms = (e_map_sec - b_map_sec) * 1000.0
        else:
            b_map_sec = float(b_sec)
            e_map_sec = float(e_sec)
            invoke_ms = (e_sec - b_sec) * 1000.0
        note = ''
        # Expand window slightly to capture trailing USB completions beyond host timestamps
        b_u = to_usb(b_map_sec) - int(1_000)  # -1 ms
        e_u = to_usb(e_map_sec) + int(15_000)  # +15 ms
        env = env_bo_to_bi_within(urbs, b_u, e_u)
        if env is None:
            note = 'no_bo_or_bi_in_window'
            env_ms = None
            host_ms = None
            env_t = (None, None)
        else:
            t0, t1 = env
            env_ms = (t1 - t0) / 1000.0
            host_ms = invoke_ms - env_ms
            env_t = (t0, t1)
        rows.append({
            'invoke_idx': inv.get('idx') if isinstance(inv, dict) and 'idx' in inv else idx,
            'invoke_ms': (e_sec - b_sec) * 1000.0,
            'env_ms': env_ms,
            'host_ms': host_ms,
            'env_t0': env_t[0] if 'env_t' in locals() and env_t else None,
            'env_t1': env_t[1] if 'env_t' in locals() and env_t else None,
            'note': note
        })
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'single_seg1_host_times.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # summary
    vals = [r['host_ms'] for r in rows if r['host_ms'] is not None]
    if vals:
        import statistics as stats
        mean = stats.mean(vals)
        stdev = stats.pstdev(vals) if len(vals)>1 else 0.0
        srt = sorted(vals)
        mid = len(srt)//2
        p50 = (srt[mid] if len(srt)%2==1 else 0.5*(srt[mid-1]+srt[mid]))
    else:
        mean = stdev = p50 = None
    with open(os.path.join(out_dir, 'single_seg1_host_summary.json'), 'w') as f:
        json.dump({'n': len(vals), 'mean': mean, 'stdev': stdev, 'p50': p50}, f, indent=2)
    return {'n': len(vals), 'mean': mean, 'stdev': stdev, 'p50': p50, 'csv': csv_path}


def compute_host_times_k2(model_dir: str, out_dir: str):
    base = os.path.join(model_dir, 'K2')
    usb_path = os.path.join(base, 'usbmon.txt')
    inv_path = os.path.join(base, 'seg1', 'invokes.json')
    map_path = os.path.join(base, 'time_map.json')
    if not (os.path.isfile(usb_path) and os.path.isfile(inv_path) and os.path.isfile(map_path)):
        return None
    with open(inv_path, 'r') as f:
        inv_obj = json.load(f)
    spans = inv_obj['spans'] if isinstance(inv_obj, dict) and 'spans' in inv_obj else inv_obj
    to_usb = load_time_map(map_path)
    urbs = parse_usbmon(usb_path)
    rows = []
    for idx, inv in enumerate(spans):
        b_sec = inv.get('begin') if isinstance(inv, dict) else None
        e_sec = inv.get('end') if isinstance(inv, dict) else None
        if b_sec is None or e_sec is None:
            continue
        # Use [b_u-1ms, e_u+20ms] window for chain invoke to include late completions
        b_u = to_usb(float(b_sec)) - int(1_000)
        e_u = to_usb(float(e_sec)) + int(20_000)
        # Use expanded window for chain invoke
        env = env_bo_to_bi_within(urbs, b_u, e_u)
        note = ''
        if env is None:
            note = 'no_bo_or_bi_in_window'
            env_ms = None
            host_ms = None
        else:
            t0, t1 = env
            env_ms = (t1 - t0) / 1000.0
            invoke_ms = (e_sec - b_sec) * 1000.0
            host_ms = invoke_ms - env_ms
        rows.append({
            'invoke_idx': inv.get('idx') if isinstance(inv, dict) and 'idx' in inv else idx,
            'invoke_ms': (e_sec - b_sec) * 1000.0,
            'env_ms': env_ms,
            'host_ms': host_ms,
            'env_t0': env[0] if env else None,
            'env_t1': env[1] if env else None,
            'note': note
        })
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'k2_seg1_host_times.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # summary
    vals = [r['host_ms'] for r in rows if r['host_ms'] is not None]
    if vals:
        import statistics as stats
        mean = stats.mean(vals)
        stdev = stats.pstdev(vals) if len(vals)>1 else 0.0
        srt = sorted(vals)
        mid = len(srt)//2
        p50 = (srt[mid] if len(srt)%2==1 else 0.5*(srt[mid-1]+srt[mid]))
    else:
        mean = stdev = p50 = None
    with open(os.path.join(out_dir, 'k2_seg1_host_summary.json'), 'w') as f:
        json.dump({'n': len(vals), 'mean': mean, 'stdev': stdev, 'p50': p50}, f, indent=2)
    return {'n': len(vals), 'mean': mean, 'stdev': stdev, 'p50': p50, 'csv': csv_path}


def main():
    # Single models
    single_root = 'results/models_local_batch_usbmon/single'
    models = [d for d in os.listdir(single_root) if d.endswith('_8seg_uniform_local')]
    out_root = 'five_models/results/host_time'
    os.makedirs(out_root, exist_ok=True)
    summary: Dict[str, Dict] = {}
    for m in models:
        res = compute_host_times_single(os.path.join(single_root, m), os.path.join(out_root, m))
        if res:
            summary[f'single:{m}'] = res
    # K2 models (only those that exist)
    k2_root = 'results/models_local_combo_chain'
    if os.path.isdir(k2_root):
        for m in os.listdir(k2_root):
            k2_dir = os.path.join(k2_root, m)
            if not os.path.isdir(k2_dir):
                continue
            if not os.path.isdir(os.path.join(k2_dir, 'K2')):
                continue
            res = compute_host_times_k2(k2_dir, os.path.join(out_root, f'{m}_K2'))
            if res:
                summary[f'K2:{m}'] = res
    with open(os.path.join(out_root, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()
