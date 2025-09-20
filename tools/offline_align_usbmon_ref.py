#!/usr/bin/env python3
"""
Offline aligner for time_map.json when usbmon_ref is missing/wrong.

Heuristic:
- Use the first invoke begin (CLOCK_BOOTTIME) from invokes.json as the reference t0.
- Find the earliest large OUT anchor in usbmon.txt:
  1) First Bo submit (S) with bytes >= --min-urb-bytes (default 65536)
  2) Fallback: First Bo complete (C) with bytes >= threshold
  3) Fallback: First usbmon timestamp (second column)
- Set usbmon_ref = anchor_ts - (t0 - boottime_ref)

This script only patches time_map.json; it does not change your window/shift policy.

Usage:
  python tools/offline_align_usbmon_ref.py <usbmon.txt> <invokes.json> <time_map.json> [--min-urb-bytes 65536]

It writes back time_map.json in place and prints the chosen anchor + new usbmon_ref.
"""

import sys
import json
import re
from typing import Optional


def parse_first_ts(usbmon_path: str) -> Optional[float]:
    try:
        with open(usbmon_path, 'r', errors='ignore') as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 2:
                    continue
                try:
                    ts = float(parts[1])
                    return ts / 1e6 if ts > 1e6 else ts
                except Exception:
                    continue
    except FileNotFoundError:
        return None
    return None


def find_first_anchor(usbmon_path: str, min_urb_bytes: int) -> Optional[float]:
    re_dir = re.compile(r"([CB][io]):(\d+):(\d+):(\d+)")
    try:
        with open(usbmon_path, 'r', errors='ignore') as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 3:
                    continue
                try:
                    ts = float(parts[1])
                    ts = ts / 1e6 if ts > 1e6 else ts
                except Exception:
                    continue
                m = re_dir.search(ln)
                if not m:
                    continue
                sc = parts[2]
                tok = m.group(1)
                # Bo submit with size >= threshold
                if sc != 'S' or not tok.startswith('Bo'):
                    continue
                # parse bytes (len=... or numeric field after token)
                nbytes = 0
                dir_idx = None
                for i, t in enumerate(parts):
                    if re_dir.match(t):
                        dir_idx = i
                        break
                if dir_idx is not None and dir_idx + 2 < len(parts):
                    try:
                        nbytes = int(parts[dir_idx + 2])
                    except Exception:
                        nbytes = 0
                if nbytes == 0:
                    m2 = re.search(r"len=(\d+)", ln)
                    if m2:
                        nbytes = int(m2.group(1))
                if nbytes >= min_urb_bytes:
                    return ts
    except FileNotFoundError:
        return None
    return None


def find_first_anchor_c(usbmon_path: str, min_urb_bytes: int) -> Optional[float]:
    re_dir = re.compile(r"([CB][io]):(\d+):(\d+):(\d+)")
    try:
        with open(usbmon_path, 'r', errors='ignore') as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 3:
                    continue
                try:
                    ts = float(parts[1])
                    ts = ts / 1e6 if ts > 1e6 else ts
                except Exception:
                    continue
                m = re_dir.search(ln)
                if not m:
                    continue
                sc = parts[2]
                tok = m.group(1)
                if sc != 'C' or not (tok.startswith('Bo') or tok.startswith('Co')):
                    continue
                # parse bytes
                nbytes = 0
                dir_idx = None
                for i, t in enumerate(parts):
                    if re_dir.match(t):
                        dir_idx = i
                        break
                if dir_idx is not None and dir_idx + 2 < len(parts):
                    try:
                        nbytes = int(parts[dir_idx + 2])
                    except Exception:
                        nbytes = 0
                if nbytes == 0:
                    m2 = re.search(r"len=(\d+)", ln)
                    if m2:
                        nbytes = int(m2.group(1))
                if nbytes >= min_urb_bytes:
                    return ts
    except FileNotFoundError:
        return None
    return None


def main(argv):
    if len(argv) < 3:
        print("Usage: offline_align_usbmon_ref.py <usbmon.txt> <invokes.json> <time_map.json> [--min-urb-bytes 65536]", file=sys.stderr)
        return 2
    usbmon_path = argv[0]
    invokes_path = argv[1]
    timemap_path = argv[2]
    # args
    min_urb_bytes = 65536
    if len(argv) >= 5 and argv[3] == '--min-urb-bytes':
        try:
            min_urb_bytes = int(argv[4])
        except Exception:
            pass

    # load time_map (need boottime_ref)
    try:
        tm = json.load(open(timemap_path))
    except Exception as e:
        print(f"Failed to read time_map: {e}", file=sys.stderr)
        return 3
    bt_ref = tm.get('boottime_ref')
    if bt_ref is None:
        print("time_map.boottime_ref is missing; cannot align.", file=sys.stderr)
        return 4

    # first invoke begin
    try:
        inv = json.load(open(invokes_path))
        spans = inv.get('spans') or []
        if not spans:
            print("invokes.json has no spans", file=sys.stderr)
            return 5
        t0 = float(spans[0]['begin'])
    except Exception as e:
        print(f"Failed to read invokes: {e}", file=sys.stderr)
        return 6

    # choose anchor
    anchor = find_first_anchor(usbmon_path, min_urb_bytes)
    if anchor is None:
        anchor = find_first_anchor_c(usbmon_path, min_urb_bytes)
    if anchor is None:
        anchor = parse_first_ts(usbmon_path)
    if anchor is None:
        print("No usable anchor found in usbmon.txt", file=sys.stderr)
        return 7

    # compute usbmon_ref
    usb_ref = float(anchor) - (t0 - float(bt_ref))
    tm['usbmon_ref'] = usb_ref
    try:
        with open(timemap_path, 'w') as f:
            json.dump(tm, f)
    except Exception as e:
        print(f"Failed to write time_map: {e}", file=sys.stderr)
        return 8

    print(json.dumps({
        'status': 'ok',
        'anchor_ts_s': anchor,
        'boottime_ref_s': bt_ref,
        'first_invoke_begin_s': t0,
        'usbmon_ref_s': usb_ref,
        'min_urb_bytes': min_urb_bytes
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

