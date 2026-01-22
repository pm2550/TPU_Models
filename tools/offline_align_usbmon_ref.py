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
  python tools/offline_align_usbmon_ref.py <usbmon.txt> <invokes.json> <time_map.json> [--min-urb-bytes 65536] [--dev NNN] [--usb-device usb:X]

Options:
  --dev NNN          Only consider usbmon packets from device NNN (e.g., 003, 004).
                     This is CRITICAL for multi-device tests where devices have different
                     first-packet times. Without this, alignment may be off by several ms.
  --usb-device usb:X Only consider invokes from this USB device (e.g., usb:0, usb:1).
                     Use this when invokes.json contains mixed devices.

It writes back time_map.json in place and prints the chosen anchor + new usbmon_ref.

WARNING: For dual-TPU or multi-device tests, you MUST run this separately for each
device with --dev and --usb-device, and save to separate time_map files (e.g., time_map_dev003.json).

Example for dual-TPU test:
  # For usb:1 (MN7) on dev003:
  python tools/offline_align_usbmon_ref.py usbmon.txt invokes.json time_map_dev003.json --dev 003 --usb-device usb:1
  # For usb:0 (DeepLab) on dev004:
  python tools/offline_align_usbmon_ref.py usbmon.txt invokes.json time_map_dev004.json --dev 004 --usb-device usb:0
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


def find_first_anchor(usbmon_path: str, min_urb_bytes: int, dev_filter: str = None) -> Optional[float]:
    """Find first Bo S packet with size >= threshold.
    
    Args:
        dev_filter: If set, only consider packets from this device (e.g., '003', '004')
    """
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
                # Check device filter
                if dev_filter and m.group(2) != dev_filter:
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


def find_first_anchor_c(usbmon_path: str, min_urb_bytes: int, dev_filter: str = None) -> Optional[float]:
    """Find first Bo/Co C packet with size >= threshold.
    
    Args:
        dev_filter: If set, only consider packets from this device (e.g., '003', '004')
    """
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
                # Check device filter
                if dev_filter and m.group(2) != dev_filter:
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


def find_first_anchor_in_window(usbmon_path: str, min_urb_bytes: int, dev_filter: str,
                                 window_start: float, window_end: float) -> Optional[float]:
    """Find first Bo S packet with size >= threshold within a time window.
    
    This is critical for multi-device tests where different devices may have 
    their first packets at very different times (e.g., 700ms apart).
    """
    re_dir = re.compile(r"([CB][io]):(\d+):(\d+):(\d+)")
    candidates = []
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
                # Only consider packets within window
                if ts < window_start or ts > window_end:
                    continue
                m = re_dir.search(ln)
                if not m:
                    continue
                # Check device filter
                if dev_filter and m.group(2) != dev_filter:
                    continue
                sc = parts[2]
                tok = m.group(1)
                if sc != 'S' or not tok.startswith('Bo'):
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
                    candidates.append(ts)
                    if len(candidates) >= 3:  # Found enough, can stop
                        break
    except FileNotFoundError:
        return None
    # Return the earliest candidate
    return min(candidates) if candidates else None


def main(argv):
    if len(argv) < 3:
        print("Usage: offline_align_usbmon_ref.py <usbmon.txt> <invokes.json> <time_map.json> [--min-urb-bytes 65536] [--dev NNN]", file=sys.stderr)
        return 2
    usbmon_path = argv[0]
    invokes_path = argv[1]
    timemap_path = argv[2]
    # args
    min_urb_bytes = 65536
    dev_filter = None
    usb_device = None  # Filter spans by usb:0, usb:1, etc.
    i = 3
    while i < len(argv):
        if argv[i] == '--min-urb-bytes' and i + 1 < len(argv):
            try:
                min_urb_bytes = int(argv[i + 1])
            except Exception:
                pass
            i += 2
        elif argv[i] == '--dev' and i + 1 < len(argv):
            dev_filter = argv[i + 1]
            i += 2
        elif argv[i] == '--usb-device' and i + 1 < len(argv):
            usb_device = argv[i + 1]  # e.g., "usb:0" or "usb:1"
            i += 2
        else:
            i += 1

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

    # first invoke begin (optionally filtered by usb_device)
    try:
        inv = json.load(open(invokes_path))
        spans = inv.get('spans') or []
        if not spans:
            print("invokes.json has no spans", file=sys.stderr)
            return 5
        # Filter by USB device if specified
        if usb_device:
            spans = [s for s in spans if s.get('device') == usb_device]
            if not spans:
                print(f"No spans found for device {usb_device}", file=sys.stderr)
                return 5
        t0 = float(spans[0]['begin'])
        t1 = float(spans[0]['end'])
        invoke_span = t1 - t0  # invoke duration in seconds
    except Exception as e:
        print(f"Failed to read invokes: {e}", file=sys.stderr)
        return 6

    # For multi-device scenarios, we need to find the anchor within the invoke window,
    # not the global first packet. Strategy:
    # 1. Use any existing usbmon_ref estimate to find approximate invoke[0] window in usbmon time
    # 2. Search for first large Bo S within that window (with some tolerance)
    
    old_usb_ref = tm.get('usbmon_ref')
    if old_usb_ref is not None and dev_filter:
        # Estimate invoke[0] window in usbmon time
        est_t0_usb = old_usb_ref + (t0 - bt_ref)
        est_t1_usb = old_usb_ref + (t1 - bt_ref)
        # Search with tolerance (±50ms before, ±invoke_span after)
        search_start = est_t0_usb - 0.05
        search_end = est_t1_usb + invoke_span
        
        anchor = find_first_anchor_in_window(usbmon_path, min_urb_bytes, dev_filter, 
                                              search_start, search_end)
        if anchor is not None:
            print(f"[INFO] Found anchor within invoke window: {anchor:.6f}s", file=sys.stderr)
    else:
        anchor = None
    
    # Fallback to original method if window search failed
    if anchor is None:
        anchor = find_first_anchor(usbmon_path, min_urb_bytes, dev_filter)
    if anchor is None:
        anchor = find_first_anchor_c(usbmon_path, min_urb_bytes, dev_filter)
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
        'min_urb_bytes': min_urb_bytes,
        'dev_filter': dev_filter,
        'usb_device': usb_device
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

