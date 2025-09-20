#!/usr/bin/env python3
"""
Plot a simple H2D -> Compute -> D2H timeline from a usbmon capture.

Usage:
  python tools/plot_usbmon_timeline.py <capture_dir> [--invoke K] [--min-bytes 64] [--expand-ms 5] \
         [--outfile out.png] [--title "..."]

Where <capture_dir> contains at least:
  - usbmon.txt
  - invokes.json  (with key 'spans': [{'begin':..,'end':..}, ...] in boottime)
  - time_map.json (with keys 'usbmon_ref' and 'boottime_ref' to align time axes)

The script:
  - Parses usbmon URB pairs (S/C) and keeps intervals with bytes >= min-bytes.
  - For the selected invoke, clips URB intervals to the invoke window (with optional expansion).
  - Computes H2D (Bo/Co) union intervals and D2H (Bi/Ci) union intervals.
  - Plots three lanes: H2D burst(s), TPU compute gap, D2H burst(s).
  - Annotates that no >min-bytes packets appear in the other's window if the sets don't overlap.

If you only want a schematic (no data), use --demo to render a synthetic example.
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def parse_urbs(usbmon_file: str, dirs: Tuple[str, str]) -> List[Tuple[float, float, int, str, Optional[int]]]:
    """Parse URB pairs from usbmon.txt. Returns list of (start_s, end_s, bytes, dirTok, devNum).

    dirs: (submit_dir, complete_dir), e.g., ('Bo','Co') or ('Bi','Ci').
    """
    sub_dir, com_dir = dirs
    pending = {}
    finished: List[Tuple[float, float, int, str, Optional[int]]] = []
    re_dir = re.compile(r"([CB][io]):(\d+):(\d+):(\d+)")
    try:
        with open(usbmon_file, 'r', errors='ignore') as f:
            for ln in f:
                cols = ln.split()
                if len(cols) < 3:
                    continue
                tag = cols[0]
                # timestamp can be seconds or microseconds
                try:
                    ts = float(cols[1])
                    ts = ts / 1e6 if ts > 1e6 else ts
                except Exception:
                    continue
                if cols[2] not in ('S','C'):
                    continue
                sc = cols[2]
                mdir = re_dir.search(ln)
                if not mdir:
                    continue
                dir_tok = mdir.group(1)
                try:
                    dev_num = int(mdir.group(3))
                except Exception:
                    dev_num = None
                if dir_tok not in (sub_dir, com_dir):
                    continue

                if sc == 'S':
                    pending[tag] = (ts, dir_tok, dev_num)
                elif sc == 'C':
                    start_ts = None
                    if tag in pending:
                        s_ts, s_dir, s_dev = pending.pop(tag)
                        if s_dir == sub_dir:
                            start_ts = s_ts
                            dev_num = s_dev if s_dev is not None else dev_num
                    # bytes: prefer len= on C, otherwise fall back to parsed column or trailing '# n'
                    nbytes = 0
                    mlen = re.search(r"len=(\d+)", ln)
                    if mlen:
                        try:
                            nbytes = int(mlen.group(1))
                        except Exception:
                            nbytes = 0
                    else:
                        parts = ln.strip().split()
                        dir_idx = None
                        for i, tok in enumerate(parts):
                            if re.match(r'^[CB][io]:\d+:', tok):
                                dir_idx = i
                                break
                        if dir_idx is not None and dir_idx + 2 < len(parts):
                            try:
                                nbytes = int(parts[dir_idx + 2])
                            except Exception:
                                nbytes = 0
                        if nbytes == 0:
                            m2 = re.search(r"#\s*(\d+)", ln)
                            if m2:
                                try:
                                    nbytes = int(m2.group(1))
                                except Exception:
                                    nbytes = 0
                    if start_ts is not None:
                        finished.append((start_ts, ts, nbytes, dir_tok, dev_num))
    except FileNotFoundError:
        pass
    return finished


def merge_union(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    itv = sorted(intervals, key=lambda x: x[0])
    merged = [list(itv[0])]
    for s, e in itv[1:]:
        cur = merged[-1]
        if s <= cur[1]:
            if e > cur[1]:
                cur[1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def clip_intervals(urbs: List[Tuple[float, float, int, str, Optional[int]]],
                   win: Tuple[float, float], min_bytes: int) -> List[Tuple[float, float]]:
    b, e = win
    out: List[Tuple[float, float]] = []
    for s, t, nb, _d, _dv in urbs:
        if (nb or 0) < min_bytes:
            continue
        if t <= b or s >= e:
            continue
        ss = b if s < b else s
        ee = e if t > e else t
        if ee > ss:
            out.append((ss, ee))
    return out


def load_invokes(invokes_json: str) -> List[Tuple[float, float]]:
    data = json.load(open(invokes_json))
    spans = data.get('spans', [])
    return [(float(x.get('begin', 0.0)), float(x.get('end', 0.0))) for x in spans]


def map_to_usbmon(win_bt: Tuple[float, float], time_map: dict) -> Tuple[float, float]:
    usb_ref = time_map.get('usbmon_ref')
    bt_ref = time_map.get('boottime_ref')
    b, e = win_bt
    if usb_ref is None or bt_ref is None:
        # Fallback: assume the same timescale
        return b, e
    return (b - bt_ref + usb_ref, e - bt_ref + usb_ref)


def pick_invoke_idx(n: int, prefer: Optional[int]) -> int:
    if prefer is not None:
        if prefer < 0:
            return max(0, n + prefer)
        return min(max(0, prefer), max(0, n - 1))
    # default: skip first (cold), pick 2nd if exists, else 0
    return 1 if n >= 2 else 0


def plot_timeline(h2d_itvs: List[Tuple[float,float]],
                  d2h_itvs: List[Tuple[float,float]],
                  title: str,
                  outfile: str):
    # Establish time zero at earliest boundary among intervals for tight layout
    times = [t for s,e in h2d_itvs + d2h_itvs for t in (s,e)]
    if not times:
        raise SystemExit('No intervals to plot.')
    t0 = min(times)
    # Shift to ms for readability
    def ms(x: float) -> float:
        return (x - t0) * 1000.0

    # Compute gap for compute visualization
    h2d_end = max(e for _s, e in h2d_itvs) if h2d_itvs else None
    d2h_start = min(s for s, _e in d2h_itvs) if d2h_itvs else None
    has_gap = h2d_end is not None and d2h_start is not None and (d2h_start - h2d_end) > 0
    gap_ms = (d2h_start - h2d_end) * 1000.0 if has_gap else 0.0

    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=140)
    y_h2d, y_comp, y_d2h = 2.0, 1.0, 0.0

    # Draw H2D and D2H union intervals as thick bars
    for s, e in h2d_itvs:
        rect = Rectangle((ms(s), y_h2d - 0.25), ms(e) - ms(s), 0.5, color='#1f77b4', alpha=0.85)
        ax.add_patch(rect)
    for s, e in d2h_itvs:
        rect = Rectangle((ms(s), y_d2h - 0.25), ms(e) - ms(s), 0.5, color='#d62728', alpha=0.85)
        ax.add_patch(rect)

    # Compute region as the gap between H2D end and D2H start
    if has_gap:
        rect = Rectangle((ms(h2d_end), y_comp - 0.2), ms(d2h_start) - ms(h2d_end), 0.4,
                         color='#7f7f7f', alpha=0.35, hatch='////')
        ax.add_patch(rect)
        ax.text((ms(h2d_end)+ms(d2h_start))/2, y_comp + 0.35,
                f'TPU Compute ~{gap_ms:.2f} ms', ha='center', va='bottom', fontsize=9)

    # Labels and cosmetics
    ax.set_yticks([y_d2h, y_comp, y_h2d])
    ax.set_yticklabels(['D2H (Bi/Ci)', 'Compute', 'H2D (Bo/Co)'])
    ax.set_xlabel('Time (ms, relative)')
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.35)
    # Tight x range
    x_min = 0.0
    x_max = max([ms(e) for _s, e in h2d_itvs + d2h_itvs] + [0.0]) + 0.2
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.6, 2.6)

    # Overlap check and annotation
    def overlaps(a: List[Tuple[float,float]], b: List[Tuple[float,float]]) -> bool:
        i, j = 0, 0
        while i < len(a) and j < len(b):
            s1, e1 = a[i]
            s2, e2 = b[j]
            if e1 <= s2:
                i += 1
            elif e2 <= s1:
                j += 1
            else:
                return True
        return False

    has_overlap = overlaps(h2d_itvs, d2h_itvs)
    note = 'No H2D/D2H (>64B) overlap' if not has_overlap else 'H2D/D2H windows overlap'
    ax.text(0.01, 0.98, note, transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    Path(os.path.dirname(outfile) or '.').mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)
    return outfile


def main():
    p = argparse.ArgumentParser(description='Plot H2D -> Compute -> D2H timeline from usbmon.')
    p.add_argument('capture_dir', nargs='?', help='Directory with usbmon.txt, invokes.json, time_map.json')
    p.add_argument('--invoke', type=int, default=None, help='Invoke index (default: 1, i.e., skip first)')
    p.add_argument('--min-bytes', type=int, default=64, help='Only count URBs >= this bytes')
    p.add_argument('--expand-ms', type=float, default=5.0, help='Window expansion on both sides in ms')
    p.add_argument('--outfile', type=str, default=None, help='Output image path')
    p.add_argument('--title', type=str, default='usbmon: H2D -> Compute -> D2H')
    p.add_argument('--demo', action='store_true', help='Render a schematic without reading data')
    args = p.parse_args()

    if args.demo:
        # Synthetic example: H2D 0-1.2ms, compute 1.2-6.5ms, D2H 6.5-7.1ms
        h2d = [(0.0000, 0.0012)]
        d2h = [(0.0065, 0.0071)]
        out = args.outfile or 'results/plots/usbmon_demo_timeline.png'
        plot_timeline(h2d, d2h, args.title, out)
        print(out)
        return

    if not args.capture_dir:
        raise SystemExit('Please provide a capture_dir or use --demo')

    cap = Path(args.capture_dir)
    usbmon_path = cap / 'usbmon.txt'
    invokes_path = cap / 'invokes.json'
    timemap_path = cap / 'time_map.json'
    if not usbmon_path.exists() or not invokes_path.exists() or not timemap_path.exists():
        raise SystemExit(f'Missing required files in {cap}: usbmon.txt/invokes.json/time_map.json')

    # Load windows and pick invoke
    invokes_bt = load_invokes(str(invokes_path))
    if not invokes_bt:
        raise SystemExit('No invokes found in invokes.json')
    k = pick_invoke_idx(len(invokes_bt), args.invoke)
    win_bt = invokes_bt[k]
    tm = json.load(open(timemap_path))
    expand_s = (args.expand_ms or 0.0) / 1000.0
    win_usb = (win_bt[0] - expand_s, win_bt[1] + expand_s)
    win_usb = map_to_usbmon(win_usb, tm)

    # Parse URBs and clip
    out_urbs = parse_urbs(str(usbmon_path), ('Bo','Co'))
    in_urbs = parse_urbs(str(usbmon_path), ('Bi','Ci'))

    h2d_intervals = clip_intervals(out_urbs, win_usb, args.min_bytes)
    d2h_intervals = clip_intervals(in_urbs, win_usb, args.min_bytes)
    h2d_union = merge_union(h2d_intervals)
    d2h_union = merge_union(d2h_intervals)

    if not h2d_union and not d2h_union:
        raise SystemExit('No qualifying URBs found in the selected window. Try lowering --min-bytes or increasing --expand-ms')

    # Output path
    default_out = Path('results/plots') / f"usbmon_timeline_invoke{k}.png"
    out = args.outfile or str(default_out)
    title = args.title or 'usbmon: H2D -> Compute -> D2H'
    title += f' (invoke {k}, min>{args.min_bytes}B)'

    outfile = plot_timeline(h2d_union, d2h_union, title, out)
    print(outfile)


if __name__ == '__main__':
    main()

