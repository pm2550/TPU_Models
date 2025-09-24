#!/usr/bin/env python3
"""
Compute average off-chip transfer time from usbmon logs for segments
that have non-zero weights_stream_MiB in five_models/results/theory_chain_source_data.csv.

For each matching segment, we parse its usbmon.txt under:
  results/models_local_batch_usbmon/single/<model>/<segX>/usbmon.txt

We identify pairs of usb OUT completions:
  - a preceding 'C Bo:* 8 >' completion (8-byte control/trigger)
  - the subsequent large 'C Bo:* N >' completion where N ~= expected off-chip bytes

Duration (microseconds) = ts_large - ts_8B. We collect these across the log and
report the average for each segment.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "five_models" / "results" / "theory_chain_source_data.csv"
USB_BASE = ROOT / "results" / "models_local_batch_usbmon" / "single"
OUT_CSV = ROOT / "results" / "offchip_usbmon_avg_times.csv"

SEG_RE = re.compile(r"^seg(\d+)$")


def parse_csv_targets(csv_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    seen: set[tuple[str, str]] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                model = r["model"].strip()
                group = r["group_name"].strip()
            except Exception:
                continue
            if not SEG_RE.match(group):
                # only single segments (segN), skip combined like seg2to8
                continue
            # Determine off-chip bytes from weights_stream_MiB
            try:
                ws = float(r.get("weights_stream_MiB", "0") or 0.0)
            except Exception:
                ws = 0.0
            if ws <= 0.0:
                continue
            # Some rows denote offchip segments with is_offchip flag
            is_off = None
            if "is_offchip" in r and r["is_offchip"] != "":
                try:
                    is_off = int(r["is_offchip"]) != 0
                except Exception:
                    is_off = None
            if is_off is False:
                continue
            expected_bytes = int(round(ws * 1024 * 1024))
            key = (model, group)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "model": model,
                "group": group,
                "expected_bytes": expected_bytes,
                "weights_stream_MiB": ws,
            })
    return rows


def iter_usbmon_bo_completions(file_path: Path):
    """Yield tuples (ts_us:int, bytes:int) for completed OUT URBs (C Bo:...).

    The timestamp field in logs may be in microseconds as integer; if a float
    with fractional part is present we convert to integer microseconds.
    """
    re_dir = re.compile(r"([CB][io]):(\d+):(\d+):(\d+)")
    with file_path.open("r", errors="ignore") as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 4:
                continue
            # Expect 'C' completion
            if parts[2] != "C":
                continue
            m = re_dir.search(ln)
            if not m:
                continue
            tok = m.group(1)  # Bi, Bo, Ci, Co
            if tok != "Bo":
                continue
            # timestamp
            ts_raw = parts[1]
            try:
                if "." in ts_raw:
                    ts_us = int(float(ts_raw) * 1_000_000.0)
                else:
                    ts_us = int(ts_raw)
            except Exception:
                continue
            # bytes – prefer len=, else token+2
            nbytes = 0
            mlen = re.search(r"len=(\d+)", ln)
            if mlen:
                try:
                    nbytes = int(mlen.group(1))
                except Exception:
                    nbytes = 0
            else:
                # find position of token in parts
                dir_idx: Optional[int] = None
                for i, tok_part in enumerate(parts):
                    if re.match(r"^[CB][io]:\d+:\d+:\d+", tok_part):
                        dir_idx = i
                        break
                if dir_idx is not None and dir_idx + 2 < len(parts):
                    try:
                        nbytes = int(parts[dir_idx + 2])
                    except Exception:
                        nbytes = 0
            yield ts_us, nbytes


def measure_offchip_durations(usb_file: Path, expected_bytes: int,
                              tol_bytes: int = 96 * 1024,
                              max_pair_gap_us: int = 1_000_000) -> Tuple[List[int], List[int]]:
    """Return (durations_us, matched_sizes) using single-chunk pairing.

    Pair the most recent 'C Bo ... 8 >' completion with the subsequent
    large 'C Bo ... N >' where N is within [expected_bytes - tol, expected_bytes + tol].
    Limit pairing to gaps <= max_pair_gap_us.
    Also collect matched N for simple diagnostics.
    """
    if not usb_file.is_file():
        return [], []
    lo = max(0, expected_bytes - tol_bytes)
    hi = expected_bytes + tol_bytes
    last_8_ts: Optional[int] = None
    durs: List[int] = []
    sizes: List[int] = []
    for ts_us, nbytes in iter_usbmon_bo_completions(usb_file):
        if nbytes == 8:
            last_8_ts = ts_us
            continue
        if lo <= nbytes <= hi:
            if last_8_ts is not None:
                gap = ts_us - last_8_ts
                if 0 < gap <= max_pair_gap_us:
                    durs.append(gap)
                    sizes.append(nbytes)
            # Do not reset last_8_ts here; a subsequent large packet may still
            # logically pair with the same 8B if they appear bursty per invoke.
            # However to avoid double-counting within the same invoke when there are
            # multiple large matches, we can reset once one is paired.
            last_8_ts = None
    return durs, sizes


def measure_offchip_durations_accum(usb_file: Path, expected_bytes: int,
                                    tol_bytes: int = 96 * 1024,
                                    max_span_us: int = 2_000_000) -> List[int]:
    """Return durations_us by accumulating multiple Bo completions after the 8B trigger
    until the sum of bytes reaches expected_bytes within tolerance.

    This handles devices that split the off-chip transfer into many chunks.
    """
    if not usb_file.is_file():
        return []
    lo = max(0, expected_bytes - tol_bytes)
    hi = expected_bytes + tol_bytes
    durations: List[int] = []
    collecting = False
    start_ts: Optional[int] = None
    acc = 0
    for ts_us, nbytes in iter_usbmon_bo_completions(usb_file):
        if nbytes == 8:
            # start a new collection window
            collecting = True
            start_ts = ts_us
            acc = 0
            continue
        if not collecting:
            continue
        # accumulate Bo completions
        acc += nbytes
        if start_ts is not None and (ts_us - start_ts) > max_span_us:
            # give up this window
            collecting = False
            start_ts = None
            acc = 0
            continue
        if lo <= acc <= hi:
            durations.append(ts_us - (start_ts or ts_us))
            collecting = False
            start_ts = None
            acc = 0
    return durations


def main() -> int:
    targets = parse_csv_targets(CSV_PATH)
    if not targets:
        print("No off-chip single segments found in", CSV_PATH)
        return 1

    results: List[Dict] = []
    # Exclude the two segments the user confirmed as 0 off-chip
    skip = {
        ('inceptionv3_8seg_uniform_local', 'seg7'),
        ('resnet101_8seg_uniform_local', 'seg7'),
    }
    for t in targets:
        model = t["model"]
        group = t["group"]
        if (model, group) in skip:
            continue
        exp = t["expected_bytes"]
        usb = USB_BASE / model / group / "usbmon.txt"
        durs_single, sizes = measure_offchip_durations(usb, exp)
        durs_accum = measure_offchip_durations_accum(usb, exp)
        if durs_single:
            avg_us_single = sum(durs_single) / float(len(durs_single))
        else:
            avg_us_single = 0.0
        if durs_accum:
            avg_us_accum = sum(durs_accum) / float(len(durs_accum))
        else:
            avg_us_accum = 0.0
        results.append({
            "model": model,
            "segment": group,
            "weights_stream_MiB": t["weights_stream_MiB"],
            "expected_bytes": exp,
            "matches_single": len(durs_single),
            "avg_offchip_ms_single": round(avg_us_single / 1000.0, 3),
            "matches_accum": len(durs_accum),
            "avg_offchip_ms_accum": round(avg_us_accum / 1000.0, 3),
            "matched_size_most": max(set(sizes), key=sizes.count) if sizes else 0,
        })

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "segment",
                "weights_stream_MiB",
                "expected_bytes",
                "matches_single",
                "avg_offchip_ms_single",
                "matches_accum",
                "avg_offchip_ms_accum",
                "matched_size_most",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow(r)

    # Print a compact summary for console
    print("Wrote:", OUT_CSV)
    for r in results:
        ms1 = r["avg_offchip_ms_single"]
        n1 = r["matches_single"]
        ms2 = r["avg_offchip_ms_accum"]
        n2 = r["matches_accum"]
        note = f"single {ms1}ms x{n1}; accum {ms2}ms x{n2}"
        print(f"{r['model']}/{r['segment']}: {note} (target ~{r['expected_bytes']}B, seen ~{r['matched_size_most']}B)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
