#!/usr/bin/env python3
import argparse
import datetime
import glob
import os
import re
from pathlib import Path


def parse_off_used_from_logs(log_paths):
    unit_to_mib = {"B": 1.0 / 1024 / 1024, "KiB": 1.0 / 1024, "MiB": 1.0}
    rx_line = re.compile(r"Off-chip memory used.*?:\s*([0-9]+(?:\.[0-9]+)?)(B|KiB|MiB)")
    rx_k = re.compile(r"/combos_K(\d+)_run1/")
    # filenames could be like: 20250905-190049_tail_seg2_to_8.log or 20250905-185923_seg1.log or seg1.log
    rx_tail_name = re.compile(r"(?:^|_)tail_seg(\d+)_to_(\d+)\.log$")
    rx_seg_name = re.compile(r"(?:^|_)(seg\d+)\.log$")

    result = {}
    for p in log_paths:
        mk = rx_k.search(p)
        if not mk:
            continue
        k_val = int(mk.group(1))
        try:
            model = p.split("/models_local/public/")[1].split("/")[0]
        except Exception:
            continue

        fname = os.path.basename(p)
        label = None
        mt = rx_tail_name.search(fname)
        if mt:
            label = f"seg{int(mt.group(1))}to{int(mt.group(2))}"
        else:
            ms = rx_seg_name.search(fname)
            if ms:
                label = ms.group(1)
        if not label:
            continue

        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
        except Exception:
            continue

        ml = rx_line.search(txt)
        if not ml:
            continue
        val = float(ml.group(1))
        unit = ml.group(2)
        mib = round(val * unit_to_mib[unit] + 1e-12, 3)

        bucket = result.setdefault(model, {}).setdefault(k_val, {})
        # keep max if multiple logs exist per label
        prev = bucket.get(label)
        if prev is None or mib > prev:
            bucket[label] = mib

    return result


def sorted_label_key(lbl: str):
    if "to" in lbl:
        a, b = lbl[3:].split("to")
        return (int(a), 1, int(b))
    return (int(lbl[3:]), 0, -1)


def format_append_lines(data):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"# combos off_used extracted at {now}\n"]
    for model in sorted(data.keys()):
        for k in sorted(data[model].keys()):
            lines.append(f"{model} combos_K{k} MIB\n")
            for lbl in sorted(data[model][k].keys(), key=sorted_label_key):
                val = data[model][k][lbl]
                lines.append(f"{lbl}: off_used {val:.3f}\n")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Extract off_used (MiB) from combo logs and append to models_compilation.txt")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="Project root (default: two levels up from this script)")
    parser.add_argument("--append", action="store_true", help="Append to baselines/models_compilation.txt instead of printing to stdout")
    args = parser.parse_args()

    root = Path(args.root)
    log_glob = str(root / "models_local/public/*/combos_K*_run1/logs/*.log")
    log_paths = sorted(glob.glob(log_glob))

    data = parse_off_used_from_logs(log_paths)
    if not data:
        print("No off_used found from logs", flush=True)
        return 2

    lines = format_append_lines(data)

    if args.append:
        out_path = root / "baselines/models_compilation.txt"
        with open(out_path, "a", encoding="utf-8") as f:
            f.writelines(lines)
        # echo a short preview
        print(f"Appended {len(lines)} lines to {out_path}")
        print("Preview:\n" + "".join(lines[:20]))
        return 0
    else:
        print("".join(lines), end="")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())


