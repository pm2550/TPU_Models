#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path


def parse_models_compilation_for_combos(path: Path):
    combos = {}
    current_model = None
    current_k = None
    rx_header = re.compile(r"^(.+?)\s+combos_K(\d+)\s+MIB\s*$")
    rx_item = re.compile(r"^(seg\d+(?:to\d+)?):\s*off_used\s+([0-9]+(?:\.[0-9]+)?)\s*$")
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            mh = rx_header.match(line)
            if mh:
                current_model = mh.group(1)
                current_k = int(mh.group(2))
                combos.setdefault(current_model, {}).setdefault(current_k, {})
                continue
            mi = rx_item.match(line)
            if mi and current_model is not None and current_k is not None:
                label = mi.group(1)
                off_mib = float(mi.group(2))
                combos[current_model][current_k][label] = off_mib
    return combos


def build_combo_theory_io(baseline_json: Path, models_compilation_txt: Path):
    with open(baseline_json, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    combos = parse_models_compilation_for_combos(models_compilation_txt)

    out = {}
    for model, k_map in combos.items():
        if model not in baseline:
            # skip unknown models
            continue
        segs = baseline[model].get("segments", {})
        out.setdefault(model, {}).setdefault("combos", {})
        for k, seg_off in sorted(k_map.items()):
            k_key = f"K{k}"
            out[model]["combos"].setdefault(k_key, {})
            for label, off_mib in seg_off.items():
                if "to" in label:
                    # segXtoY
                    body = label[3:]
                    start_str, end_str = body.split("to", 1)
                    start_seg = f"seg{int(start_str)}"
                    end_seg = f"seg{int(end_str)}"
                    in_bytes = segs.get(start_seg, {}).get("input", {}).get("bytes")
                    out_bytes = segs.get(end_seg, {}).get("output", {}).get("bytes")
                else:
                    in_bytes = segs.get(label, {}).get("input", {}).get("bytes")
                    out_bytes = segs.get(label, {}).get("output", {}).get("bytes")
                if in_bytes is None or out_bytes is None:
                    continue
                off_bytes = int(round(off_mib * 1024 * 1024))
                out_entry = {
                    "theory_OUT_bytes": int(in_bytes) + off_bytes,
                    "theory_IN_bytes": int(out_bytes),
                    "off_used_MiB": round(off_mib, 3),
                    "base_input_bytes": int(in_bytes),
                    "base_output_bytes": int(out_bytes),
                }
                out[model]["combos"][k_key][label] = out_entry
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate theoretical IO for combo segments (OUT adds off_used)")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    baseline_json = root / "baselines/models_io_baseline_correct.json"
    models_compilation_txt = root / "baselines/models_compilation.txt"

    data = build_combo_theory_io(baseline_json, models_compilation_txt)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()


