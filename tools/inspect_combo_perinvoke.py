#!/usr/bin/env python3
import os
import sys
import re
import json
import statistics as st

def load_json_per_invoke(stdout_text: str):
    m = re.search(r"JSON_PER_INVOKE:\s*(\[.*?\])", stdout_text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

def avg(values):
    return st.mean(values) if values else 0.0

def main():
    if len(sys.argv) < 2:
        print("usage: inspect_combo_perinvoke.py <combo_root_dir>")
        sys.exit(2)
    combo_root = sys.argv[1]
    stdout_path = os.path.join(combo_root, 'correct_per_invoke_stdout.txt')
    merged_path = os.path.join(combo_root, 'merged_invokes.json')
    if not (os.path.isfile(stdout_path) and os.path.isfile(merged_path)):
        print("missing files:")
        print(stdout_path if os.path.isfile(stdout_path) else '(no stdout)')
        print(merged_path if os.path.isfile(merged_path) else '(no merged)')
        sys.exit(1)
    txt = open(stdout_path, 'r', errors='ignore').read()
    arr = load_json_per_invoke(txt)
    if not isinstance(arr, list):
        print("no JSON_PER_INVOKE in stdout")
        sys.exit(1)
    merged = json.load(open(merged_path))
    spans = merged.get('spans', [])
    # 建立 seg -> 索引列表
    seg_to_idxs = {}
    for i, sp in enumerate(spans):
        lbl = sp.get('seg_label')
        seg_to_idxs.setdefault(lbl, []).append(i)

    print("segment, num, avg_in_bytes, avg_out_bytes, avg_in_ms, avg_out_ms, avg_union_ms, avg_overlap_ms, coverage(overlap/in)")
    for seg, idxs in sorted(seg_to_idxs.items(), key=lambda kv: kv[0]):
        xs = []
        for i in idxs:
            if 0 <= i < len(arr):
                xs.append(arr[i])
        xs = [x for x in xs if (x.get('bytes_in', 0)>0 or x.get('bytes_out', 0)>0)]
        if not xs:
            continue
        in_b = avg([float(x.get('bytes_in', 0.0) or 0.0) for x in xs])
        out_b = avg([float(x.get('bytes_out', 0.0) or 0.0) for x in xs])
        in_s = avg([float(x.get('in_active_s', 0.0) or 0.0) for x in xs])
        out_s = avg([float(x.get('out_active_s', 0.0) or 0.0) for x in xs])
        uni_s = avg([float((x.get('union_active_s', None) if x.get('union_active_s', None) is not None else x.get('union_active_span_s', 0.0)) or 0.0) for x in xs])
        ovl_s = avg([float(x.get('overlap_active_s', 0.0) or 0.0) for x in xs])
        cov = (ovl_s / in_s) if in_s > 0 else 0.0
        print(
            f"{seg}, {len(xs)}, {int(in_b)}, {int(out_b)}, {in_s*1000:.2f}, {out_s*1000:.2f}, {uni_s*1000:.2f}, {ovl_s*1000:.2f}, {cov:.2f}"
        )

if __name__ == '__main__':
    main()



