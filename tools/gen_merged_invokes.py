#!/usr/bin/env python3
import os
import sys
import json

def main():
    if len(sys.argv) < 2:
        print("usage: gen_merged_invokes.py <combo_root>")
        sys.exit(1)
    root = sys.argv[1]
    spans = []
    if not os.path.isdir(root):
        print(f"not a dir: {root}")
        sys.exit(1)
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not (os.path.isdir(p) and name.startswith('seg')):
            continue
        iv = os.path.join(p, 'invokes.json')
        if not os.path.exists(iv):
            continue
        try:
            J = json.load(open(iv))
        except Exception:
            continue
        for s in (J.get('spans') or []):
            spans.append({'begin': s['begin'], 'end': s['end'], 'seg_label': name})
    spans.sort(key=lambda x: x['begin'])
    outp = os.path.join(root, 'merged_invokes.json')
    json.dump({'spans': spans}, open(outp, 'w'))
    print(f"merged_invokes.json written: {len(spans)} spans -> {outp}")

if __name__ == '__main__':
    main()


