#!/usr/bin/env python3
import os, sys, json, re

def main():
    if len(sys.argv) < 4:
        print('usage: extract_single_capture_snippet.py <capture_out_dir> <pre_ms> <post_ms>')
        sys.exit(2)
    out_dir = sys.argv[1]
    pre_s = float(sys.argv[2]) / 1000.0
    post_s = float(sys.argv[3]) / 1000.0
    usb = os.path.join(out_dir, 'usbmon.txt')
    tm = os.path.join(out_dir, 'time_map.json')
    inv = os.path.join(out_dir, 'invokes.json')
    assert os.path.isfile(usb) and os.path.isfile(tm) and os.path.isfile(inv)
    spans = json.load(open(inv)).get('spans', [])
    if not spans:
        print('no spans in invokes.json'); sys.exit(1)
    w = spans[0]
    tmj = json.load(open(tm))
    usb_ref = tmj.get('usbmon_ref'); bt_ref = tmj.get('boottime_ref')
    if usb_ref is None or bt_ref is None:
        print('time_map missing refs'); sys.exit(1)
    b0 = (w['begin'] - bt_ref) + usb_ref
    e0 = (w['end']   - bt_ref) + usb_ref
    sb = b0 - pre_s
    se = e0 + post_s
    # 提取原始片段
    lines_all = []
    lines_bibo = []
    with open(usb, 'r', errors='ignore') as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 2:
                continue
            ts = None
            try:
                ts = float(parts[1])
            except Exception:
                try:
                    ts = float(parts[0])
                except Exception:
                    continue
            ts = ts/1e6 if ts and ts>1e6 else ts
            if ts is None or ts < sb or ts > se:
                continue
            lines_all.append(ln.rstrip('\n'))
            if re.search(r'\b[BC][io]:', ln):
                if re.search(r'\bBi:|\bBo:', ln):
                    lines_bibo.append(ln.rstrip('\n'))
    snip = os.path.join(out_dir, 'single_invoke_snippet.txt')
    snip_bibo = os.path.join(out_dir, 'single_invoke_snippet_BiBo.txt')
    with open(snip, 'w') as f:
        f.write('\n'.join(lines_all))
    with open(snip_bibo, 'w') as f:
        f.write('\n'.join(lines_bibo))
    print(json.dumps({
        'invoke_ms': (w['end']-w['begin'])*1000.0,
        'mapped_window_s': [b0, e0],
        'snippet_window_s': [sb, se],
        'snippet_path': snip,
        'snippet_bibo_path': snip_bibo,
        'lines_all': len(lines_all),
        'lines_bibo': len(lines_bibo)
    }, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()



