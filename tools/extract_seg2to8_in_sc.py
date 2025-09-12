#!/usr/bin/env python3
import sys
import json
import re
from pathlib import Path


def load_time_refs(base: Path):
    tm = json.load(open(base / 'time_map.json'))
    usb_ref = tm.get('usbmon_ref')
    bt_ref = tm.get('boottime_ref')
    if usb_ref is None or bt_ref is None:
        raise RuntimeError('time_map.json 缺少 usbmon_ref/boottime_ref')
    return float(usb_ref), float(bt_ref)


def load_seg_window(base: Path):
    merged = json.load(open(base / 'merged_invokes.json'))
    spans = merged.get('spans') or []
    if not spans:
        raise RuntimeError('merged_invokes.json 无 spans')
    # 优先找 seg2to8；找不到则取第二个
    idx = None
    for i, sp in enumerate(spans):
        if str(sp.get('seg_label', '')).startswith('seg2to8'):
            idx = i
            break
    if idx is None:
        idx = min(1, len(spans) - 1)
    return spans[idx]


def parse_usb_ts(tok: str) -> float:
    # usbmon 第二列可能是秒或微秒
    ts = float(tok)
    return ts / 1e6 if ts > 1e6 else ts


def extract_in_sc(cap_path: Path, seg_start_usb: float, seg_end_usb: float):
    # 匹配 S/C Bi 行
    pat_s = re.compile(r"^\s*(\S+)\s+([0-9]+(?:\.[0-9]+)?)\s+.*?\sS\s+Bi:\d+:\d+:\d+")
    pat_c = re.compile(r"^\s*(\S+)\s+([0-9]+(?:\.[0-9]+)?)\s+.*?\sC\s+Bi:\d+:\d+:\d+\s+.*?(?:len=(\d+)|\s(\d+))")

    s_map = {}
    c_list = []  # (urb, t_c, size)

    with open(cap_path, 'r', errors='ignore') as f:
        for ln in f:
            m = pat_s.match(ln)
            if m:
                urb = m.group(1)
                ts = parse_usb_ts(m.group(2))
                s_map[urb] = ts
                continue
            m = pat_c.match(ln)
            if m:
                urb = m.group(1)
                ts = parse_usb_ts(m.group(2))
                size = None
                if m.group(3) and m.group(3).isdigit():
                    size = int(m.group(3))
                elif m.group(4) and m.group(4).isdigit():
                    size = int(m.group(4))
                if size is not None and size in (1000, 1024):
                    c_list.append((urb, ts, size))

    if not c_list:
        return None

    # 选择完成时刻最接近 seg 结束的那个 1000/1024B IN 包
    urb, t_c, size = min(c_list, key=lambda x: abs(x[1] - seg_end_usb))
    t_s = s_map.get(urb)
    return (t_s, t_c, size)


def main():
    # base 目录（包含 usbmon.txt/merged_invokes.json/time_map.json），默认 K2
    base_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results/models_local_combo_chain/densenet201_8seg_uniform_local/K2')
    base = base_arg.resolve()

    usb_ref, bt_ref = load_time_refs(base)
    seg_span = load_seg_window(base)
    seg_start_usb = (float(seg_span['begin']) - bt_ref) + usb_ref
    seg_end_usb = (float(seg_span['end']) - bt_ref) + usb_ref

    cap_path = base / 'usbmon.txt'
    if not cap_path.exists():
        print('IN_START_REL_MS: NA')
        print('IN_END_REL_MS: NA')
        print('IN_DURATION_MS: NA')
        return

    sc = extract_in_sc(cap_path, seg_start_usb, seg_end_usb)
    if sc is None:
        print('IN_START_REL_MS: NA')
        print('IN_END_REL_MS: NA')
        print('IN_DURATION_MS: NA')
        return

    t_s, t_c, size = sc
    end_ms = (t_c - seg_start_usb) * 1000.0
    if t_s is None:
        print('IN_START_REL_MS: NA')
        print(f'IN_END_REL_MS: {end_ms:.3f}')
        print('IN_DURATION_MS: NA')
    else:
        start_ms = (t_s - seg_start_usb) * 1000.0
        dur_ms = (t_c - t_s) * 1000.0
        print(f'IN_START_REL_MS: {start_ms:.3f}')
        print(f'IN_END_REL_MS: {end_ms:.3f}')
        print(f'IN_DURATION_MS: {dur_ms:.3f}')


if __name__ == '__main__':
    main()


