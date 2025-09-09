#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict


ROOT = Path('/home/10210/Desktop/OS').resolve()
RUN_USBMON_INVOKE = ROOT / 'run_usbmon_invoke.sh'
ANALYZE_OUT = ROOT / 'analyze_usbmon_out_active.py'
ANALYZE_IN = ROOT / 'analyze_usbmon_in_active.py'
SUMMARIZE = ROOT / 'summarize_warm_usbmon.py'
PW_FILE = ROOT / 'password.text'


@dataclass
class ModelItem:
    path: Path
    name: str
    size_bytes: int


def is_edgetpu_model(p: Path) -> bool:
    try:
        with p.open('rb') as f:
            head = f.read(1024 * 1024)
        return b'edgetpu-custom-op' in head
    except Exception:
        return False


def discover_models(max_total: int) -> List[ModelItem]:
    # Candidate directories to scan
    cand_dirs = [
        ROOT / 'onlyCo',
        ROOT / 'segment_models',
        ROOT / 'layered models' / 'mn' / 'tpu',
        ROOT / 'layered models' / 'enhanced tpu',
        ROOT / 'model' / 'large_resnet50_split' / 'tpu',
        ROOT / 'tpu',
        ROOT / 'edgetpu' / 'test_data' / 'tools',
    ]
    items: List[ModelItem] = []
    seen: set = set()
    for d in cand_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob('*.tflite')):
            if len(items) >= max_total:
                break
            if p in seen:
                continue
            if not is_edgetpu_model(p):
                continue
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            name = p.stem
            items.append(ModelItem(path=p, name=name, size_bytes=size))
            seen.add(p)
    return items


def run_one(model: ModelItem, out_root: Path, bus: str) -> Tuple[float, float, Path]:
    out_dir = out_root / model.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Capture usbmon + invokes + time_map
    cmd = [str(RUN_USBMON_INVOKE), str(model.path), model.name, str(out_dir), bus]
    print(f"[CAPTURE] {model.name} -> {out_dir}")
    subprocess.run(cmd, check=True)

    usb_txt = out_dir / 'usbmon.txt'
    invokes = out_dir / 'invokes.json'
    time_map = out_dir / 'time_map.json'

    # 2) Analyze out/in active unions
    out_union = out_dir / 'out_active_union.json'
    in_union = out_dir / 'in_active_union.json'
    print(f"[ANALYZE OUT] {out_union}")
    with out_union.open('w') as fo:
        subprocess.run([str(sys.executable), str(ANALYZE_OUT), str(usb_txt), str(invokes), str(time_map)], check=True, stdout=fo)
    print(f"[ANALYZE IN ] {in_union}")
    with in_union.open('w') as fo:
        subprocess.run([str(sys.executable), str(ANALYZE_IN), str(usb_txt), str(invokes), str(time_map)], check=True, stdout=fo)

    # 3) Summarize warm invokes (skip first)
    warm_json = out_dir / 'warm_summary.json'
    print(f"[SUMMARIZE ] {warm_json}")
    with warm_json.open('w') as fo:
        subprocess.run([str(sys.executable), str(SUMMARIZE), str(out_union), str(in_union)], check=True, stdout=fo)

    # 读取结果
    j = json.loads(warm_json.read_text())
    return float(j.get('out_overall_MBps', 0.0)), float(j.get('in_overall_MBps', 0.0)), warm_json


def main():
    parser = argparse.ArgumentParser(description='批量使用 usbmon 采集并计算 TPU<->CPU IN/OUT 带宽')
    parser.add_argument('--bus', default='2', help='usbmon 总线号 (对应 /sys/kernel/debug/usb/usbmon/<BUS>u)')
    parser.add_argument('--max', type=int, default=16, help='最多选择多少个模型')
    parser.add_argument('--out', default=str(ROOT / 'results' / 'usbmon_bulk'), help='输出根目录')
    args = parser.parse_args()

    if not PW_FILE.exists():
        print(f"[ERROR] 缺少 sudo 密码文件: {PW_FILE}", file=sys.stderr)
        sys.exit(1)
    if not RUN_USBMON_INVOKE.exists():
        print(f"[ERROR] 缺少脚本: {RUN_USBMON_INVOKE}", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    models = discover_models(args.max)
    if not models:
        print("[ERROR] 未发现 EdgeTPU 可用模型", file=sys.stderr)
        sys.exit(2)

    rows: List[Dict] = []
    for i, m in enumerate(models, 1):
        print(f"===== [{i}/{len(models)}] {m.name} =====")
        try:
            out_mbps, in_mbps, warm_path = run_one(m, out_root, args.bus)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] 采集/分析失败: {m.name} ({e})", file=sys.stderr)
            continue
        rows.append({
            'name': m.name,
            'path': str(m.path),
            'size_bytes': m.size_bytes,
            'out_overall_MBps': out_mbps,
            'in_overall_MBps': in_mbps,
            'warm_summary': str(warm_path),
        })

    # 保存汇总
    summary_json = out_root / 'summary.json'
    summary_csv = out_root / 'summary.csv'
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    with summary_csv.open('w') as fo:
        fo.write('name,size_bytes,out_overall_MBps,in_overall_MBps,path\n')
        for r in rows:
            fo.write(f"{r['name']},{r['size_bytes']},{r['out_overall_MBps']:.6f},{r['in_overall_MBps']:.6f},{r['path']}\n")

    # 打印总体均值
    if rows:
        avg_out = sum(r['out_overall_MBps'] for r in rows) / len(rows)
        avg_in = sum(r['in_overall_MBps'] for r in rows) / len(rows)
        print(json.dumps({
            'num_models': len(rows),
            'avg_out_overall_MBps': avg_out,
            'avg_in_overall_MBps': avg_in,
            'summary_json': str(summary_json),
            'summary_csv': str(summary_csv),
        }, ensure_ascii=False))
    else:
        print('[WARN] 没有成功的记录')


if __name__ == '__main__':
    main()


