#!/usr/bin/env python3
import os
import csv
from pathlib import Path
import re


ROOT = Path('/home/10210/Desktop/OS')
PUBLIC = ROOT / 'models_local/public'
OUT_CSV = ROOT / 'results/model_file_sizes.csv'

MODELS = [
    'densenet201_8seg_uniform_local',
    'inceptionv3_8seg_uniform_local',
    'resnet101_8seg_uniform_local',
    'resnet50_8seg_uniform_local',
    'xception_8seg_uniform_local',
]


def seg_label_from_name(filename: str) -> str:
    bn = os.path.basename(filename)
    m = re.match(r"seg(\d+)_", bn)
    if m:
        return f"seg{int(m.group(1))}"
    m = re.match(r"tail_seg(\d+)_?to_?(\d+)_", bn)
    if m:
        return f"seg{int(m.group(1))}to{int(m.group(2))}"
    return 'seg'


def main():
    rows = []
    for model in MODELS:
        base = PUBLIC / model
        # single segments
        single_tpu = base / 'full_split_pipeline_local' / 'tpu'
        if single_tpu.is_dir():
            for p in sorted(single_tpu.glob('*_edgetpu.tflite')):
                size_b = p.stat().st_size
                rows.append([
                    model, 'single', '', seg_label_from_name(p.name), p.name, size_b, f"{size_b/(1024*1024):.3f}", str(p)
                ])

        # combos K2..K7
        for k in range(2, 8):
            combo_tpu = base / f'combos_K{k}_run1' / 'tpu'
            if not combo_tpu.is_dir():
                continue
            for p in sorted(combo_tpu.glob('*_edgetpu.tflite')):
                size_b = p.stat().st_size
                rows.append([
                    model, 'combo', str(k), seg_label_from_name(p.name), p.name, size_b, f"{size_b/(1024*1024):.3f}", str(p)
                ])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'type', 'K', 'segment_label', 'file_name', 'size_bytes', 'size_MiB', 'path'])
        w.writerows(rows)
    print(f'saved: {OUT_CSV}, rows={len(rows)}')


if __name__ == '__main__':
    main()


