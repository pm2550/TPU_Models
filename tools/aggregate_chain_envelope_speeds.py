#!/usr/bin/env python3
import os, json, sys, subprocess, shlex
from pathlib import Path

MiB = 1024.0*1024.0

def run_bo_span(k_dir: Path, py: str, script: Path):
    try:
        out = subprocess.check_output([py, str(script), str(k_dir)], text=True)
        return json.loads(out)
    except Exception as e:
        return {}

def main():
    if len(sys.argv) < 2:
        print("Usage: aggregate_chain_envelope_speeds.py <combo_chain_root>")
        sys.exit(1)
    root = Path(sys.argv[1]).resolve()
    py = sys.executable or 'python3'
    script = (Path(__file__).parent/'bo_envelope_span.py').resolve()
    # Target models
    targets = [
        'inceptionv3_8seg_uniform_local',
        'resnet50_8seg_uniform_local',
        'resnet101_8seg_uniform_local',
    ]
    result = {}
    for model in targets:
        mdir = root/model
        if not mdir.exists():
            continue
        # Pick K* dirs that contain usbmon.txt/time_map.json/merged_invokes.json
        kdirs = []
        for p in sorted(mdir.glob('K*')):
            if not p.is_dir():
                continue
            if (p/'usbmon.txt').exists() and (p/'time_map.json').exists() and (p/'merged_invokes.json').exists():
                kdirs.append(p)
        # Aggregate per seg
        agg = {}
        for kd in kdirs:
            data = run_bo_span(kd, py, script)
            for seg, stats in data.items():
                a = agg.setdefault(seg, {
                    'total_bytes': 0.0,
                    'total_span': 0.0,
                    'total_count': 0,
                    'min_speed': float('inf'),
                    'max_speed': 0.0,
                })
                cnt = int(stats.get('count') or 0)
                avg_b = float(stats.get('avg_bytes_out') or 0.0)
                avg_s = float(stats.get('avg_span_s') or 0.0)
                # total bytes/spans across spans in this K
                a['total_bytes'] += avg_b * cnt
                a['total_span'] += avg_s * cnt
                a['total_count'] += cnt
                # global min/max across Ks from per-K min/max
                ms = float(stats.get('min_speed_MiBps') or 0.0)
                mx = float(stats.get('max_speed_MiBps') or 0.0)
                if ms < a['min_speed']:
                    a['min_speed'] = ms
                if mx > a['max_speed']:
                    a['max_speed'] = mx
        # finalize per seg
        outm = {}
        for seg, a in agg.items():
            avg_speed = (a['total_bytes']/MiB)/a['total_span'] if a['total_span']>0 else 0.0
            # If we never updated min_speed, set to 0
            if a['min_speed'] == float('inf'):
                a['min_speed'] = 0.0
            outm[seg] = {
                'count_total': a['total_count'],
                'avg_speed_MiBps': avg_speed,
                'min_speed_MiBps': a['min_speed'],
                'max_speed_MiBps': a['max_speed'],
            }
        result[model] = outm
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
