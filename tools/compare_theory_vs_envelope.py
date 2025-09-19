#!/usr/bin/env python3
import os, sys, json, csv
from pathlib import Path
from statistics import mean

BASE = Path('/home/10210/Desktop/OS')
CHAIN_ROOT = BASE/'results/models_local_combo_chain'
THEORY_CSV = BASE/'five_models/results/theory_chain_times.csv'

def load_theory_rows(model: str):
    rows = []
    if not THEORY_CSV.exists():
        return rows
    with THEORY_CSV.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r.get('model') == model and r.get('group_index') not in (None, '', 'TOTAL'):
                rows.append(r)
    return rows

def per_k_groups(rows):
    byk = {}
    for r in rows:
        try:
            K = int(r.get('K') or 0)
        except Exception:
            K = 0
        byk.setdefault(K, []).append(r)
    # sort by group_index where possible
    for K, lst in byk.items():
        lst.sort(key=lambda x: int(x.get('group_index') or 0))
    return byk

def run_bo_span(py: str, script: Path, k_dir: Path):
    import subprocess
    try:
        out = subprocess.check_output([py, str(script), str(k_dir)], text=True)
        return json.loads(out)
    except Exception:
        return {}

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else 'resnet101_8seg_uniform_local'
    chain_root = CHAIN_ROOT/model
    if not chain_root.exists():
        print('missing', chain_root)
        sys.exit(2)
    # ensure theory CSV exists (user should have run compute_theory_chain_times)
    thr = load_theory_rows(model)
    if not thr:
        print('No theory rows; run compute_theory_chain_times.py first')
        sys.exit(3)
    thr_by_k = per_k_groups(thr)
    py = sys.executable or 'python3'
    bo_script = (BASE/'tools/bo_envelope_span.py').resolve()
    report = {}
    for Kdir in sorted(chain_root.glob('K*')):
        if not Kdir.is_dir():
            continue
        # simple K value
        try:
            K = int(Kdir.name[1:])
        except Exception:
            continue
        if (Kdir/'usbmon.txt').exists() and (Kdir/'time_map.json').exists() and (Kdir/'merged_invokes.json').exists():
            measured = run_bo_span(py, bo_script, Kdir)
        else:
            continue
        # theory for this K
        gro = thr_by_k.get(K) or []
        # Build map seg_label -> per-invoke span_s list
        # bo_envelope_span returns per-seg aggregate only; we need average span seconds and count
        comp = []
        for idx, r in enumerate(gro, start=1):
            gname = r.get('group_name') or ''
            segs = [s for s in (r.get('group_segs') or '').split(',') if s]
            # measured: for groups like segXto8，我们用最后一段标签（例如 seg8）来匹配窗口；单段就用自身
            if 'to8' in gname:
                key = segs[-1] if segs else None
            else:
                key = gname if gname else (segs[0] if segs else None)
            m = measured.get(key or '', {})
            count = int(m.get('count') or 0)
            avg_span_s = float(m.get('avg_span_s') or 0.0)  # seconds
            # convert to ms
            span_ms = avg_span_s * 1000.0
            # theory breakdown
            def tf(x):
                try:
                    return float(x)
                except Exception:
                    return 0.0
            Wi_lb_host_ms = tf(r.get('Wi_lb_ms_hosted'))
            Cin_ms = tf(r.get('Cin_ms'))
            Cout_ms = tf(r.get('Cout_ms'))
            Ce_ms = tf(r.get('Ce_ms'))
            t_warm_ms = tf(r.get('t_warm_ms'))
            t_rem_lb_ms = tf(r.get('t_rem_lb_ms'))
            Th_ms = tf(r.get('Th_ms'))
            mismatch = Wi_lb_host_ms - span_ms
            comp.append({
                'K': K,
                'group_index': idx,
                'group_name': gname,
                'match_key': key,
                'measured_span_ms': span_ms,
                'theory_Wi_lb_host_ms': Wi_lb_host_ms,
                'mismatch_ms': mismatch,
                'Cin_ms': Cin_ms,
                'Cout_ms': Cout_ms,
                'Ce_ms': Ce_ms,
                't_warm_ms': t_warm_ms,
                't_rem_lb_ms': t_rem_lb_ms,
                'Th_ms': Th_ms,
                'measured_count': count,
            })
        report[K] = comp
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
