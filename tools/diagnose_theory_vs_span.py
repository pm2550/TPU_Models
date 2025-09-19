#!/usr/bin/env python3
import csv, json, os, sys
from pathlib import Path

BASE = Path('/home/10210/Desktop/OS')
THEORY_CSV = BASE/'five_models/results/theory_chain_times.csv'
SPAN_SUMMARY = BASE/'results/models_local_batch_usbmon/single/combined_summary_span.json'

# Mirror constants used in compute_theory_chain_times.py
KAPPA_MS_PER_MS = 0.2992134815732149
HOST_C_MS = 0.5527747236073199

def expand_group_to_segments(group_name: str):
    g = group_name.strip()
    if 'to8' in g:
        try:
            start = int(g.replace('seg','').replace('to8',''))
        except Exception:
            return []
        return [f'seg{i}' for i in range(start, 9)]
    return [g]

def load_spans():
    if not SPAN_SUMMARY.exists():
        return {}
    try:
        J = json.loads(SPAN_SUMMARY.read_text())
    except Exception:
        return {}
    out = {}
    for m, segs in (J or {}).items():
        out[m] = {}
        for seg, sd in (segs or {}).items():
            try:
                out[m][seg] = float(sd.get('in_span_ms_mean') or 0.0)
            except Exception:
                out[m][seg] = 0.0
    return out

def diagnose(model: str):
    spans = load_spans()
    # Read theory rows for this model (exclude TOTAL)
    rows = []
    with THEORY_CSV.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r.get('model') != model:
                continue
            if str(r.get('group_index')) == 'TOTAL':
                continue
            rows.append(r)
    out = []
    for r in rows:
        K = int(r['K']) if r.get('K') else 0
        gname = r.get('group_name')
        gsegs = [s.strip() for s in (r.get('group_segs') or '').split(',') if s.strip() and s.strip() != '-']
        if not gsegs:
            gsegs = expand_group_to_segments(gname or '')
        # Gather spans per policy
        seg_map = spans.get(model, {})
        sum_span = sum(seg_map.get(seg, 0.0) for seg in gsegs)
        last_span = seg_map.get(gsegs[-1], 0.0) if gsegs else 0.0
        use_last_only = ('to8' in (gname or ''))
        used_span = last_span if use_last_only else sum_span

        # Parse theory components
        def fget(k):
            try:
                v = r.get(k)
                return float(v) if (v is not None and v != '') else 0.0
            except Exception:
                return 0.0
        Cin = fget('Cin_ms')
        Cout = fget('Cout_ms')
        Ce = fget('Ce_ms')
        tw = fget('t_warm_ms')
        trem = fget('t_rem_lb_ms')
        Th = fget('Th_ms')  # stored as delta_host_ms
        Wi_lb = fget('Wi_lb_ms')
        Wi_lb_hosted = fget('Wi_lb_ms_hosted')

        # Re-compute host delta based on spans and alternative policy
        host_used = HOST_C_MS + KAPPA_MS_PER_MS * used_span
        host_sum = HOST_C_MS + KAPPA_MS_PER_MS * sum_span
        host_last = HOST_C_MS + KAPPA_MS_PER_MS * last_span

        # Assemble diagnostics
        out.append({
            'K': K,
            'group': gname,
            'segs': gsegs,
            'in_span_ms': {
                'sum': sum_span,
                'last': last_span,
                'used': used_span,
            },
            'theory_ms': {
                'Cin': Cin,
                'Cout': Cout,
                'Ce': Ce,
                't_warm': tw,
                't_rem_lb': trem,
                'host_delta_csv': Th,
                'host_delta_used': host_used,
                'host_delta_sum': host_sum,
                'host_delta_last': host_last,
                'Wi_lb': Wi_lb,
                'Wi_lb_hosted': Wi_lb_hosted,
            },
            'host_delta_diff_ms': Th - host_used,
            'component_shares': {
                'Cin%': (Cin / Wi_lb_hosted * 100.0) if Wi_lb_hosted>0 else 0.0,
                'Cout%': (Cout / Wi_lb_hosted * 100.0) if Wi_lb_hosted>0 else 0.0,
                'Ce%': (Ce / Wi_lb_hosted * 100.0) if Wi_lb_hosted>0 else 0.0,
                't_warm%': (tw / Wi_lb_hosted * 100.0) if Wi_lb_hosted>0 else 0.0,
                't_rem_lb%': (trem / Wi_lb_hosted * 100.0) if Wi_lb_hosted>0 else 0.0,
                'host%': (Th / Wi_lb_hosted * 100.0) if Wi_lb_hosted>0 else 0.0,
            }
        })

    # Sort by K then group name for readability
    out.sort(key=lambda x: (x['K'], str(x['group'])))
    return out

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else 'resnet101_8seg_uniform_local'
    diag = diagnose(model)
    print(json.dumps({'model': model, 'rows': diag}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
