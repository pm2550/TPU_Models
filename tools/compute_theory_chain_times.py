#!/usr/bin/env python3
import csv
import json
from pathlib import Path

BASE = Path('/home/10210/Desktop/OS')
# Use combo-specific theory tensors and weight splits (on-chip/off-chip) per K and group
THEORY_COMBOS = BASE/'five_models/baselines/theory_io_combos.json'
THEORY_SEG = BASE/'five_models/baselines/theory_io_seg.json'
PURE_CSV = BASE/'five_models/results/single_pure_invoke_times.csv'
OUT_CSV = BASE/'five_models/results/theory_chain_times.csv'

# Config: effective link bandwidths (MiB/s)
# Per user's latest instruction: B_in = 320 (h2d), B_out = 60 (d2h)
B_IN = 320.0
B_OUT = 60.0
EPS_MS = 0.0   # small overhead per segment (ignored by default)

MODELS = [
    'densenet201_8seg_uniform_local',
    'inceptionv3_8seg_uniform_local',
    'resnet101_8seg_uniform_local',
    'resnet50_8seg_uniform_local',
    'xception_8seg_uniform_local',
]

def read_combos():
        """Read combo-specific theory definitions.
        Structure: {model: {combos: {K2: {groupName: {fields...}}, K3: {...}, ...}}}
        We will use fields:
            - theory_IN_bytes (group input bytes)
            - after warm up out_bytes (group output bytes)
            - segment_model_MiB (total model bytes for the group)
            - off_used_MiB (off-chip bytes to be streamed for the group)
        """
        J = json.loads(THEORY_COMBOS.read_text())
        C = {}
        for m, obj in J.items():
                C[m] = obj.get('combos', {})
        return C

def read_segments():
    """Read per-segment theory definitions for K=8 case.
    Structure: {model: {segments: {segX: {fields...}}}}
    We will use fields similar to combos groups:
      - base_input_bytes, base_output_bytes
      - segment_model_MiB, off_used_MiB (fallback to off_chip_MiB if missing)
    """
    J = json.loads(THEORY_SEG.read_text())
    S = {}
    for m, obj in J.items():
        segs = obj.get('segments', {})
        # normalize
        norm = {}
        for name, gd in segs.items():
            norm[name] = {
                'base_input_bytes': gd.get('base_input_bytes', 0.0),
                'base_output_bytes': gd.get('base_output_bytes', 0.0),
                'segment_model_MiB': gd.get('segment_model_MiB', 0.0),
                'off_used_MiB': gd.get('off_used_MiB', gd.get('off_chip_MiB', 0.0)),
            }
        S[m] = norm
    return S

def read_pure_times():
    # CSV columns expected: model, segment, type, count, pure_ms_final, pure_ms_pre, pure_ms_post, adjustment_applied, theory_out_mibps_used
    rows = []
    with PURE_CSV.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    # Map: {model: {segX: pure_ms_final}}
    P = {m: {} for m in MODELS}
    for r in rows:
        m = r.get('model'); seg = r.get('segment')
        if not m or not seg: continue
        if m not in P: P[m] = {}
        try:
            v = float(r.get('pure_ms_final') or r.get('pure_ms_pre') or 0.0)
        except Exception:
            v = 0.0
        P[m][seg] = v
    return P


def expand_group_to_segments(group_name: str):
    """Map a combo group name like 'seg3' or 'seg4to8' to a list of concrete segments.
    Returns list like ['seg3'] or ['seg4','seg5','seg6','seg7','seg8'].
    """
    group_name = group_name.strip()
    if 'to8' in group_name:
        # format: seg{start}to8
        try:
            start = int(group_name.replace('seg', '').replace('to8', ''))
        except Exception:
            return []
        return [f'seg{i}' for i in range(start, 9)]
    # single seg like seg1
    return [group_name]


def mib_to_bytes(mib: float) -> float:
    return float(mib) * 1024.0 * 1024.0


def compute_chain_times():
    combos = read_combos()
    segments = read_segments()
    P = read_pure_times()
    out_rows = []
    for model in MODELS:
        if model not in combos or model not in P:
            # still allow K=8 via segments even if combos missing
            kmap = {}
        else:
            kmap = combos[model]  # dict like {"K2": {...}, "K3": {...}}
        # Sort K keys numerically (K2 < K3 < ...)
        def k_to_int(k: str) -> int:
            try:
                return int(k[1:])
            except Exception:
                return 0
        # ensure K8 present using per-segment definitions
        k_keys = sorted(kmap.keys(), key=k_to_int)
        if 'K8' not in k_keys:
            k_keys = k_keys + ['K8']
        for Kkey in k_keys:
            K = k_to_int(Kkey)
            # Build groups_def: for K8 use per-segment data, otherwise from combos
            if K == 8:
                segs_def = segments.get(model, {})
                # maintain seg1..seg8 order
                ordered = {f'seg{i}': segs_def.get(f'seg{i}', {}) for i in range(1, 9)}
                groups_def = ordered
            else:
                groups_def = kmap.get(Kkey, {})  # dict of groupName -> fields

            total_lb_ms = 0.0
            total_ub_ms = 0.0
            notes = []
            for gi, (gname, gd) in enumerate(groups_def.items(), start=1):
                # IO sizes (bytes) – use base_input_bytes/base_output_bytes
                d_in = float(gd.get('base_input_bytes') or 0.0)
                d_out = float(gd.get('base_output_bytes') or 0.0)
                # Weights (MiB)
                w_tot_mib = float(gd.get('segment_model_MiB') or 0.0)
                w_rem_mib = float(gd.get('off_used_MiB') or 0.0)
                w_warm_mib = max(w_tot_mib - w_rem_mib, 0.0)
                if w_rem_mib > 0.0:
                    notes.append(f'{gname}:off_used_MiB={w_rem_mib:.3f}')

                # Compute time for this group (sum pure across its segments)
                segs = expand_group_to_segments(gname)
                Ce_ms = sum(P[model].get(seg, 0.0) for seg in segs)

                # Data transfer times (ms)
                Cin_ms = (d_in / (B_IN * 1024 * 1024.0)) * 1000.0 if d_in else 0.0
                Cout_ms = (d_out / (B_OUT * 1024 * 1024.0)) * 1000.0 if d_out else 0.0

                # Weight warm time for resident part (ms)
                t_warm_ms = (w_warm_mib / B_IN) * 1000.0 if w_warm_mib else 0.0

                # Per-group t_rem: only subtract this group's compute (non-aggregated)
                t_rem_ms_raw = (w_rem_mib / B_IN) * 1000.0 if w_rem_mib else 0.0
                t_rem_lb_ms = max(t_rem_ms_raw - Ce_ms, 0.0)
                t_rem_ub_ms = t_rem_ms_raw

                # Group makespan bounds
                Wi_lb_ms = Cin_ms + Cout_ms + Ce_ms + t_warm_ms + t_rem_lb_ms + EPS_MS
                Wi_ub_ms = Cin_ms + Cout_ms + Ce_ms + t_warm_ms + t_rem_ub_ms + EPS_MS
                total_lb_ms += Wi_lb_ms
                total_ub_ms += Wi_ub_ms

                out_rows.append({
                    'model': model,
                    'K': K,
                    'group_index': gi,
                    'group_name': gname,
                    'group_segs': ','.join(segs),
                    'Cin_ms': round(Cin_ms, 3),
                    'Cout_ms': round(Cout_ms, 3),
                    'Ce_ms': round(Ce_ms, 3),
                    't_warm_ms': round(t_warm_ms, 3),
                    't_rem_lb_ms': round(t_rem_lb_ms, 3),
                    't_rem_ub_ms': round(t_rem_ub_ms, 3),
                    'Wi_lb_ms': round(Wi_lb_ms, 3),
                    'Wi_ub_ms': round(Wi_ub_ms, 3),
                    'w_warm_MiB': round(w_warm_mib, 3),
                    'w_rem_MiB': round(w_rem_mib, 3),
                    'notes': ';'.join(notes),
                })

            # Add a model/K total row
            out_rows.append({
                'model': model,
                'K': K,
                'group_index': 'TOTAL',
                'group_name': '-',
                'group_segs': '-',
                'Cin_ms': '',
                'Cout_ms': '',
                'Ce_ms': '',
                't_warm_ms': '',
                't_rem_lb_ms': '',
                't_rem_ub_ms': '',
                'Wi_lb_ms': round(total_lb_ms, 3),
                'Wi_ub_ms': round(total_ub_ms, 3),
                'w_warm_MiB': '',
                'w_rem_MiB': '',
                'notes': ';'.join(notes),
            })
    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='') as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                'model','K','group_index','group_name','group_segs',
                'Cin_ms','Cout_ms','Ce_ms','t_warm_ms','t_rem_lb_ms','t_rem_ub_ms',
                'Wi_lb_ms','Wi_ub_ms','w_warm_MiB','w_rem_MiB','notes'
            ]
        )
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print('Wrote', OUT_CSV)

if __name__ == '__main__':
    compute_chain_times()
