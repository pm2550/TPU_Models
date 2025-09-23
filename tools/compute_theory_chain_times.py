7#!/usr/bin/env python3
import csv
import json
import os
from pathlib import Path

BASE = Path('/home/10210/Desktop/OS')
# Use combo-specific theory tensors and weight splits (on-chip/off-chip) per K and group
THEORY_COMBOS = BASE/'five_models/baselines/theory_io_combos.json'
THEORY_SEG = BASE/'five_models/baselines/theory_io_seg.json'
PURE_CSV = BASE/'five_models/results/single_pure_invoke_times.csv'
OUT_CSV = BASE/'five_models/results/theory_chain_times.csv'
PURE_COMBINED = BASE/'results/models_local_batch_usbmon/single/combined_pure_gap_seg1-8_summary.csv'
SPAN_SUMMARY = BASE/'results/models_local_batch_usbmon/single/combined_summary_span.json'

# Config: effective link bandwidths (MiB/s)
# Per user's latest instruction: B_in = 320 (h2d), B_out = 60 (d2h)
# Mapping per user: H2D (Cin) faster, D2H (Cout) slower
#   B_IN  = 330 MiB/s  (Cin, host → device)
#   B_OUT = 60  MiB/s  (Cout, device → host)
# Variant lower-bound bandwidths (MiB/s) for sensitivity/UB line
#   B_IN2  (Cin variant, host → device)
#   B_OUT2 (Cout variant, device → host)
# Support env overrides so batch scripts can sweep bounds without editing this file.
def _fenv(name: str, default: float) -> float:
    try:
        v = os.environ.get(name)
        return float(v) if v is not None and v != '' else float(default)
    except Exception:
        return float(default)

# Bandwidth source toggle:
# If True, ignore environment variables and always use the hardcoded defaults below.
# If False (default), read from environment with the defaults as fallbacks.
USE_CODE_DEFAULTS = True
# Hardcoded defaults (MiB/s)
DEFAULT_B_IN = 325
DEFAULT_B_OUT = 87.0
DEFAULT_B_IN2 = 325  # lower variant for H2D
DEFAULT_B_OUT2 = 35.0   # lower variant for D2H (assumption; override via env)


if USE_CODE_DEFAULTS:
    B_IN = float(DEFAULT_B_IN)
    B_OUT = float(DEFAULT_B_OUT)
    B_IN2 = float(DEFAULT_B_IN2)
    B_OUT2 = float(DEFAULT_B_OUT2)
else:
    B_IN = _fenv('B_IN', DEFAULT_B_IN)
    B_OUT = _fenv('B_OUT', DEFAULT_B_OUT)
    # Variant lower-bound H2D bandwidth (MiB/s) for sensitivity line (used as second bound / UB)
    B_IN2 = _fenv('B_IN2', _fenv('UB', DEFAULT_B_IN2))
    # Variant lower-bound D2H bandwidth
    B_OUT2 = _fenv('B_OUT2', _fenv('UB_OUT', DEFAULT_B_OUT2))

EPS_MS = 0.0   # small overhead per segment (ignored by default)

# Host-side handling model (function of in_span per segment):
# Option A (default): per-model intercept Th(model) and global slope kappa
#   Delta_i = Th(model) + kappa * U_in_i
# Option B: global intercept HOST_C_MS and global slope kappa
#   Delta_i = HOST_C_MS + kappa * U_in_i
KAPPA_MS_PER_MS = 0.2992134815732149
HOST_C_MS = 0.5527747236073199
USE_PER_MODEL_THOST = False

# Host delta span source policy:
# - 'measured_in': use usbmon measured H2D envelope span
# - 'theory_cin':  use computed Cin_ms for the group
# - 'theory_cout': use computed Cout_ms for the group
# Can be overridden by env HOST_DELTA_SPAN_SOURCE, but you can set the default here.
HOST_DELTA_SPAN_SOURCE = os.environ.get('HOST_DELTA_SPAN_SOURCE', 'theory_cout')

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

def read_in_span_ms():
    """Read per-segment input envelope span (ms). Returns {model:{segX: in_span_ms_mean}}"""
    if not SPAN_SUMMARY.exists():
        return {}
    try:
        J = json.loads(SPAN_SUMMARY.read_text())
    except Exception:
        return {}
    M = {}
    for m, obj in J.items():
        segs = obj or {}
        mm = {}
        for seg, sd in segs.items():
            try:
                mm[seg] = float(sd.get('in_span_ms_mean') or 0.0)
            except Exception:
                mm[seg] = 0.0
        M[m] = mm
    return M

def read_T_host_per_model(kappa: float) -> dict:
    """Estimate per-model Th(host) as mean(delta - kappa*in_span_ms) over that model's rows.
    Falls back to HOST_C_MS if data missing.
    Returns {model: T_host_ms}
    """
    T = {m: HOST_C_MS for m in MODELS}
    from pathlib import Path as _P
    dm = BASE/'results/envelope_delta_merged.csv'
    if not dm.exists():
        return T
    try:
        rows = []
        with dm.open() as f:
            rd = csv.DictReader(f)
            for r in rd:
                rows.append(r)
        by_model = {m: [] for m in MODELS}
        for r in rows:
            m = r.get('model')
            if m not in by_model:
                continue
            try:
                delta = float(r.get('pre_delta') or 0.0)
                in_ms = float(r.get('in_span_ms') or 0.0)
            except Exception:
                continue
            by_model[m].append(delta - kappa*in_ms)
        for m, vals in by_model.items():
            if vals:
                T[m] = sum(vals)/len(vals)
    except Exception:
        pass
    return T

def read_pure_times():
    """Read final pure times from CSV (pure_ms_final preferred). Returns {model:{segX: ms}}"""
    rows = []
    with PURE_CSV.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    P = {m: {} for m in MODELS}
    for r in rows:
        m = r.get('model'); seg = r.get('segment')
        if not m or not seg: continue
        if m not in P: P[m] = {}
        try:
            v = float(r.get('pure_ms_final') or r.get('pure_ms_post') or r.get('pure_ms_pre') or 0.0)
        except Exception:
            v = 0.0
        P[m][seg] = v
    return P

def sync_pure_pre_from_combined():
    """Write new pure (p50_ms) into CSV's pure_ms_pre for segments present in combined file.
    Also updates 'count' and marks 'source'='new_pure_gap' for those rows.
    """
    if not PURE_COMBINED.exists() or not PURE_CSV.exists():
        return
    # Load combined p50
    combined = {}
    with PURE_COMBINED.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            m = r.get('model'); seg = r.get('segment')
            if not m or not seg: continue
            try:
                p50 = float(r.get('p50_ms') or 0.0)
                cnt = int(r.get('count') or 0)
            except Exception:
                p50, cnt = 0.0, 0
            combined.setdefault(m, {})[seg] = {'p50': p50, 'count': cnt}
    # Read CSV, update pure_ms_pre/count/source only
    rows = []
    with PURE_CSV.open() as f:
        rd = csv.DictReader(f)
        fieldnames = rd.fieldnames or []
        if 'source' not in fieldnames:
            fieldnames.append('source')
        for r in rd:
            m = r.get('model'); seg = r.get('segment')
            info = (combined.get(m) or {}).get(seg)
            if info:
                r['pure_ms_pre'] = f"{info['p50']}"
                if info['count']:
                    r['count'] = str(info['count'])
                # tag the source of pre values
                r['source'] = 'new_pure_gap'
            rows.append(r)
    # Write back
    with PURE_CSV.open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader(); wr.writerows(rows)


def update_pure_csv_with_offchip(segments_def):
    """Apply off-chip adjustment to CSV in-place.
    post = max(pre, off_used_MiB/B_IN*1000) for off-chip segments; else post=pre.
    Also sets pure_ms_final=post and adjustment_applied flag.
    """
    import csv
    from shutil import copyfile

    # Build lookup for off_used_MiB per model+segment
    off_used = {m: {} for m in MODELS}
    for m, segs in segments_def.items():
        if m not in off_used:
            continue
        for i in range(1, 9):
            seg = f'seg{i}'
            gd = segs.get(seg, {}) if isinstance(segs, dict) else {}
            try:
                off_used[m][seg] = float(gd.get('off_used_MiB') or 0.0)
            except Exception:
                off_used[m][seg] = 0.0

    # Read/update/write
    rows = []
    with PURE_CSV.open() as f:
        rd = csv.DictReader(f)
        fieldnames = rd.fieldnames or []
        for need in ('pure_ms_pre','pure_ms_post','pure_ms_final','adjustment_applied','theory_out_mibps_used','source'):
            if need not in fieldnames:
                fieldnames.append(need)
        for r in rd:
            m = r.get('model'); seg = r.get('segment')
            if m in MODELS and seg in {f'seg{i}' for i in range(1,9)}:
                try:
                    pre = float(r.get('pure_ms_pre') or 0.0)
                except Exception:
                    pre = 0.0
                off_mib = (off_used.get(m, {}) or {}).get(seg, 0.0)
                if off_mib and off_mib > 0.0:
                    t_off_ms = (off_mib / B_IN) * 1000.0
                    post = max(pre, t_off_ms)
                    r['adjustment_applied'] = '1' if post > pre else '0'
                    r['theory_out_mibps_used'] = str(B_IN)
                    # mark adjusted-from-new if applicable
                    if (r.get('source') or '').startswith('new_pure_gap'):
                        r['source'] = 'new_pure_gap+adjusted'
                else:
                    post = pre
                    r['adjustment_applied'] = '0'
                    if not (r.get('theory_out_mibps_used')):
                        r['theory_out_mibps_used'] = '320.0'
                r['pure_ms_post'] = f"{post}"
                r['pure_ms_final'] = f"{post}"
            rows.append(r)

    try:
        copyfile(PURE_CSV, PURE_CSV.with_suffix('.csv.bak'))
    except Exception:
        pass
    with PURE_CSV.open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader(); wr.writerows(rows)


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
    # Optionally refresh pure_ms_pre from gap-based combined medians and apply off-chip adjustment.
    # Guarded by env vars to avoid overwriting user's manual edits by default.
    if os.environ.get('SYNC_PURE_GAP') == '1':
        try:
            sync_pure_pre_from_combined()
        except Exception:
            pass
    combos = read_combos()
    segments = read_segments()
    # After (optional) refresh, optionally compute pure_ms_post/final via off-chip adjustment
    if os.environ.get('APPLY_OFFCHIP') == '1':
        try:
            update_pure_csv_with_offchip(segments)
        except Exception:
            pass
    in_span = read_in_span_ms()
    # Prepare Th(host) per model if enabled
    T_host = read_T_host_per_model(KAPPA_MS_PER_MS) if USE_PER_MODEL_THOST else {m: HOST_C_MS for m in MODELS}
    # 使用 CSV 中的最终纯推理时间，不再在此处改写或再调整
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
            total_lb_ms_in2 = 0.0  # variant lower-bound total using B_IN2
            total_ub_ms = 0.0
            # new: accumulate per-group remainder totals for TOTAL row visibility
            total_trem_lb_ms = 0.0
            total_trem_lb_ms_in2 = 0.0
            total_trem_ub_ms = 0.0
            notes = []
            total_host_delta_ms = 0.0
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
                # Variant using B_IN2 for H2D only
                Cin_ms_in2 = (d_in / (B_IN2 * 1024 * 1024.0)) * 1000.0 if d_in else 0.0
                Cout_ms = (d_out / (B_OUT * 1024 * 1024.0)) * 1000.0 if d_out else 0.0
                Cout_ms_in2 = (d_out / (B_OUT2 * 1024 * 1024.0)) * 1000.0 if d_out else 0.0

                # Weight warm time for resident part (ms)
                t_warm_ms = (w_warm_mib / B_IN) * 1000.0 if w_warm_mib else 0.0
                t_warm_ms_in2 = (w_warm_mib / B_IN2) * 1000.0 if w_warm_mib else 0.0

                # Per-group t_rem: only subtract this group's compute (non-aggregated)
                t_rem_ms_raw = (w_rem_mib / B_IN) * 1000.0 if w_rem_mib else 0.0
                t_rem_ms_raw_in2 = (w_rem_mib / B_IN2) * 1000.0 if w_rem_mib else 0.0
                t_rem_lb_ms = max(t_rem_ms_raw - Ce_ms, 0.0)
                t_rem_ub_ms = t_rem_ms_raw
                t_rem_lb_ms_in2 = max(t_rem_ms_raw_in2 - Ce_ms, 0.0)
                # accumulate totals
                total_trem_lb_ms += t_rem_lb_ms
                total_trem_lb_ms_in2 += t_rem_lb_ms_in2
                total_trem_ub_ms += t_rem_ub_ms

                # Host-side overhead for this group: choose span policy for U value driving host delta
                seg_count = len(segs)
                if HOST_DELTA_SPAN_SOURCE == 'theory_cin':
                    # Use computed Cin_ms for this group
                    Uin_used_ms = Cin_ms
                elif HOST_DELTA_SPAN_SOURCE == 'theory_cout':
                    # Use computed Cout_ms for this group (some analyses prefer correlating host delta with D2H)
                    Uin_used_ms = Cout_ms
                else:
                    # Default: measured H2D envelope span from usbmon.
                    # For groups like 'segXto8', use only the last segment's in_span (e.g., seg8);
                    # otherwise, sum in_span across the group's segments.
                    use_last_only = ('to8' in gname)
                    if use_last_only:
                        last_seg = segs[-1] if segs else None
                        Uin_used_ms = (in_span.get(model, {}) or {}).get(last_seg, 0.0) if last_seg else 0.0
                    else:
                        Uin_used_ms = sum((in_span.get(model, {}) or {}).get(seg, 0.0) for seg in segs)
                Th_ms = T_host.get(model, HOST_C_MS)
                # Group host delta: one invoke per group (not per segment)
                group_invokes = 1
                delta_host_ms = group_invokes * Th_ms + KAPPA_MS_PER_MS * Uin_used_ms

                # Group makespan bounds
                Wi_lb_ms = Cin_ms + Cout_ms + Ce_ms + t_warm_ms + t_rem_lb_ms + EPS_MS
                Wi_lb_ms_in2 = Cin_ms_in2 + Cout_ms_in2 + Ce_ms + t_warm_ms_in2 + t_rem_lb_ms_in2 + EPS_MS
                Wi_ub_ms = Cin_ms + Cout_ms + Ce_ms + t_warm_ms + t_rem_ub_ms + EPS_MS
                Wi_lb_host_ms = Wi_lb_ms + delta_host_ms
                Wi_lb_host_ms_in2 = Wi_lb_ms_in2 + delta_host_ms
                Wi_ub_host_ms = Wi_ub_ms + delta_host_ms
                total_lb_ms += Wi_lb_ms
                total_lb_ms_in2 += Wi_lb_ms_in2
                total_ub_ms += Wi_ub_ms
                total_host_delta_ms += delta_host_ms
                # For CSV simplicity: expose only full host delta as Th_ms (per-group total)
                Th_ms_per_invoke = Th_ms

                out_rows.append({
                    'model': model,
                    'K': K,
                    'group_index': gi,
                    'group_name': gname,
                    'group_segs': ','.join(segs),
                    'Cin_ms': round(Cin_ms, 3),
                    'Cin_ms_in2': round(Cin_ms_in2, 3),
                    'Cout_ms': round(Cout_ms, 3),
                    'Cout_ms_in2': round(Cout_ms_in2, 3),
                    'Ce_ms': round(Ce_ms, 3),
                    't_warm_ms': round(t_warm_ms, 3),
                    't_warm_ms_in2': round(t_warm_ms_in2, 3),
                    't_rem_lb_ms': round(t_rem_lb_ms, 3),
                    't_rem_lb_ms_in2': round(t_rem_lb_ms_in2, 3),
                    't_rem_ub_ms': round(t_rem_ub_ms, 3),
                    'Wi_lb_ms': round(Wi_lb_ms, 3),
                    'Wi_lb_ms_in2': round(Wi_lb_ms_in2, 3),
                    'Wi_ub_ms': round(Wi_ub_ms, 3),
                    'Th_ms': round(delta_host_ms, 3),
                    'Wi_lb_ms_hosted': round(Wi_lb_host_ms, 3),
                    'Wi_lb_ms_hosted_in2': round(Wi_lb_host_ms_in2, 3),
                    'Wi_ub_ms_hosted': round(Wi_ub_host_ms, 3),
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
                'Cin_ms_in2': '',
                'Cout_ms': '',
                'Cout_ms_in2': '',
                'Ce_ms': '',
                't_warm_ms': '',
                't_warm_ms_in2': '',
                't_rem_lb_ms': round(total_trem_lb_ms, 3),
                't_rem_lb_ms_in2': round(total_trem_lb_ms_in2, 3),
                't_rem_ub_ms': round(total_trem_ub_ms, 3),
                'Wi_lb_ms': round(total_lb_ms, 3),
                'Wi_lb_ms_in2': round(total_lb_ms_in2, 3),
                'Wi_ub_ms': round(total_ub_ms, 3),
                'Th_ms': round(total_host_delta_ms, 3),
                'Wi_lb_ms_hosted': round(total_lb_ms + total_host_delta_ms, 3),
                'Wi_lb_ms_hosted_in2': round(total_lb_ms_in2 + total_host_delta_ms, 3),
                'Wi_ub_ms_hosted': round(total_ub_ms + total_host_delta_ms, 3),
                'notes': ';'.join(notes),
            })
    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='') as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                'model','K','group_index','group_name','group_segs',
                'Cin_ms','Cin_ms_in2','Cout_ms','Cout_ms_in2','Ce_ms','t_warm_ms','t_warm_ms_in2','t_rem_lb_ms','t_rem_lb_ms_in2','t_rem_ub_ms',
                'Wi_lb_ms','Wi_lb_ms_in2','Wi_ub_ms',
                'Th_ms','Wi_lb_ms_hosted','Wi_lb_ms_hosted_in2','Wi_ub_ms_hosted',
                'notes'
            ]
        )
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print('Wrote', OUT_CSV)

if __name__ == '__main__':
    compute_chain_times()
