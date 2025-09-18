#!/usr/bin/env python3
import os
import json
import csv

SINGLE_ROOT = "/home/10210/Desktop/OS/results/models_local_batch_usbmon/single"
MODELS_BASE = "/home/10210/Desktop/OS/models_local/public"
THEORY_JSON = "/home/10210/Desktop/OS/baselines/theory_io_only.json"
# Write to five_models results for consistency with other outputs
OUT_CSV = "/home/10210/Desktop/OS/five_models/results/single_segment_metrics.csv"

def load_theory():
    try:
        T = json.load(open(THEORY_JSON))
    except Exception:
        T = {}
    # map[(model, seg)] -> (OUT_bytes, IN_bytes)
    m = {}
    for model, rows in (T.items() if isinstance(T, dict) else []):
        for r in rows:
            seg = r.get('seg')
            outb = r.get('OUT_bytes')
            inb = r.get('IN_bytes')
            m[(model, seg)] = (outb, inb)
    return m

def main():
    theory = load_theory()
    rows = []
    if not os.path.isdir(SINGLE_ROOT):
        print(f"not found: {SINGLE_ROOT}")
        return
    for model in sorted(os.listdir(SINGLE_ROOT)):
        mdir = os.path.join(SINGLE_ROOT, model)
        if not os.path.isdir(mdir):
            continue
        # 读取切点层名（full_split summary.json 的 cut_points 决定 seg 切分）
        cut_start = {}
        cut_end = {}
        try:
            summ = os.path.join(MODELS_BASE, model, 'full_split_pipeline_local', 'summary.json')
            SJ = json.load(open(summ))
            cps = SJ.get('cut_points') or []
            # seg1: start=INPUT, end=cut_points[0]
            # segN (2..7): start=cut_points[N-2], end=cut_points[N-1]
            # seg8: start=cut_points[6] (若不足则最后一个), end=OUTPUT
            for seg in range(1,9):
                if seg == 1:
                    cut_start[seg] = 'INPUT'
                    cut_end[seg] = cps[0] if len(cps) >= 1 else 'OUTPUT'
                elif seg == 8:
                    cut_start[seg] = cps[6] if len(cps) >= 7 else (cps[-1] if cps else 'INPUT')
                    cut_end[seg] = 'OUTPUT'
                else:
                    i0 = seg - 2
                    i1 = seg - 1
                    cut_start[seg] = cps[i0] if i0 < len(cps) else (cps[-1] if cps else 'INPUT')
                    cut_end[seg] = cps[i1] if i1 < len(cps) else 'OUTPUT'
        except Exception:
            for seg in range(1,9):
                cut_start[seg] = f'seg{seg}_start'
                cut_end[seg] = f'seg{seg}_end'
        for seg in range(1, 9):
            sdir = os.path.join(mdir, f"seg{seg}")
            invp = os.path.join(sdir, 'invokes.json')
            perfp = os.path.join(sdir, 'performance_summary.json')
            if not (os.path.isfile(invp) and os.path.isfile(perfp)):
                continue
            start_layer = cut_start.get(seg, '')
            end_layer = cut_end.get(seg, '')
            try:
                P = json.load(open(perfp))
            except Exception:
                P = {}
            io_overall = (((P.get('io_performance') or {}).get('strict_window') or {}).get('overall_avg') or {})
            # Prefer span S->C metrics from active_analysis_strict.json
            ana_p = os.path.join(sdir, 'active_analysis_strict.json')
            avg_in_act_ms = None
            avg_out_act_ms = None
            try:
                A = json.load(open(ana_p)) if os.path.isfile(ana_p) else {}
                per = A.get('per_invoke') or []
                # skip first (cold)
                per = per[1:] if len(per) > 1 else per
                if per:
                    in_spans = [float(x.get('in_span_sc_ms') or 0.0) for x in per]
                    out_spans = [float(x.get('out_span_sc_ms') or 0.0) for x in per]
                    import statistics as _st
                    avg_in_act_ms = _st.mean(in_spans) if in_spans else 0.0
                    avg_out_act_ms = _st.mean(out_spans) if out_spans else 0.0
            except Exception:
                pass

            # Fallback to previous active-union metrics if span not available
            if avg_in_act_ms is None or avg_out_act_ms is None:
                act = (P.get('io_active_union_avg') or {})
                avg_in_act_ms = float(act.get('avg_in_active_ms', 0) or 0)
                avg_out_act_ms = float(act.get('avg_out_active_ms', 0) or 0)

            avg_out_b = float(io_overall.get('avg_bytes_out_per_invoke', 0) or 0)
            avg_in_b = float(io_overall.get('avg_bytes_in_per_invoke', 0) or 0)
            # 速率（MiB/s，按各自活跃时长计）
            def mibps(bytes_avg, ms):
                if ms and ms > 0:
                    return (bytes_avg / (1024.0*1024.0)) / (ms/1000.0)
                return 0.0
            in_mibps = mibps(avg_in_b, avg_in_act_ms)
            out_mibps = mibps(avg_out_b, avg_out_act_ms)
            th_out, th_in = theory.get((model, seg), (None, None))
            rows.append([
                model, f"seg{seg}", start_layer, end_layer,
                th_out if th_out is not None else '',
                th_in if th_in is not None else '',
                int(avg_out_b), int(avg_in_b),
                f"{avg_out_act_ms:.3f}", f"{avg_in_act_ms:.3f}",
                f"{out_mibps:.3f}", f"{in_mibps:.3f}",
            ])
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'model','segment','cut_start_layer','cut_end_layer',
            'theory_OUT_bytes','theory_IN_bytes',
            'actual_avg_OUT_bytes','actual_avg_IN_bytes',
            'avg_OUT_active_ms','avg_IN_active_ms',
            'OUT_MiBps_active','IN_MiBps_active'
        ])
        w.writerows(rows)
    print(f"saved: {OUT_CSV}, rows={len(rows)}")

if __name__ == '__main__':
    main()


