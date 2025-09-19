#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 models_local_combo_chain 下各模型 K=2..8 结果：
- 用 usbmon.txt 的“任意两条记录间隔 >100ms”为节点分割窗口
- 每窗口内：先找 Bo 的第一个 S 到最后一个 C 定义 Bo span；
  再从该 C 之后，统计 Bi 的跨度（不强制配对，直接用事件时间）和字节（用 Ci 完成字节），计算 Bi 速度。
- 过滤掉 Bi 总字节过小的窗口（默认 <64 字节）

并且支持两种“输出很小段”的过滤方式，避免极小量传输污染统计：
1) 基于 five_models/baselines 中的每段 baseline 输出字节（默认阈值 4 KiB）直接剔除；
2) 可选：基于该 seg 的 Bi 字节样本中位数的动态剔除（默认关闭）。

输出：
- 每个模型/每个 K（K2..K8）的 Bi 速度列表、均值、p50/p95、min/max
- 判断是否集中在 [35,105] MiB/s 区间；以及是否落在“自身均值 ±50%”区间
"""
import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse


def parse_usbmon_events(usb_path: Path):
    all_ts: List[float] = []
    bo_s: List[float] = []
    bo_c: List[float] = []
    bi_any: List[float] = []  # Bi 事件时间（S/C皆可，用于跨度）
    ci_bytes: List[Tuple[float, int]] = []  # (ts, bytes) 仅完成行用于统计字节

    dir_pat = re.compile(r'^[CB][io]:\d+:\d+:\d+')

    with open(usb_path, 'r', errors='ignore') as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 3:
                continue
            # 时间戳在第2列；若为微秒级大数，规范到秒
            try:
                ts = float(parts[1])
                ts = ts / 1e6 if ts > 1e6 else ts
            except Exception:
                continue
            all_ts.append(ts)

            sc = parts[2]
            tok = None
            for p in parts:
                if dir_pat.match(p):
                    tok = p
                    break
            if not tok:
                continue

            # 方向前缀：Bi/Bo/Ci/Co
            prefix = tok[:2]

            # 提取字节数（len= 或 方向标记后第2列，或 # N 形式）
            nbytes = 0
            m = re.search(r'len=(\d+)', ln)
            if m:
                nbytes = int(m.group(1))
            else:
                dir_idx = None
                for i, p in enumerate(parts):
                    if dir_pat.match(p):
                        dir_idx = i
                        break
                if dir_idx is not None and dir_idx + 2 < len(parts):
                    try:
                        nbytes = int(parts[dir_idx + 2])
                    except Exception:
                        nbytes = 0
                if nbytes == 0:
                    m2 = re.search(r'#\s*(\d+)', ln)
                    if m2:
                        nbytes = int(m2.group(1))

            # 记录 Bo S/C
            if sc == 'S' and prefix == 'Bo':
                bo_s.append(ts)
            if sc == 'C' and (prefix == 'Bo' or prefix == 'Co'):
                bo_c.append(ts)

            # 记录 Bi 事件与 Ci 完成字节
            if prefix in ('Bi', 'Ci'):
                # 任意S/C都计入跨度参考
                bi_any.append(ts)
                # 仅完成行用于字节
                if sc == 'C':
                    ci_bytes.append((ts, nbytes))

    all_ts.sort(); bo_s.sort(); bo_c.sort(); bi_any.sort(); ci_bytes.sort(key=lambda x: x[0])
    return all_ts, bo_s, bo_c, bi_any, ci_bytes


def split_windows_by_gap(all_ts: List[float], gap_ms: float = 100.0) -> List[Tuple[float, float]]:
    if not all_ts:
        return []
    gap_s = gap_ms / 1000.0
    w = []
    s = all_ts[0]
    prev = s
    for ts in all_ts[1:]:
        if ts - prev > gap_s:
            w.append((s, prev))
            s = ts
        prev = ts
    w.append((s, prev))
    return w


def compute_bo_and_bi_in_window(ws: float, we: float,
                                bo_s: List[float], bo_c: List[float],
                                bi_any: List[float], ci_bytes: List[Tuple[float, int]],
                                min_bi_total_bytes: int = 64) -> Dict:
    import bisect
    # Bo: 第一个S、最后一个C
    i = bisect.bisect_left(bo_s, ws)
    first_s = bo_s[i] if i < len(bo_s) and bo_s[i] <= we else None
    j = bisect.bisect_right(bo_c, we) - 1
    last_c = bo_c[j] if j >= 0 and bo_c[j] >= ws else None

    bo_span = 0.0
    if first_s is not None and last_c is not None and last_c > first_s:
        bo_span = last_c - first_s

    # Bi: 仅在 Bo 结束之后的区段，使用任意 Bi 事件定义跨度；字节仅累加 Ci 完成
    bi_span = 0.0
    bi_bytes = 0
    if last_c is not None:
        k0 = bisect.bisect_left(bi_any, last_c)
        # 至少需要1个事件来定义跨度起点；理想情况下 >=2个事件确定跨度>0
        bi_events_after = [t for t in bi_any[k0:] if t <= we]
        if bi_events_after:
            bi_start = bi_events_after[0]
            bi_end = bi_events_after[-1]
            if bi_end > bi_start:
                bi_span = bi_end - bi_start
            # 累加该区间内的 Ci 完成字节
            for ts, nb in ci_bytes:
                if ts < bi_start:
                    continue
                if ts > we:
                    break
                bi_bytes += int(nb or 0)

    valid_bi = bi_span > 0 and bi_bytes >= min_bi_total_bytes
    return {
        'bo_span_s': bo_span,
        'bi_span_s': bi_span,
        'bi_bytes': bi_bytes,
        'bi_valid': valid_bi,
    }


def pct(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    k = p/100.0 * (n-1)
    i = int(k)
    f = k - i
    return s[-1] if i >= n-1 else s[i]*(1-f) + s[i+1]*f


def analyze_combo_chain(root: Path, *, gap_ms: float = 100.0, min_bi_total_bytes: int = 64,
                        min_seg_output_bytes: int = 0,
                        baseline_json: Optional[Path] = None,
                        baseline_output_threshold_bytes: int = 4096,
                        use_baseline_exclude: bool = True):
    model_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.endswith('_8seg_uniform_local')]
    summary: Dict[str, Dict] = {}
    # 载入 baseline 输出字节表
    baseline_data = None
    if baseline_json and baseline_json.exists():
        try:
            with open(baseline_json, 'r') as bf:
                baseline_data = json.load(bf)
        except Exception:
            baseline_data = None
    global_speeds: List[float] = []  # 用于跨模型“通用”边界统计（已剔除小输出段）
    for mdir in sorted(model_dirs):
        model_name = mdir.name.split('_8seg_uniform_local')[0]
        summary[model_name] = {}
        # 基于 baseline 的每段输出字节（若可用）确定需要剔除的段
        baseline_key = f"{model_name}_8seg_uniform_local"
        baseline_seg_out: Dict[int, int] = {}
        exclude_by_baseline: Dict[int, bool] = {}
        if baseline_data and baseline_key in baseline_data:
            segs_info = baseline_data[baseline_key].get('segments', {})
            for seg_name, seg_info in segs_info.items():
                try:
                    seg_index = int(seg_name.replace('seg',''))
                except Exception:
                    continue
                out_bytes = int(seg_info.get('output', {}).get('bytes', 0))
                baseline_seg_out[seg_index] = out_bytes
                exclude_by_baseline[seg_index] = use_baseline_exclude and (out_bytes < baseline_output_threshold_bytes)
        for k in range(2, 9):
            kdir = mdir / f'K{k}'
            usb = kdir / 'usbmon.txt'
            if not usb.exists():
                continue
            all_ts, bo_s, bo_c, bi_any, ci_bytes = parse_usbmon_events(usb)
            windows = split_windows_by_gap(all_ts, gap_ms=gap_ms)

            # 逐窗口：先计算 Bo/Bi，再将含 Bo 的有效窗口按 1..K 分配 seg_id（仅 Bo 有效才推进序列）
            details = []
            valid_bo_count = 0
            per_seg_speeds: Dict[int, List[float]] = {seg: [] for seg in range(1, k+1)}
            per_seg_counts: Dict[int, int] = {seg: 0 for seg in range(1, k+1)}  # Bo有效次数计数
            per_seg_bytes: Dict[int, List[int]] = {seg: [] for seg in range(1, k+1)}  # 该 seg 的 Bi 字节样本
            for idx, (ws, we) in enumerate(windows):
                stat = compute_bo_and_bi_in_window(ws, we, bo_s, bo_c, bi_any, ci_bytes, min_bi_total_bytes=min_bi_total_bytes)
                # Bo 是否有效：有正 span
                bo_valid = stat['bo_span_s'] > 0
                seg_id = None
                if bo_valid:
                    seg_id = (valid_bo_count % k) + 1
                    per_seg_counts[seg_id] += 1
                    valid_bo_count += 1
                speed = 0.0
                if stat['bi_valid']:
                    speed = (stat['bi_bytes'] / (1024*1024)) / stat['bi_span_s'] if stat['bi_span_s'] > 0 else 0.0
                    if seg_id is not None:
                        per_seg_speeds[seg_id].append(speed)
                        per_seg_bytes[seg_id].append(int(stat['bi_bytes']))

                details.append({
                    'window': idx,
                    'ws': ws,
                    'we': we,
                    'seg_id': seg_id,
                    'bo_span_s': stat['bo_span_s'],
                    'bi_span_s': stat['bi_span_s'],
                    'bi_bytes': stat['bi_bytes'],
                    'bi_speed_MiBps': speed if stat['bi_valid'] else 0.0,
                    'bo_valid': bo_valid,
                    'bi_valid': stat['bi_valid']
                })

            # 按 seg 计算字节分布并标记需剔除的“输出非常小”的 seg
            # 先：可选的动态中位数判定（默认关闭：min_seg_output_bytes=0）
            per_seg_excluded: Dict[int, bool] = {}
            per_seg_median_bytes: Dict[int, float] = {}
            per_seg_mean_bytes: Dict[int, float] = {}
            for seg in range(1, k+1):
                b = per_seg_bytes[seg]
                if b:
                    per_seg_median_bytes[seg] = pct([float(x) for x in b], 50)
                    per_seg_mean_bytes[seg] = sum(b)/len(b)
                else:
                    per_seg_median_bytes[seg] = 0.0
                    per_seg_mean_bytes[seg] = 0.0
                per_seg_excluded[seg] = (min_seg_output_bytes > 0 and per_seg_median_bytes[seg] < float(min_seg_output_bytes))
            # 再：若有 baseline，则用 baseline 的小输出判定（优先生效）
            for seg in range(1, k+1):
                if exclude_by_baseline.get(seg, False):
                    per_seg_excluded[seg] = True

            # 总体（按K聚合，排除被标记的 seg）
            all_speeds: List[float] = []
            for seg, lst in per_seg_speeds.items():
                if not per_seg_excluded.get(seg, False):
                    all_speeds.extend(lst)
                    global_speeds.extend(lst)
            if all_speeds:
                mean = sum(all_speeds)/len(all_speeds)
                lower = min(all_speeds)
                upper = max(all_speeds)
                p50 = pct(all_speeds, 50)
                p10 = pct(all_speeds, 10)
                p90 = pct(all_speeds, 90)
                p5 = pct(all_speeds, 5)
                p95 = pct(all_speeds, 95)
                in_35_105 = sum(1 for s in all_speeds if 35.0 <= s <= 105.0)
                frac_35_105 = in_35_105 / len(all_speeds)
                band_l = 0.5*mean
                band_u = 1.5*mean
                in_band = sum(1 for s in all_speeds if band_l <= s <= band_u) / len(all_speeds)
            else:
                mean=lower=upper=p50=p10=p90=p5=p95=frac_35_105=in_band=0.0

            # 保存每个K的结果文件
            out = {
                'model': model_name,
                'K': k,
                'count_valid': len(all_speeds),
                'mean': mean,
                'min': lower,
                'max': upper,
                'p50': p50,
                'p10': p10,
                'p90': p90,
                'p5': p5,
                'p95': p95,
                'coverage_35_105': frac_35_105,
                'coverage_mean_pm50pct': in_band,
                'speeds': all_speeds,
                'per_seg': {
                    str(seg): {
                        'count_valid': len(per_seg_speeds[seg]),
                        'mean': (sum(per_seg_speeds[seg])/len(per_seg_speeds[seg])) if per_seg_speeds[seg] else 0.0,
                        'min': min(per_seg_speeds[seg]) if per_seg_speeds[seg] else 0.0,
                        'max': max(per_seg_speeds[seg]) if per_seg_speeds[seg] else 0.0,
                        'p50': pct(per_seg_speeds[seg], 50) if per_seg_speeds[seg] else 0.0,
                        'p10': pct(per_seg_speeds[seg], 10) if per_seg_speeds[seg] else 0.0,
                        'p90': pct(per_seg_speeds[seg], 90) if per_seg_speeds[seg] else 0.0,
                        'p5': pct(per_seg_speeds[seg], 5) if per_seg_speeds[seg] else 0.0,
                        'p95': pct(per_seg_speeds[seg], 95) if per_seg_speeds[seg] else 0.0,
                        'coverage_35_105': (sum(1 for s in per_seg_speeds[seg] if 35.0 <= s <= 105.0) / len(per_seg_speeds[seg])) if per_seg_speeds[seg] else 0.0,
                        'coverage_mean_pm50pct': (sum(1 for s in per_seg_speeds[seg] if 0.5*((sum(per_seg_speeds[seg])/len(per_seg_speeds[seg]))) <= s <= 1.5*((sum(per_seg_speeds[seg])/len(per_seg_speeds[seg])))) / len(per_seg_speeds[seg])) if per_seg_speeds[seg] else 0.0,
                        'bo_invokes': per_seg_counts[seg],
                        'median_bytes': per_seg_median_bytes[seg],
                        'mean_bytes': per_seg_mean_bytes[seg],
                        'excluded_small_output': per_seg_excluded[seg],
                        'baseline_output_bytes': baseline_seg_out.get(seg, 0),
                        'excluded_by_baseline': exclude_by_baseline.get(seg, False),
                    }
                    for seg in range(1, k+1)
                },
                'details': details,
                'config': {
                    'gap_ms': gap_ms,
                    'min_bi_total_bytes_per_window': min_bi_total_bytes,
                    'min_seg_output_bytes_median': min_seg_output_bytes,
                    'baseline_json': str(baseline_json) if baseline_json else None,
                    'baseline_output_threshold_bytes': baseline_output_threshold_bytes,
                    'use_baseline_exclude': use_baseline_exclude,
                }
            }
            with open(kdir / 'bi_span_by_nodes.json', 'w') as f:
                json.dump(out, f, indent=2)

            summary[model_name][f'K{k}'] = {
                'count_valid': len(all_speeds),
                'mean': mean,
                'min': lower,
                'max': upper,
                'p50': p50,
                'p10': p10,
                'p90': p90,
                'p5': p5,
                'p95': p95,
                'coverage_35_105': frac_35_105,
                'coverage_mean_pm50pct': in_band,
                'excluded_segs': [seg for seg, ex in per_seg_excluded.items() if ex],
            }

    # 打印汇总
    print('Bi 速度统计（Bo结束后区段，Ci字节/跨度；按>100ms节点分窗）')
    print('='*80)
    for model, ks in summary.items():
        print(f'模型: {model}')
        for k in range(2,9):
            kk = f'K{k}'
            if kk not in ks:
                continue
            s = ks[kk]
            print(f"  {kk}: n={s['count_valid']:<3} mean={s['mean']:.1f} MiB/s  min={s['min']:.1f}  max={s['max']:.1f}  p5={s.get('p5',0):.1f}  p10={s.get('p10',0):.1f}  p50={s['p50']:.1f}  p90={s.get('p90',0):.1f}  p95={s['p95']:.1f}  in[35,105]={s['coverage_35_105']*100:.0f}%  in±50%={s['coverage_mean_pm50pct']*100:.0f}%")
            # 读取每K的文件打印每个seg（跳过被标记为输出极小的段）
            try:
                kfile = (root / f"{model}_8seg_uniform_local" / kk / 'bi_span_by_nodes.json')
                kout = json.load(open(kfile,'r'))
                per_seg = kout.get('per_seg', {})
                for seg_id in sorted(per_seg.keys(), key=lambda x: int(x)):
                    ps = per_seg[seg_id]
                    if ps.get('excluded_small_output'):
                        continue
                    print(f"    seg{seg_id}: n={ps['count_valid']:<3} mean={ps['mean']:.1f} min={ps['min']:.1f} max={ps['max']:.1f} p5={ps.get('p5',0):.1f} p10={ps.get('p10',0):.1f} p50={ps['p50']:.1f} p90={ps.get('p90',0):.1f} p95={ps['p95']:.1f} in[35,105]={ps['coverage_35_105']*100:.0f}% in±50%={ps['coverage_mean_pm50pct']*100:.0f}%  (baseline bytes={ps.get('baseline_output_bytes',0)})")
            except Exception:
                pass
        print('-'*80)

    # 计算“通用”Bi 速度边界（已剔除小输出段）
    print('全模型通用 Bi 速度边界（已剔除基线小输出段）')
    print('='*80)
    if global_speeds:
        g_mean = sum(global_speeds)/len(global_speeds)
        g_p50 = pct(global_speeds, 50)
        g_p10 = pct(global_speeds, 10)
        g_p90 = pct(global_speeds, 90)
        g_p05 = pct(global_speeds, 5)
        g_p95 = pct(global_speeds, 95)
        # MAD 估计
        med = g_p50
        abs_dev = [abs(x - med) for x in global_speeds]
        mad = pct(abs_dev, 50)
        robust_sigma = 1.4826 * mad
        band_mad_l = max(0.0, med - 1.5*robust_sigma)
        band_mad_u = med + 1.5*robust_sigma
        print(f"样本数={len(global_speeds)}  mean={g_mean:.1f}  median={g_p50:.1f}  p10={g_p10:.1f}  p90={g_p90:.1f}  p5={g_p05:.1f}  p95={g_p95:.1f}")
        print(f"建议边界：percentile带 [p10,p90]=[{g_p10:.1f},{g_p90:.1f}] MiB/s；MAD带 [median±1.5*MAD*1.4826]=[{band_mad_l:.1f},{band_mad_u:.1f}] MiB/s")
        # 保存到根目录
        uni_out = {
            'count': len(global_speeds),
            'mean': g_mean,
            'median': g_p50,
            'p10': g_p10,
            'p90': g_p90,
            'p5': g_p05,
            'p95': g_p95,
            'mad': mad,
            'robust_sigma': robust_sigma,
            'band_percentile_10_90': [g_p10, g_p90],
            'band_mad_1p5sigma': [band_mad_l, band_mad_u],
        }
        try:
            uni_path = root / 'bi_speed_universal_bounds.json'
            with open(uni_path, 'w') as uf:
                json.dump(uni_out, uf, indent=2)
            print(f"保存通用边界到: {uni_path}")
        except Exception:
            pass
    else:
        print('无有效样本（可能所有段都被排除）。')


def main():
    parser = argparse.ArgumentParser(description='Analyze Bi speed per segment by usbmon node windows (>gap_ms).')
    parser.add_argument('--root', type=str, default='results/models_local_combo_chain', help='Root directory of combo chain results')
    parser.add_argument('--gap-ms', type=float, default=100.0, help='Gap threshold in milliseconds to split windows')
    parser.add_argument('--min-bi-total-bytes', type=int, default=64, help='Minimum Bi bytes per window to consider valid')
    parser.add_argument('--min-seg-output-bytes', type=int, default=0, help='Dynamic exclude if seg median Bi bytes < this (0 disables)')
    parser.add_argument('--baseline-json', type=str, default='five_models/baselines/models_io_baseline_correct.json', help='Path to baseline IO json')
    parser.add_argument('--baseline-output-threshold-bytes', type=int, default=4096, help='Exclude seg if its baseline output bytes < this')
    parser.add_argument('--no-baseline-exclude', action='store_true', help='Disable baseline-based small-output exclusion')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f'Not found: {root}')
        return
    analyze_combo_chain(
        root,
        gap_ms=args.gap_ms,
        min_bi_total_bytes=args.min_bi_total_bytes,
        min_seg_output_bytes=args.min_seg_output_bytes,
        baseline_json=Path(args.baseline_json) if args.baseline_json else None,
        baseline_output_threshold_bytes=args.baseline_output_threshold_bytes,
        use_baseline_exclude=(not args.no_baseline_exclude),
    )


if __name__ == '__main__':
    main()
