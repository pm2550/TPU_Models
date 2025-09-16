#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
聚合 span 口径的 USBMON 分析结果：
- 读取 results/models_local_batch_usbmon/single 下各模型 seg*/active_analysis_strict.json
- 使用方向内 S→C 跨度（in_span_sc_ms/out_span_sc_ms）与 io_span_sum_ms/pure_span_sum_ms 作为唯一口径
- 对 off-chip 段额外计算修正后的纯推理时间（基于 OUT span 速率与理论速率的差额校正）
- 输出三份汇总：onchip_summary_span.json, offchip_summary_span.json, combined_summary_span.json

注：
- 理论速率优先从各 seg 下的 run_env.json 读取 OFFCHIP_OUT_THEORY_MIBPS / OFFCHIP_OUT_MIBPS / OFFCHIP_OUT_THEORY_MIB_PER_MS，
  若均缺省则默认 320 MiB/s。
- off-chip 段的判定来自 five_models/baselines/theory_io_seg.json 的 off_used_MiB>0。
"""

import os
import json
from pathlib import Path
from statistics import mean

WORKDIR = Path('/home/10210/Desktop/OS')
RESULTS_BASE = WORKDIR / 'results' / 'models_local_batch_usbmon' / 'single'
THEORY_PATH = WORKDIR / 'five_models' / 'baselines' / 'theory_io_seg.json'

MODEL_NAMES = [
    'densenet201_8seg_uniform_local',
    'inceptionv3_8seg_uniform_local',
    'resnet101_8seg_uniform_local',
    'resnet50_8seg_uniform_local',
    'xception_8seg_uniform_local',
]

def _safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return (mean(vals) if vals else None)

def _load_json(p: Path):
    try:
        with open(p, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def _get_theory_mibps_from_env(run_env: dict):
    if not isinstance(run_env, dict):
        run_env = {}
    # 兼容多种变量名
    v = run_env.get('OFFCHIP_OUT_THEORY_MIBPS')
    if v is None:
        v = run_env.get('OFFCHIP_OUT_MIBPS')
    try:
        if v is not None:
            return float(v)
    except Exception:
        pass
    per_ms = run_env.get('OFFCHIP_OUT_THEORY_MIB_PER_MS')
    try:
        if per_ms is not None:
            return float(per_ms) * 1000.0
    except Exception:
        pass
    # 默认 320 MiB/s
    return 320.0

def _compute_offchip_adjusted_span_pure(rec: dict, theory_mibps: float):
    """基于 span 速率计算 off-chip 修正：
    delta_ms = max(0, MiB_out/out_speed_span - MiB_out/theory_mibps) * 1000
    返回 (pure_span_pre_ms, pure_span_post_ms)
    """
    try:
        pure_pre = float(rec.get('pure_span_sum_ms'))
    except Exception:
        pure_pre = None
    try:
        out_speed_span = rec.get('out_speed_span_mibps')
        out_speed_span = float(out_speed_span) if out_speed_span is not None else None
    except Exception:
        out_speed_span = None
    try:
        bytes_out = float(rec.get('bytes_out', 0.0) or 0.0)
    except Exception:
        bytes_out = 0.0
    if pure_pre is None:
        return None, None
    MiB_out = bytes_out / (1024.0 * 1024.0)
    delta_ms = 0.0
    if out_speed_span and out_speed_span > 0 and theory_mibps and theory_mibps > 0:
        t_cur = MiB_out / out_speed_span
        t_th = MiB_out / theory_mibps
        delta_ms = max(0.0, (t_cur - t_th) * 1000.0)
    pure_post = pure_pre + delta_ms
    return pure_pre, pure_post

def collect_segment_span_metrics(model_dir: Path, seg: int):
    seg_dir = model_dir / f'seg{seg}'
    ana_p = seg_dir / 'active_analysis_strict.json'
    env_p = seg_dir / 'run_env.json'
    if not ana_p.exists():
        return None
    data = _load_json(ana_p)
    if not isinstance(data, dict) or 'per_invoke' not in data:
        return None
    run_env = _load_json(env_p) or {}
    theory_mibps = _get_theory_mibps_from_env(run_env)
    # 跳过首帧
    per = data['per_invoke'][1:] if len(data['per_invoke']) > 1 else data['per_invoke']
    if not per:
        return None
    in_span_ms = [float(x.get('in_span_sc_ms') or 0.0) for x in per]
    out_span_ms = [float(x.get('out_span_sc_ms') or 0.0) for x in per]
    io_span_sum_ms = [float(x.get('io_span_sum_ms') or ((x.get('in_span_sc_ms') or 0.0) + (x.get('out_span_sc_ms') or 0.0))) for x in per]
    pure_span_ms = [float(x.get('pure_span_sum_ms')) if x.get('pure_span_sum_ms') is not None else None for x in per]
    in_speed_span = [x.get('in_speed_span_mibps') for x in per]
    out_speed_span = [x.get('out_speed_span_mibps') for x in per]
    bytes_in = [int(x.get('bytes_in', 0) or 0) for x in per]
    bytes_out = [int(x.get('bytes_out', 0) or 0) for x in per]

    # off-chip 修正（以 span 为基）
    pure_pre_list = []
    pure_post_list = []
    for x in per:
        pre, post = _compute_offchip_adjusted_span_pure(x, theory_mibps)
        pure_pre_list.append(pre)
        pure_post_list.append(post)

    return {
        'count': len(per),
        'in_span_ms_mean': _safe_mean(in_span_ms),
        'out_span_ms_mean': _safe_mean(out_span_ms),
        'io_span_sum_ms_mean': _safe_mean(io_span_sum_ms),
        'in_speed_span_mibps_mean': _safe_mean(in_speed_span),
        'out_speed_span_mibps_mean': _safe_mean(out_speed_span),
        'bytes_in_mean': _safe_mean(bytes_in),
        'bytes_out_mean': _safe_mean(bytes_out),
        'pure_span_ms_mean': _safe_mean(pure_span_ms),
        'pure_span_offchip_adjusted_ms_mean': _safe_mean(pure_post_list),
        'theory_out_mibps_used': theory_mibps,
    }

def main():
    theory = _load_json(THEORY_PATH) or {}
    out_on = {}
    out_off = {}
    out_all = {}

    for model in MODEL_NAMES:
        mdir = RESULTS_BASE / model
        if not mdir.exists():
            continue
        # 判定 seg 是否 off-chip
        theory_segs = (((theory.get(model) or {}).get('segments')) or {})
        model_on = {}
        model_off = {}
        model_all = {}
        for seg in range(1, 9):
            seg_key = f'seg{seg}'
            metrics = collect_segment_span_metrics(mdir, seg)
            if not metrics:
                continue
            tseg = theory_segs.get(seg_key) or {}
            is_off = False
            try:
                is_off = float(tseg.get('off_used_MiB', 0) or 0) > 0
            except Exception:
                is_off = False
            model_all[seg_key] = metrics
            if is_off:
                model_off[seg_key] = metrics
            else:
                model_on[seg_key] = metrics
        if model_on:
            out_on[model] = model_on
        if model_off:
            out_off[model] = model_off
        if model_all:
            out_all[model] = model_all

    # 写文件
    def dump(obj, name):
        p = RESULTS_BASE / name
        with open(p, 'w') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        return str(p)

    p_on = dump(out_on, 'onchip_summary_span.json')
    p_off = dump(out_off, 'offchip_summary_span.json')
    p_all = dump(out_all, 'combined_summary_span.json')
    print(json.dumps({
        'written': {
            'onchip': p_on,
            'offchip': p_off,
            'combined': p_all
        }
    }, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
