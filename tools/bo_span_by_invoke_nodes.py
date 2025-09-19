#!/usr/bin/env python3
"""
按usbmon时间戳顺序找invoke节点，每>100ms gap分界，按模型顺序分组计算Bo span
- 模型顺序: densenet201, inceptionv3, resnet101, resnet50, xception × 20次
- 每个invoke: 第一个Bo S到最后一个Bo C的span + 总bytes
"""
import json, re
from pathlib import Path

def parse_usbmon_with_bo(usb_path):
    """解析usbmon，返回所有时间戳和Bo事件"""
    all_ts = []
    bo_s = []
    bo_c = []
    bo_co = []  # (ts, bytes)
    
    with open(usb_path, 'r', errors='ignore') as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 2:
                continue
            try:
                ts = float(parts[0])  # epoch时间戳
            except:
                continue
            
            all_ts.append(ts)
            
            # 查找Bo事件
            bo_idx = None
            for i, p in enumerate(parts):
                if p.startswith('Bo:'):
                    bo_idx = i
                    break
            if bo_idx is None:
                continue
                
            ev = parts[bo_idx-1] if bo_idx-1 >= 0 else ''
            if ev == 'S':
                bo_s.append(ts)
            elif ev == 'C':
                bo_c.append(ts)
                # 提取bytes
                nb = 0
                m = re.search(r"len=(\d+)", ln)
                if m:
                    nb = int(m.group(1))
                else:
                    try:
                        if bo_idx+2 < len(parts):
                            nb = int(parts[bo_idx+2])
                    except:
                        nb = 0
                bo_co.append((ts, nb))
    
    return sorted(all_ts), sorted(bo_s), sorted(bo_c), sorted(bo_co, key=lambda x: x[0])

def find_invoke_windows(all_ts, gap_ms=100):
    """按>gap_ms的间隔分割invoke窗口"""
    if not all_ts:
        return []
    
    gap_s = gap_ms / 1000.0
    windows = []
    start = all_ts[0]
    prev = start
    
    for ts in all_ts[1:]:
        if ts - prev > gap_s:
            windows.append((start, prev))
            start = ts
        prev = ts
    windows.append((start, prev))
    
    return windows

def compute_bo_span_in_window(bo_s, bo_c, bo_co, window_start, window_end):
    """计算窗口内Bo span和bytes"""
    import bisect
    
    # 找窗口内第一个Bo S
    i = bisect.bisect_left(bo_s, window_start)
    first_s = bo_s[i] if i < len(bo_s) and bo_s[i] <= window_end else None
    
    # 找窗口内最后一个Bo C
    j = bisect.bisect_right(bo_c, window_end) - 1
    last_c = bo_c[j] if j >= 0 and bo_c[j] >= window_start else None
    
    span = 0.0
    total_bytes = 0
    if first_s is not None and last_c is not None and last_c > first_s:
        span = last_c - first_s
        # 仅统计S->C区间的Bo完成字节
        for ts, nb in bo_co:
            if ts < first_s:
                continue
            if ts > last_c:
                break
            total_bytes += int(nb or 0)
    
    return span, total_bytes

def main():
    base = Path('results/full_models_alt_20x_gap100')
    usb_path = str(base / 'usbmon.txt')
    
    print("解析usbmon数据...")
    all_ts, bo_s, bo_c, bo_co = parse_usbmon_with_bo(usb_path)
    print(f"总时间戳: {len(all_ts)}, Bo S: {len(bo_s)}, Bo C: {len(bo_c)}")
    
    print("查找invoke窗口...")
    windows = find_invoke_windows(all_ts, gap_ms=100)
    print(f"找到 {len(windows)} 个窗口")
    
    # 按模型顺序分组 (5模型 × 20次 = 100个invoke)
    model_order = ['densenet201', 'inceptionv3', 'resnet101', 'resnet50', 'xception']
    
    results = []
    model_stats = {model: {'spans': [], 'bytes': [], 'speeds': []} for model in model_order}
    
    MiB = 1024.0 * 1024.0
    
    # 先预计算每个窗口的Bo统计
    pre_results = []
    for i, (ws, we) in enumerate(windows):
        span, total_bytes = compute_bo_span_in_window(bo_s, bo_c, bo_co, ws, we)
        is_valid = span > 0 and total_bytes > 0
        speed = (total_bytes / MiB) / span if is_valid else 0.0
        pre_results.append({
            'window': i,
            'window_start': ws,
            'window_end': we,
            'window_duration_ms': (we - ws) * 1000,
            'bo_span_s': span,
            'bo_bytes': total_bytes,
            'bo_speed_MiBps': speed,
            'valid': is_valid
        })

    # 从第一个有效窗口开始，向后遍历，分配恰好100个“有效+不错位”的invoke到模型轮转；
    # 无效/错位窗口跳过分配但保留记录，避免轮次错位
    first_valid_idx = next((r['window'] for r in pre_results if r['valid']), 0)
    assign_target = 100
    assigned = 0
    traversed = 0
    valid_invoke_idx = 0
    invalid_count = 0
    mismatch_count = 0
    results = []
    i = first_valid_idx
    while assigned < assign_target and i < len(pre_results):
        r = pre_results[i]
        is_valid = r['valid']
        mismatch = False
        model = None
        cycle = None
        if is_valid:
            model = model_order[valid_invoke_idx % len(model_order)]
            cycle = valid_invoke_idx // len(model_order)
            # resnet101的体积特征：若明显过小，视为错位，跳过分配
            if model == 'resnet101' and r['bo_bytes'] < (40 * 1024 * 1024):
                mismatch = True
                mismatch_count += 1
                model = None
                cycle = None
            else:
                # 接受该窗口作为当前模型的一次有效invoke
                model_stats[model]['spans'].append(r['bo_span_s'])
                model_stats[model]['bytes'].append(r['bo_bytes'])
                model_stats[model]['speeds'].append(r['bo_speed_MiBps'])
                valid_invoke_idx += 1
                assigned += 1
        else:
            invalid_count += 1

        r_out = dict(r)
        r_out.update({'model': model, 'cycle': cycle, 'mismatch': mismatch})
        results.append(r_out)
        traversed += 1
        i += 1
    
    # 统计每个模型
    def percentile(values, p):
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        k = p/100.0 * (n-1)
        i = int(k)
        f = k - i
        if i >= n-1:
            return sorted_vals[-1]
        return sorted_vals[i] * (1-f) + sorted_vals[i+1] * f
    
    print(f"\n{'='*60}")
    print("按模型统计 (严格基于有效invoke节点顺序):")
    print(f"{'模型':<12} {'数量':<4} {'均值':<8} {'下限':<8} {'上限':<8} {'p50':<8} {'p95':<8}")
    print("-" * 60)
    
    summary = {}
    for model in model_order:
        speeds = model_stats[model]['speeds']
        if speeds:
            # 加权平均：Σbytes/Σspan
            total_bytes = sum(model_stats[model]['bytes'])
            total_span = sum(model_stats[model]['spans'])
            mean_speed = (total_bytes / total_span) / (1024*1024) if total_span > 0 else 0.0
            mn = min(speeds)
            mx = max(speeds)
            p50 = percentile(speeds, 50)
            p95 = percentile(speeds, 95)
            print(f"{model:<12} {len(speeds):<4} {mean_speed:<8.1f} {mn:<8.1f} {mx:<8.1f} {p50:<8.1f} {p95:<8.1f}")
            summary[model] = {
                'count': len(speeds),
                'weighted_mean': mean_speed,
                'p50': p50,
                'p95': p95,
                'min': mn,
                'max': mx
            }
        else:
            print(f"{model:<12} {0:<4} {'0.0':<8} {'0.0':<8} {'0.0':<8} {'0.0':<8} {'0.0':<8}")
    
    # 保存详细结果
    output = {
        'total_windows': len(windows),
        'start_window': first_valid_idx,
        'traversed_windows': traversed,
        'assigned_invokes': assigned,
        'invalid_windows_skipped': invalid_count,
        'mismatch_windows_skipped': mismatch_count,
        'per_model_summary': summary,
        'detailed_results': results
    }
    
    output_path = base / 'bo_span_by_invoke_nodes.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_path}")
    
    # 显示前几个窗口的详情
    print(f"\n前10个invoke窗口详情 (含无效窗口):")
    print(f"{'#':<3} {'模型':<12} {'窗口时长':<8} {'Bo span':<8} {'字节':<10} {'速度':<8} {'有效':<4}")
    print("-" * 60)
    for i, r in enumerate(results[:10]):
        print(f"{i:<3} {str(r['model']):<12} {r['window_duration_ms']:<8.1f} {r['bo_span_s']:<8.3f} {r['bo_bytes']:<10} {r['bo_speed_MiBps']:<8.1f} {str(r['valid']):<4}")

if __name__ == '__main__':
    main()