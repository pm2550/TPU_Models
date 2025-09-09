#!/usr/bin/env python3
"""
基于原始 run_usbmon_capture_offline.sh 的正确逻辑重写的逐次统计
"""
import json
import sys
import re
import os
import argparse


def parse_records(cap_file):
    """解析 usbmon.txt，返回每行记录字典，用于后续 URB S/C 配对。
    字段：urb_id, ts, ev(S/C), dir(Bi/Bo/Ci/Co), len(bytes, 若无len=则None), dev, ep
    """
    recs = []
    with open(cap_file, 'r', errors='ignore') as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 4:
                continue
            try:
                # 修正：当前 usbmon 格式为 "URB_ID TIMESTAMP ..."，事件为第2列
                urb_id = parts[0]
                ts = float(parts[1])
                ts = (ts/1e6) if ts > 1e6 else ts
                # 某些内核可能把事件放在第2列或第3列，这里先取第2列
                ev = parts[2]
            except Exception:
                # 兼容另一种格式："TIMESTAMP URB_ID ..."
                try:
                    ts = float(parts[0])
                    ts = (ts/1e6) if ts > 1e6 else ts
                    urb_id = parts[1]
                    ev = parts[2] if len(parts) > 2 else 'S'
                except Exception:
                    continue
            
            # 过滤无效 urb_id（必须以十六进制/指针样式开头）
            if not re.match(r'^[0-9a-fA-Fx]+', urb_id):
                continue

            # 查找事件类型及总线/设备/端点 (Bi:BUS:DEV:EP) - 通常在第4列
            tok = None
            dev = None
            ep = None
            idx = None
            # 修正：方向字段通常在第3列，向后扫描以兼容不同内核格式
            for i, t in enumerate(parts[3:], start=3):
                if re.match(r'^[BC][io]:\d+:\d+:\d+', t):
                    tok = t[:2]
                    try:
                        m = re.match(r'^[BC][io]:(\d+):(\d+):(\d+)', t)
                        if m:
                            # bus = int(m.group(1))  # 未使用
                            dev = int(m.group(2))
                            ep = int(m.group(3))
                    except Exception:
                        pass
                    idx = i
                    break
            if not tok:
                continue

            # 查找字节数：优先 len=，否则从固定位置解析
            nb = None
            mlen = re.search(r'len=(\d+)', ln)
            if mlen:
                try:
                    nb = int(mlen.group(1))
                except Exception:
                    nb = None
            elif idx is not None and len(parts) > idx + 2:
                # 没有 len= 时，尝试从方向字段后2位解析字节数
                try:
                    nb = int(parts[idx + 2])
                except Exception:
                    nb = None

            recs.append({
                'urb_id': urb_id,
                'ts': ts,
                'ev': ev,
                'dir': tok,
                'len': nb,
                'dev': dev,
                'ep': ep,
            })

    return recs

def build_urb_pairs(records):
    """将 S/C 事件按 urb_id 配对，返回已配对的 URB 列表。
    每个条目包含：dir(Bi/Bo/Ci/Co)、dev、ep、len(bytes from S)、ts_submit、ts_complete、duration
    若 S 无 len=，则 len 视为 0。
    """
    submit_map = {}
    pairs = []
    for r in records:
        urb_id = r['urb_id']
        ev = r['ev']
        if ev == 'S':
            submit_map[urb_id] = r
        elif ev == 'C':
            s = submit_map.pop(urb_id, None)
            if not s:
                continue
            # 方向以 S 行为准（更稳定）
            direction = s['dir']
            if direction not in ('Bi', 'Bo', 'Ci', 'Co'):
                continue
            dev = s['dev'] if s['dev'] is not None else r['dev']
            ep = s['ep'] if s['ep'] is not None else r['ep']
            length = s['len'] if s['len'] is not None else 0
            ts_submit = s['ts']
            ts_complete = r['ts']
            duration = max(0.0, ts_complete - ts_submit)
            pairs.append({
                'dir': direction,
                'dev': dev,
                'ep': ep,
                'len': length,
                'ts_submit': ts_submit,
                'ts_complete': ts_complete,
                'duration': duration,
            })
    return pairs


def stat_window(pairs, win, usb_ref, bt_ref, epoch_ref=None, extra=0.0, mode: str = 'default', *, allow_dev=None, allow_ep_in=None, allow_ep_out=None):
    """统计单个窗口的传输，完全按照原始脚本逻辑"""
    # 时间对齐：如果没有usbmon_ref但有epoch_ref，用epoch_ref对齐
    if usb_ref is None and epoch_ref is not None:
        # usbmon时间戳是绝对时间，invoke窗口也是绝对时间，直接比较
        b0 = win['begin'] - extra
        e0 = win['end'] + extra
    elif usb_ref is not None and bt_ref is not None:
        # 原始逻辑：转换到usbmon时间轴
        b0 = (win['begin'] - bt_ref) + usb_ref - extra
        e0 = (win['end'] - bt_ref) + usb_ref + extra
    else:
        # 回退：视为绝对时间轴一致（与 show_overlap_positions.py 一致的宽容处理）
        b0 = win['begin'] - extra
        e0 = win['end'] + extra
    
    bi = bo = ci = co = 0
    binb = boutb = 0
    in_times = []
    out_times = []

    if mode == 'bulk_complete':
        # 严格口径：仅统计已配对的 Bulk URB，且完全落入窗口（S>=b0 且 C<=e0）
        for p in pairs:
            d = p['dir']
            if d not in ('Bi', 'Bo'):
                continue
            if allow_dev is not None and p['dev'] is not None and p['dev'] != allow_dev:
                continue
            if d == 'Bi' and allow_ep_in is not None and p['ep'] is not None and p['ep'] != allow_ep_in:
                continue
            if d == 'Bo' and allow_ep_out is not None and p['ep'] is not None and p['ep'] != allow_ep_out:
                continue
            # 修正：不要对 URB 时间再次应用 extra，extra 仅对窗口扩展
            ts_s = p['ts_submit']
            ts_c = p['ts_complete']
            if ts_s >= b0 and ts_c <= e0:
                # 过滤异常小长包
                if p['len'] < 64 and p['duration'] > 0.002:
                    continue
                if d == 'Bi':
                    bi += 1
                    binb += p['len']
                    in_times.append(p['ts_complete'])
                else:
                    bo += 1
                    boutb += p['len']
                    out_times.append(p['ts_complete'])
    else:
        # 兼容旧口径（不推荐）：基于行级别统计
        for r in pairs:
            pass  # 不实现旧逻辑的细节，避免误用
    
    span = e0 - b0
    toMB = lambda x: x / (1024 * 1024.0)
    
    # 计算活跃时长（第一个到最后一个传输的时间跨度）
    in_active_s = max(in_times) - min(in_times) if len(in_times) > 1 else 0.0
    out_active_s = max(out_times) - min(out_times) if len(out_times) > 1 else 0.0
    
    return {
        'span_s': span,
        'Bi': bi, 'Bo': bo, 'Ci': ci, 'Co': co,
        'bytes_in': binb, 'bytes_out': boutb,
        'MBps_in': (toMB(binb) / span if span > 0 else 0.0),
        'MBps_out': (toMB(boutb) / span if span > 0 else 0.0),
        'in_active_s': in_active_s,
        'out_active_s': out_active_s,
        'in_active_MBps': (toMB(binb) / in_active_s if in_active_s > 0 else 0.0),
        'out_active_MBps': (toMB(boutb) / out_active_s if out_active_s > 0 else 0.0)
    }


def main():
    parser = argparse.ArgumentParser(description='正确的逐次 usbmon 统计')
    parser.add_argument('usbmon_txt', help='usbmon.txt 文件路径')
    parser.add_argument('invokes_json', help='invokes.json 文件路径')
    parser.add_argument('time_map_json', help='time_map.json 文件路径')
    parser.add_argument('--details', action='store_true', help='显示每次详情')
    parser.add_argument('--extra', type=float, default=0.0, help='扩展窗口秒数')
    parser.add_argument('--mode', choices=['default','bulk_complete'], default='default', help='统计口径')
    parser.add_argument('--dev', type=int, default=None, help='仅统计指定设备号(DEV)')
    parser.add_argument('--ep-in', dest='ep_in', type=int, default=None, help='仅统计指定IN端点号(EP)')
    parser.add_argument('--ep-out', dest='ep_out', type=int, default=None, help='仅统计指定OUT端点号(EP)')
    
    args = parser.parse_args()
    
    # 加载时间映射
    if os.path.exists(args.time_map_json):
        tm = json.load(open(args.time_map_json))
    else:
        tm = {'usbmon_ref': None, 'boottime_ref': None}
    
    usb_ref = tm.get('usbmon_ref')
    bt_ref = tm.get('boottime_ref')
    epoch_ref = tm.get('epoch_ref_at_usbmon')
    
    print(f"时间映射: usbmon_ref={usb_ref}, boottime_ref={bt_ref}, epoch_ref={epoch_ref}")
    
    # 加载 invoke 窗口
    iv = json.load(open(args.invokes_json))['spans']
    print(f"总 invoke 数: {len(iv)}")
    
    # 解析 usbmon 记录
    records = parse_records(args.usbmon_txt)
    print(f"总 usbmon 记录: {len(records)}")
    pairs = build_urb_pairs(records)
    print(f"已配对 URB: {len(pairs)}")

    # 自动探测设备号/端点号（若未指定），选择字节占比最高的DEV/EP（Bi/Bo各自独立）
    auto_dev = args.dev
    auto_ep_in = args.ep_in
    auto_ep_out = args.ep_out
    if args.dev is None or args.ep_in is None or args.ep_out is None:
        bytes_by_dev = {}
        bytes_by_ep_in = {}
        bytes_by_ep_out = {}
        for p in pairs:
            d = p['dir']
            if d not in ('Bi', 'Bo'):
                continue
            dev = p['dev']; ep = p['ep']; nb = p['len'] or 0
            if dev is not None:
                bytes_by_dev[dev] = bytes_by_dev.get(dev, 0) + nb
            if d == 'Bi' and ep is not None:
                bytes_by_ep_in[ep] = bytes_by_ep_in.get(ep, 0) + nb
            if d == 'Bo' and ep is not None:
                bytes_by_ep_out[ep] = bytes_by_ep_out.get(ep, 0) + nb
        if auto_dev is None and bytes_by_dev:
            auto_dev = max(bytes_by_dev.items(), key=lambda x: x[1])[0]
        if auto_ep_in is None and bytes_by_ep_in:
            auto_ep_in = max(bytes_by_ep_in.items(), key=lambda x: x[1])[0]
        if auto_ep_out is None and bytes_by_ep_out:
            auto_ep_out = max(bytes_by_ep_out.items(), key=lambda x: x[1])[0]
    # 环境变量覆盖（若提供）
    env_dev = os.environ.get('USBMON_DEV')
    if env_dev and env_dev.isdigit():
        auto_dev = int(env_dev)
    print(f"使用过滤: DEV={auto_dev}, EP_IN={auto_ep_in}, EP_OUT={auto_ep_out}")
    
    # 统计每个窗口
    results = []
    for i, win in enumerate(iv):
        stat = stat_window(pairs, win, usb_ref, bt_ref, epoch_ref, args.extra, args.mode,
                           allow_dev=auto_dev, allow_ep_in=auto_ep_in, allow_ep_out=auto_ep_out)
        stat['invoke'] = i
        results.append(stat)
        
        if args.details:
            print(f"invoke {i}: IN={stat['bytes_in']:,}B OUT={stat['bytes_out']:,}B "
                  f"Bi={stat['Bi']} Bo={stat['Bo']} Ci={stat['Ci']} Co={stat['Co']} "
                  f"in_active={stat['in_active_s']:.4f}s out_active={stat['out_active_s']:.4f}s")
    
    # 计算 warm 摘要 (跳过第一次)
    if len(results) > 1:
        warm_results = results[1:]
        total_in = sum(r['bytes_in'] for r in warm_results)
        total_out = sum(r['bytes_out'] for r in warm_results)
        total_in_active = sum(r['in_active_s'] for r in warm_results)
        total_out_active = sum(r['out_active_s'] for r in warm_results)
        non_zero_in = len([r for r in warm_results if r['bytes_in'] > 0])
        non_zero_out = len([r for r in warm_results if r['bytes_out'] > 0])
        
        print(f"\n=== Warm 摘要 (跳过第1次) ===")
        print(f"Warm invoke 数: {len(warm_results)}")
        avg_in = total_in/len(warm_results)
        avg_out = total_out/len(warm_results)
        avg_in_active = total_in_active/len(warm_results)
        avg_out_active = total_out_active/len(warm_results)
        print(f"IN:  总字节={total_in:,}, 平均={avg_in:.0f}, 非零次数={non_zero_in}")
        print(f"OUT: 总字节={total_out:,}, 平均={avg_out:.0f}, 非零次数={non_zero_out}")
        print(f"平均传输: IN={avg_in/1024/1024:.3f}MB, OUT={avg_out/1024/1024:.3f}MB")
        print(f"平均活跃时长: IN={avg_in_active:.4f}s, OUT={avg_out_active:.4f}s")
        print(f"平均活跃速度: IN={avg_in/1024/1024/avg_in_active if avg_in_active > 0 else 0:.1f}MB/s, OUT={avg_out/1024/1024/avg_out_active if avg_out_active > 0 else 0:.1f}MB/s")
        # 机器可解析的 JSON 摘要
        import json as _json
        json_summary = {
            'dev': auto_dev,
            'ep_in': auto_ep_in,
            'ep_out': auto_ep_out,
            'warm_invokes': len(warm_results),
            'warm_avg_in_bytes': avg_in,
            'warm_avg_out_bytes': avg_out,
            'warm_total_in_bytes': total_in,
            'warm_total_out_bytes': total_out,
            'warm_avg_in_active_s': avg_in_active,
            'warm_avg_out_active_s': avg_out_active
        }
        print('JSON_SUMMARY:', _json.dumps(json_summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
