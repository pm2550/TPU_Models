#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import time
from typing import List, Dict, Tuple

import numpy as np
from pycoral.utils.edgetpu import make_interpreter


MN_TPU_DIR = os.path.join(os.getcwd(), 'layered models', 'mn', 'tpu')
RESULT_DIR = os.path.join(os.getcwd(), 'results', 'mn_layers')
os.makedirs(RESULT_DIR, exist_ok=True)


def list_models() -> Tuple[Dict[str, str], Dict[str, str]]:
    # 单独连续模型（联合编译 l1-2, l1-3, ... l1-6）
    series_models = {}
    for k in ['l1-2', 'l1-3', 'l1-4', 'l1-5', 'l1-6']:
        name = f"mnv2_224_{k}_int8_edgetpu.tflite"
        path = os.path.join(MN_TPU_DIR, name)
        if os.path.exists(path):
            series_models[k] = path

    # 单层模型（layer1 ... layer6）
    layer_models = {}
    for i in range(1, 7):
        name = f"mnv2_224_layer{i}_int8_edgetpu.tflite"
        path = os.path.join(MN_TPU_DIR, name)
        if os.path.exists(path):
            layer_models[f"layer{i}"] = path

    return series_models, layer_models


def prepare_input(interpreter) -> Tuple[int, np.ndarray]:
    input_details = interpreter.get_input_details()[0]
    input_index = input_details['index']
    input_shape = input_details['shape']
    input_dtype = input_details['dtype']
    if input_dtype == np.uint8:
        data = np.random.randint(0, 256, input_shape, dtype=np.uint8)
    elif input_dtype == np.int8:
        data = np.random.randint(-128, 128, input_shape, dtype=np.int8)
    else:
        data = np.random.randn(*input_shape).astype(input_dtype)
    return input_index, data


def run_invoke_only(interpreter, input_index: int, input_data: np.ndarray, num_iters: int = 1000) -> List[float]:
    times_ms: List[float] = []
    for _ in range(num_iters):
        interpreter.set_tensor(input_index, input_data)
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    return times_ms


def write_times_csv(path: str, header: List[str], rows: List[List[float]]):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def stats(times_ms: List[float]) -> Dict[str, float]:
    arr = np.array(times_ms, dtype=np.float64)
    return {
        'count': int(arr.size),
        'mean': float(arr.mean()) if arr.size else 0.0,
        'p50': float(np.percentile(arr, 50)) if arr.size else 0.0,
        'p90': float(np.percentile(arr, 90)) if arr.size else 0.0,
        'min': float(arr.min()) if arr.size else 0.0,
        'max': float(arr.max()) if arr.size else 0.0,
    }


def benchmark_series(series_models: Dict[str, str], num_iters: int = 1000):
    print('=== 连续模型（l1-2, l1-3, ..., l1-6）单独1000次仅invoke计时 ===')
    for tag, model_path in sorted(series_models.items()):
        print(f'- 模型 {tag}: {model_path}')
        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()
        input_index, input_data = prepare_input(interpreter)
        times_ms = run_invoke_only(interpreter, input_index, input_data, num_iters=num_iters)
        s = stats(times_ms)
        print(f"  -> mean={s['mean']:.3f} ms, p50={s['p50']:.3f}, p90={s['p90']:.3f}, min={s['min']:.3f}, max={s['max']:.3f}")
        rows = [[i + 1, t] for i, t in enumerate(times_ms)]
        out_csv = os.path.join(RESULT_DIR, f"invoke_times_{tag}.csv")
        write_times_csv(out_csv, ['iter', 'invoke_ms'], rows)


def benchmark_chain(layer_models: Dict[str, str], num_iters: int = 1000):
    if len(layer_models) < 6:
        print('警告: layer1~layer6 不全，跳过串行链式测试')
        return

    print('=== 串行链式（layer1 -> ... -> layer6）1000次，仅invoke计时 ===')

    # 固定顺序
    ordered_keys = [f'layer{i}' for i in range(1, 7)]
    interpreters = []
    input_indices = []
    input_details = []
    output_indices = []

    for key in ordered_keys:
        path = layer_models[key]
        itp = make_interpreter(path)
        itp.allocate_tensors()
        interpreters.append(itp)
        inp = itp.get_input_details()[0]
        out = itp.get_output_details()[0]
        input_indices.append(inp['index'])
        input_details.append(inp)
        output_indices.append(out['index'])

    # 检查形状链路是否连通
    ok_chain = True
    for i in range(5):
        out_shape = interpreters[i].get_output_details()[0]['shape']
        in_shape_next = interpreters[i + 1].get_input_details()[0]['shape']
        if tuple(out_shape) != tuple(in_shape_next):
            ok_chain = False
            break

    if not ok_chain:
        print('❌ 检测到layer模型的输出/输入形状不匹配，无法做真正串联。')
        print('   建议改用联合编译模型(l1-2...l1-6)，或提供按层切分且I/O对齐的模型。')
        return

    # 准备 layer1 输入
    first_input_idx, first_input = prepare_input(interpreters[0])

    per_layer_times_rows: List[List[float]] = []
    # header: iter, l1, l2, l3, l4, l5, l6, total
    for it in range(1, num_iters + 1):
        current_data = first_input
        layer_times: List[float] = []

        for li in range(6):
            itp = interpreters[li]
            inp_idx = input_indices[li]
            # 确保 dtype 匹配
            expected_dtype = input_details[li]['dtype']
            if current_data.dtype != expected_dtype:
                current_data = current_data.astype(expected_dtype, copy=False)

            # 确保形状匹配（若前一层输出与后一层输入不同，尝试直接 set 由 EdgeTPU 报错，属于模型链不匹配情况）
            itp.set_tensor(inp_idx, current_data)
            t0 = time.perf_counter()
            itp.invoke()
            t1 = time.perf_counter()
            layer_times.append((t1 - t0) * 1000.0)

            # 下一层输入
            if li < 5:
                current_data = itp.get_tensor(output_indices[li])

        total_time = sum(layer_times)
        per_layer_times_rows.append([it] + [f"{v:.6f}" for v in layer_times] + [f"{total_time:.6f}"])

    # 写 CSV
    out_csv = os.path.join(RESULT_DIR, 'invoke_times_chain_l1_to_l6.csv')
    header = ['iter'] + [f'l{i}_invoke_ms' for i in range(1, 7)] + ['total_invoke_ms']
    write_times_csv(out_csv, header, per_layer_times_rows)

    # 输出整体统计
    # 统计 total
    total_list = [float(row[-1]) for row in per_layer_times_rows]
    s_total = stats(total_list)
    print(f"总计 -> mean={s_total['mean']:.3f} ms, p50={s_total['p50']:.3f}, p90={s_total['p90']:.3f}, min={s_total['min']:.3f}, max={s_total['max']:.3f}")


def benchmark_chain_est_from_series(series_models: Dict[str, str], layer_models: Dict[str, str], num_iters: int = 1000):
    """
    当按层模型无法串联（I/O不匹配）时，使用联合编译模型序列(l1-2...l1-6)
    估算每层增量耗时：
      layer1 ≈ time(layer1)（若存在按层模型）或 time(l1-2)
      layerk (k>=2) ≈ time(l1-k) - time(l1-(k-1))
    并写出模拟的链路每层耗时与总耗时。
    """
    need_tags = ['l1-2', 'l1-3', 'l1-4', 'l1-5', 'l1-6']
    if not all(t in series_models for t in need_tags):
        print('❌ 无法估算：未找到完整的 l1-2..l1-6 联合模型')
        return

    print('=== 基于联合模型的链路耗时估算（仅invoke）===')

    # 测 layer1（若存在单层模型）
    layer1_path = layer_models.get('layer1') if layer_models else None
    layer1_mean = None
    if layer1_path and os.path.exists(layer1_path):
        it = make_interpreter(layer1_path); it.allocate_tensors()
        idx, data = prepare_input(it)
        tms = run_invoke_only(it, idx, data, num_iters=num_iters)
        layer1_mean = stats(tms)['mean']
        print(f'- layer1: mean={layer1_mean:.3f} ms (来自layer1模型)')

    # 测 l1-2..l1-6 的总体
    series_mean: Dict[str, float] = {}
    for tag in need_tags:
        path = series_models[tag]
        it = make_interpreter(path); it.allocate_tensors()
        idx, data = prepare_input(it)
        tms = run_invoke_only(it, idx, data, num_iters=num_iters)
        m = stats(tms)['mean']
        series_mean[tag] = m
        print(f'- {tag}: mean={m:.3f} ms')

    # 估算每层增量
    per_layer = []
    if layer1_mean is not None:
        per_layer.append(layer1_mean)
        base = layer1_mean
        prev = layer1_mean
    else:
        # 无layer1，使用 l1-2 的均值当作前两层总和，再做差
        per_layer.append(series_mean['l1-2'])
        prev = series_mean['l1-2']

    for k in [2, 3, 4, 5, 6]:
        tag = f'l1-{k}'
        inc = series_mean[tag] - (series_mean[f'l1-{k-1}'] if k > 2 or layer1_mean is not None else 0.0)
        per_layer.append(max(0.0, inc))

    # 写 CSV
    out_csv = os.path.join(RESULT_DIR, 'invoke_times_chain_l1_to_l6_est.csv')
    header = ['layer', 'mean_invoke_ms']
    rows = [[f'l{i}', f'{per_layer[i-1]:.6f}'] for i in range(1, 7)]
    write_times_csv(out_csv, header, rows)
    print(f'已写出估算链路每层耗时: {out_csv}')


def main():
    series_models, layer_models = list_models()

    if not series_models and not layer_models:
        print('未找到分层模型，请检查目录: ', MN_TPU_DIR)
        return

    # 连续模型单独 1000 次
    if series_models:
        benchmark_series(series_models, num_iters=1000)
    else:
        print('未发现 l1-2 ~ l1-6 模型，跳过该部分')

    # 串行链式 1000 次
    if layer_models:
        try:
            benchmark_chain(layer_models, num_iters=1000)
        except Exception as e:
            print(f'⚠️ 真实串联失败: {e}')
    else:
        print('未发现 layer1 ~ layer6 模型，跳过该部分')

    # 若真实串联不可行，尝试基于联合模型估算
    try:
        benchmark_chain_est_from_series(series_models, layer_models, num_iters=500)
    except Exception as e:
        print(f'⚠️ 估算链路失败: {e}')


if __name__ == '__main__':
    main()



