#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四模型单独推理测试
=================
1. TPU0: EfficientDet Lite2 448 目标检测
2. TPU0/1: DeepLabv3 m0.5 语义分割
3. TPU1: MobileNet (NetVLAD特征提取)
4. CPU: NetVLAD 头部 (WPCA512)

测量每个模型的冷启动和热运行耗时
固定CPU频率，使用模拟输入
"""

import json
import time
import numpy as np
import sys
import os
import torch
from typing import Optional

# pycoral 在某些环境里不可用（缺少 pybind）；这里做兼容回退到 tflite_runtime + edgetpu delegate
try:
    from pycoral.utils.edgetpu import make_interpreter as coral_make_interpreter  # type: ignore
except Exception:
    coral_make_interpreter = None

# 模型路径
# 说明：
# - TPU 模型：*.tflite（edgetpu 编译版）
# - CPU 模型：NetVLAD head（PyTorch checkpoint）
MODELS = {
    # Detector（可选：SSD MobileNet / EfficientDet）
    "ssd_mobilenet_v2": "/home/10210/Desktop/ROS/models/ssd_mobilenet_v2_coco_edgetpu.tflite",
    # 联合编译版（你说的 co-compile/joint）：与 mobilenet_v2 共享编译/缓存，通常更快更稳
    "ssd_mobilenet_v2_joint": "/home/10210/Desktop/OS/real_time_analysis/ssd_mobilenet_v2_coco_quant_postprocess_joint_backbone_edgetpu.tflite",
    "efficientdet_lite2_448": "/home/10210/Desktop/OS/models_local/public/efficientdet_lite2_448_ptq_edgetpu.tflite",

    # Segmentation
    "deeplabv3_dm05": "/home/10210/Desktop/OS/models_local/public/deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite",

    # NetVLAD feature extractor backbone (TPU)
    "mobilenet_v2": "/home/10210/Desktop/OS/models_local/public/mobilenet_v2_1.0_224_quant_edgetpu.tflite",
    "mobilenet_v2_joint": "/home/10210/Desktop/OS/real_time_analysis/mobilenet_v2_1.0_224_quant_joint_ssd_edgetpu.tflite",

    # NetVLAD head (CPU)
    "netvlad_head": "/home/10210/Desktop/ROS/models/mapillary_WPCA512.pth.tar",
}


def _create_interpreter(model_path: str, use_tpu: bool = True):
    """创建解释器并返回输入/输出详情及模拟输入。"""
    if use_tpu:
        if coral_make_interpreter is not None:
            # pycoral 会读取环境变量 EDGETPU_DEVICE（run_burst_measurements 会设置它）
            interpreter = coral_make_interpreter(model_path)
        else:
            # 回退：tflite_runtime + EdgeTPU delegate
            from tflite_runtime.interpreter import Interpreter, load_delegate

            dev = os.environ.get("EDGETPU_DEVICE")
            delegate = load_delegate("libedgetpu.so.1", {"device": dev} if dev else {})
            interpreter = Interpreter(model_path=model_path, experimental_delegates=[delegate], num_threads=1)
    else:
        from tflite_runtime.interpreter import Interpreter
        interpreter = Interpreter(model_path=model_path, num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    dummy = _generate_dummy_input(inp)
    return interpreter, inp, out, dummy


def _generate_dummy_input(inp_detail):
    """根据输入张量信息生成模拟输入。"""
    input_shape = inp_detail['shape']
    input_dtype = inp_detail['dtype']
    if input_dtype == np.uint8:
        return np.random.randint(0, 256, input_shape, dtype=np.uint8)
    if input_dtype == np.int8:
        return np.random.randint(-128, 128, input_shape, dtype=np.int8)
    return np.random.random_sample(input_shape).astype(input_dtype)


def _run_warm_invocations(
    interpreter,
    inp_detail,
    out_detail,
    dummy,
    warm_repeats: int,
    warmup: int,
    sleep_between_ms: Optional[float] = None,
    idle_every: Optional[int] = None,
    idle_duration_ms: Optional[float] = None,
    capture_cycle: bool = False,
):
    """执行预热与正式推理循环，可选周期与空闲控制。"""
    inp_index = inp_detail['index']
    out_index = out_detail['index']

    for _ in range(warmup):
        interpreter.set_tensor(inp_index, dummy)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_index)

    warm_times = []
    cycle_times = [] if capture_cycle else None

    for i in range(warm_repeats):
        cycle_start = time.perf_counter()
        interpreter.set_tensor(inp_index, dummy)
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        _ = interpreter.get_tensor(out_index)
        warm_time_ms = (t1 - t0) * 1000.0
        warm_times.append(warm_time_ms)

        cycle_end = time.perf_counter()
        if capture_cycle:
            cycle_times.append((cycle_end - cycle_start) * 1000.0)

        if sleep_between_ms is not None:
            remain = (sleep_between_ms / 1000.0) - (cycle_end - cycle_start)
            if remain > 0:
                time.sleep(remain)

        if idle_every and idle_duration_ms and (i + 1) % idle_every == 0:
            time.sleep(idle_duration_ms / 1000.0)

    return warm_times, cycle_times


def measure_invoke_only(interpreter, input_tensor):
    """仅测量 invoke 时间"""
    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0  # ms


def test_tpu_model(
    model_path: str,
    model_name: str,
    cold_repeats: int = 10,
    warm_repeats: int = 50,
    warmup: int = 50,
    use_tpu: bool = True,
    sleep_between_ms: Optional[float] = None,
    idle_every: Optional[int] = None,
    idle_duration_ms: Optional[float] = None,
    capture_cycle: bool = False,
):
    """测试 TPU/CPU 模型"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"路径: {model_path}")
    print(f"设备: {'TPU' if use_tpu else 'CPU'}")
    print(f"{'='*60}")
    
    results = {
        'model_name': model_name,
        'model_path': model_path,
        'device': 'TPU' if use_tpu else 'CPU',
        'cold_start_ms': [],
        'warm_run_ms': [],
    }
    
    # 1. 冷启动测试：第一次 invoke 作为冷启动
    print(f"\n[1/2] 冷启动测试 (使用第一次 invoke)...")
    interpreter, inp, out, dummy = _create_interpreter(model_path, use_tpu=use_tpu)
    
    # 冷启动：第一次 invoke
    interpreter.set_tensor(inp['index'], dummy)
    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()
    _ = interpreter.get_tensor(out['index'])
    cold_time = (t1 - t0) * 1000.0
    results['cold_start_ms'].append(cold_time)
    print(f"  冷启动: {cold_time:.2f} ms")
    
    # 2. 热启动测试：继续用同一个解释器
    print(f"\n[2/2] 热运行测试 (预热 {warmup} 次，测量 {warm_repeats} 次)...")
    
    warm_times, cycle_times = _run_warm_invocations(
        interpreter,
        inp,
        out,
        dummy,
        warm_repeats=warm_repeats,
        warmup=warmup,
        sleep_between_ms=sleep_between_ms,
        idle_every=idle_every,
        idle_duration_ms=idle_duration_ms,
        capture_cycle=capture_cycle,
    )

    for idx, warm_time in enumerate(warm_times, start=1):
        results['warm_run_ms'].append(warm_time)
        if idx % 10 == 0:
            print(f"  完成 {idx}/{warm_repeats} 次")

    if cycle_times is not None:
        results['cycle_ms'] = cycle_times
    
    # 统计
    cold_avg = results['cold_start_ms'][0]  # 只有一次冷启动
    warm_avg = np.mean(results['warm_run_ms'])
    warm_std = np.std(results['warm_run_ms'])
    
    print(f"\n📊 统计结果:")
    print(f"  冷启动: {cold_avg:.2f} ms (n=1)")
    print(f"  热运行: {warm_avg:.2f} ± {warm_std:.2f} ms (n={warm_repeats})")
    print(f"  加速比: {cold_avg/warm_avg:.2f}x")
    
    results['statistics'] = {
        'cold_avg_ms': cold_avg,
        'cold_std_ms': 0.0,
        'warm_avg_ms': warm_avg,
        'warm_std_ms': warm_std,
        'speedup': cold_avg / warm_avg if warm_avg > 0 else 0,
    }
    
    if cycle_times is not None:
        results.setdefault('statistics', {})['cycle_avg_ms'] = float(np.mean(cycle_times)) if cycle_times else 0.0
        results['statistics']['cycle_std_ms'] = float(np.std(cycle_times)) if cycle_times else 0.0

    results['metadata'] = {
        'sleep_between_ms': sleep_between_ms,
        'idle_every': idle_every,
        'idle_duration_ms': idle_duration_ms,
        'warmup': warmup,
        'warm_repeats': warm_repeats,
    }

    return results


def test_cpu_netvlad(
    model_path: str,
    model_name: str,
    cold_repeats: int = 10,
    warm_repeats: int = 50,
    warmup: int = 50,
    sleep_between_ms: Optional[float] = None,
    idle_every: Optional[int] = None,
    idle_duration_ms: Optional[float] = None,
    capture_cycle: bool = False,
):
    """测试 CPU NetVLAD 头部"""
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"路径: {model_path}")
    print(f"{'='*60}")
    
    results = {
        'model_name': model_name,
        'model_path': model_path,
        'cold_start_ms': [],
        'warm_run_ms': [],
    }
    
    # 预先生成固定的权重，避免每次 randn
    n_clusters = 64
    descriptor_dim = 1280
    conv_weights = torch.randn(n_clusters, descriptor_dim, 1, 1)
    cluster_centers = [torch.randn(descriptor_dim, 1) for _ in range(n_clusters)]
    
    # 1. 冷启动测试：只测第一次（包含加载 checkpoint）
    print(f"\n[1/2] 冷启动测试 (第一次加载)...")
    t0 = time.perf_counter()
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 模拟 NetVLAD 输入
    input_features = torch.randn(1, 1280, 7, 7)
    
    # NetVLAD 聚合
    soft_assign = torch.nn.functional.conv2d(input_features, conv_weights)
    soft_assign = torch.nn.functional.softmax(soft_assign, dim=1)
    feature_flat = input_features.view(1, descriptor_dim, -1)
    soft_assign_flat = soft_assign.view(1, n_clusters, -1)
    
    residuals = []
    for k in range(n_clusters):
        residual = (feature_flat - cluster_centers[k]) * soft_assign_flat[:, k:k+1, :]
        residuals.append(residual.sum(dim=2))
    
    vlad = torch.cat(residuals, dim=1)
    vlad = torch.nn.functional.normalize(vlad, p=2, dim=1)
    
    # WPCA projection
    if 'WPCA' in checkpoint:
        wpca_matrix = checkpoint['WPCA']
        if isinstance(wpca_matrix, np.ndarray):
            wpca_matrix = torch.from_numpy(wpca_matrix).float()
        if wpca_matrix.shape[0] == vlad.shape[1]:
            final_descriptor = torch.matmul(vlad, wpca_matrix.t())
        else:
            final_descriptor = vlad[:, :512]
    else:
        final_descriptor = vlad[:, :512]
    
    final_descriptor = torch.nn.functional.normalize(final_descriptor, p=2, dim=1)
    
    t1 = time.perf_counter()
    cold_time = (t1 - t0) * 1000.0
    results['cold_start_ms'].append(cold_time)
    print(f"  冷启动 (含加载): {cold_time:.2f} ms")
    
    # 2. 热启动测试：checkpoint 已加载，只测推理
    print(f"\n[2/2] 热运行测试 (预热 {warmup} 次，测量 {warm_repeats} 次)...")
    
    def _netvlad_forward():
        soft_assign_local = torch.nn.functional.conv2d(input_features, conv_weights)
        soft_assign_local = torch.nn.functional.softmax(soft_assign_local, dim=1)
        feature_flat_local = input_features.view(1, descriptor_dim, -1)
        soft_assign_flat_local = soft_assign_local.view(1, n_clusters, -1)
        residuals_local = []
        for k_local in range(n_clusters):
            residual_local = (feature_flat_local - cluster_centers[k_local]) * soft_assign_flat_local[:, k_local:k_local+1, :]
            residuals_local.append(residual_local.sum(dim=2))
        vlad_local = torch.cat(residuals_local, dim=1)
        vlad_local = torch.nn.functional.normalize(vlad_local, p=2, dim=1)
        if 'WPCA' in checkpoint:
            wpca_matrix_local = checkpoint['WPCA']
            if isinstance(wpca_matrix_local, np.ndarray):
                wpca_matrix_local = torch.from_numpy(wpca_matrix_local).float()
            if wpca_matrix_local.shape[0] == vlad_local.shape[1]:
                final_descriptor_local = torch.matmul(vlad_local, wpca_matrix_local.t())
            else:
                final_descriptor_local = vlad_local[:, :512]
        else:
            final_descriptor_local = vlad_local[:, :512]
        return torch.nn.functional.normalize(final_descriptor_local, p=2, dim=1)

    # 预热（不计时）
    for _ in range(warmup):
        _netvlad_forward()

    # 测量（计时仅推理部分）
    cycle_times = [] if capture_cycle else None
    for i in range(warm_repeats):
        cycle_start = time.perf_counter()
        t0 = time.perf_counter()
        _netvlad_forward()
        t1 = time.perf_counter()
        warm_time = (t1 - t0) * 1000.0
        results['warm_run_ms'].append(warm_time)

        cycle_end = time.perf_counter()
        if capture_cycle:
            cycle_times.append((cycle_end - cycle_start) * 1000.0)

        if sleep_between_ms is not None:
            remain = (sleep_between_ms / 1000.0) - (cycle_end - cycle_start)
            if remain > 0:
                time.sleep(remain)

        if idle_every and idle_duration_ms and (i + 1) % idle_every == 0:
            time.sleep(idle_duration_ms / 1000.0)

        if (i + 1) % 10 == 0:
            print(f"  完成 {i+1}/{warm_repeats} 次")
    
    # 统计
    cold_avg = results['cold_start_ms'][0]
    warm_avg = np.mean(results['warm_run_ms'])
    warm_std = np.std(results['warm_run_ms'])
    
    print(f"\n📊 统计结果:")
    print(f"  冷启动: {cold_avg:.2f} ms (含加载，n=1)")
    print(f"  热运行: {warm_avg:.2f} ± {warm_std:.2f} ms (n={warm_repeats})")
    print(f"  加速比: {cold_avg/warm_avg:.2f}x")
    
    results['statistics'] = {
        'cold_avg_ms': cold_avg,
        'cold_std_ms': 0.0,
        'warm_avg_ms': warm_avg,
        'warm_std_ms': warm_std,
        'speedup': cold_avg / warm_avg if warm_avg > 0 else 0,
    }
    
    if cycle_times is not None:
        results.setdefault('statistics', {})['cycle_avg_ms'] = float(np.mean(cycle_times)) if cycle_times else 0.0
        results['statistics']['cycle_std_ms'] = float(np.std(cycle_times)) if cycle_times else 0.0
        results['cycle_ms'] = cycle_times

    results['metadata'] = {
        'sleep_between_ms': sleep_between_ms,
        'idle_every': idle_every,
        'idle_duration_ms': idle_duration_ms,
        'warmup': warmup,
        'warm_repeats': warm_repeats,
    }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='四模型单独推理测试')
    parser.add_argument('--cold-repeats', type=int, default=10, help='冷启动重复次数')
    parser.add_argument('--warm-repeats', type=int, default=50, help='热运行重复次数')
    parser.add_argument('--warmup', type=int, default=50, help='热运行前的预热次数')
    parser.add_argument('--output', type=str, default='four_models_benchmark.json', help='输出JSON文件')
    parser.add_argument('--models', type=str, nargs='+',
                        choices=['ssd_mobilenet_v2', 'ssd_mobilenet_v2_joint', 'efficientdet_lite2_448', 'deeplabv3_dm05', 'mobilenet_v2', 'mobilenet_v2_joint', 'netvlad_head', 'all'],
                        default=['all'], help='选择要测试的模型')
    args = parser.parse_args()
    
    # 确定要测试的模型
    if 'all' in args.models:
        # 默认四模型：SSD MobileNet V2 + DeepLab + MobileNetV2(backbone) + NetVLAD head
        # 注意：如果你想跑联合编译版，把下面两个键替换成 *_joint 即可
        models_to_test = ['ssd_mobilenet_v2', 'deeplabv3_dm05', 'mobilenet_v2', 'netvlad_head']
    else:
        models_to_test = args.models
    
    print(f"\n🚀 四模型单独推理测试")
    print(f"{'='*60}")
    print(f"冷启动重复: {args.cold_repeats} 次")
    print(f"热运行重复: {args.warm_repeats} 次")
    print(f"预热次数: {args.warmup} 次")
    print(f"测试模型: {', '.join(models_to_test)}")
    print(f"{'='*60}")
    
    all_results = {}
    
    # 测试每个模型
    for model_key in models_to_test:
        model_path = MODELS[model_key]
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            print(f"\n⚠️  模型文件不存在: {model_path}")
            continue
        
        try:
            if model_key == 'netvlad_head':
                results = test_cpu_netvlad(
                    model_path, 
                    model_key,
                    cold_repeats=args.cold_repeats,
                    warm_repeats=args.warm_repeats,
                    warmup=args.warmup
                )
            elif model_key == 'deeplabv3_dm05':
                # DeepLabv3 使用 TPU 版本
                results = test_tpu_model(
                    model_path, 
                    model_key,
                    cold_repeats=args.cold_repeats,
                    warm_repeats=args.warm_repeats,
                    warmup=args.warmup,
                    use_tpu=True
                )
            else:
                # SSD 和 MobileNet 使用 TPU
                results = test_tpu_model(
                    model_path, 
                    model_key,
                    cold_repeats=args.cold_repeats,
                    warm_repeats=args.warm_repeats,
                    warmup=args.warmup,
                    use_tpu=True
                )
            
            all_results[model_key] = results
            
        except Exception as e:
            print(f"\n❌ 测试失败 ({model_key}): {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    output_path = os.path.join('/home/10210/Desktop/OS/results', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {output_path}")
    
    # 打印汇总
    print(f"\n{'='*60}")
    print(f"📊 汇总统计")
    print(f"{'='*60}")
    for model_key, results in all_results.items():
        stats = results.get('statistics', {})
        print(f"\n{model_key}:")
        print(f"  冷启动: {stats.get('cold_avg_ms', 0):.2f} ± {stats.get('cold_std_ms', 0):.2f} ms")
        print(f"  热运行: {stats.get('warm_avg_ms', 0):.2f} ± {stats.get('warm_std_ms', 0):.2f} ms")
        print(f"  加速比: {stats.get('speedup', 0):.2f}x")


if __name__ == '__main__':
    main()
