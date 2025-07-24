#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tflite_runtime.interpreter import Interpreter
try:
    from pycoral.utils.edgetpu import make_interpreter
    from pycoral.utils import edgetpu
except ImportError:
    print("PyCoral not installed. Please install: pip install pycoral")
    exit(1)

def build_tpu_interpreter(model_path):
    """构建TPU解释器，优先选择USB 3.0设备"""
    from pycoral.utils import edgetpu
    
    # 获取所有可用的EdgeTPU设备
    devices = edgetpu.list_edge_tpus()
    
    if not devices:
        print("❌ 未找到任何EdgeTPU设备")
        return None
    
    print(f"发现 {len(devices)} 个EdgeTPU设备:")
    
    # 寻找USB 3.0设备
    usb3_device_index = None
    for i, device in enumerate(devices):
        path = device['path']
        try:
            with open(f'{path}/speed', 'r') as f:
                speed = f.read().strip()
            
            usb_type = "USB 3.0 SuperSpeed" if speed == '5000' else "USB 2.0 High Speed"
            print(f"  设备 {i}: {path} ({usb_type}, {speed} Mbps)")
            
            # 优先选择USB 3.0设备
            if speed == '5000' and usb3_device_index is None:
                usb3_device_index = i
                
        except Exception as e:
            print(f"  设备 {i}: {path} (无法读取速度信息)")
    
    # 选择设备策略
    if usb3_device_index is not None:
        print(f"✅ 选择USB 3.0设备 (索引: {usb3_device_index}) 获得最佳性能")
        try:
            # 通过设备索引指定USB 3.0设备
            device_spec = f":{usb3_device_index}"
            return make_interpreter(model_path, device=device_spec)
        except Exception as e:
            print(f"⚠️  USB 3.0设备加载失败，使用默认设备: {e}")
            return make_interpreter(model_path)
    else:
        print("⚠️  未找到USB 3.0设备，使用默认设备 (性能可能受限)")
        return make_interpreter(model_path)

def prepare_tpu_interpreter(model_path, warmup=10):
    """准备TPU解释器并进行预热"""
    try:
        interpreter = build_tpu_interpreter(model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error loading TPU model {model_path}: {e}")
        return None, None, None, None
    
    # 获取输入输出信息
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    inp_idx = input_details[0]['index']
    out_idx = output_details[0]['index']
    
    # 准备输入数据
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    print(f"Model expects input dtype: {input_dtype}")
    
    if input_dtype == np.uint8:
        dummy_input = np.random.randint(0, 256, input_shape, dtype=np.uint8)
    else:
        dummy_input = np.random.rand(*input_shape).astype(input_dtype)
    
    # 预热 - 确保TPU完全预热
    print(f"Warming up with {warmup} runs...")
    warmup_times = []
    for i in range(warmup):
        start_time = time.perf_counter()
        interpreter.set_tensor(inp_idx, dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_idx)
        end_time = time.perf_counter()
        warmup_time = (end_time - start_time) * 1000
        warmup_times.append(warmup_time)
        
        if i == 0:
            print(f"  First warmup run: {warmup_time:.1f} ms (cold start)")
        elif i == warmup - 1:
            print(f"  Last warmup run: {warmup_time:.1f} ms (warmed up)")
    
    # 验证预热效果
    avg_warmup = np.mean(warmup_times[1:])  # 排除第一次
    print(f"  Average warmup time (excluding first): {avg_warmup:.1f} ms")
    
    return interpreter, inp_idx, out_idx, dummy_input

def run_tpu_inference_once(interpreter, inp_idx, out_idx, dummy_input):
    """执行一次TPU推理并返回各阶段时间"""
    # 测量各个阶段的时间
    t_a = time.perf_counter_ns()
    interpreter.set_tensor(inp_idx, dummy_input)
    t_b = time.perf_counter_ns()
    
    interpreter.invoke()
    t_c = time.perf_counter_ns()
    
    _ = interpreter.get_tensor(out_idx)
    t_d = time.perf_counter_ns()
    
    # 转换为毫秒
    return ((t_b - t_a) / 1e6,  # pre: 设置输入时间
            (t_c - t_b) / 1e6,  # infer: 推理时间
            (t_d - t_c) / 1e6,  # post: 获取输出时间
            (t_d - t_a) / 1e6)  # total: 总时间

def test_model_tpu(model_path, num_runs=1000):
    """
    测试单个TPU模型，返回详细的推理统计信息
    """
    print(f"Testing {os.path.basename(model_path)} with {num_runs} runs...")
    
    # 准备解释器
    interpreter, inp_idx, out_idx, dummy_input = prepare_tpu_interpreter(model_path, warmup=10)
    if interpreter is None:
        return None
    
    # 存储各阶段时间
    pre_times = []
    infer_times = []
    post_times = []
    total_times = []
    
    print(f"Running {num_runs} inferences...")
    print("Monitor your power meter now and record the reading!")
    
    # 执行推理测试
    for i in range(num_runs):
        if i % 100 == 0:
            print(f"Progress: {i}/{num_runs}")
        
        t_pre, t_infer, t_post, t_total = run_tpu_inference_once(
            interpreter, inp_idx, out_idx, dummy_input
        )
        
        pre_times.append(t_pre)
        infer_times.append(t_infer)
        post_times.append(t_post)
        total_times.append(t_total)
    
    # 检测异常的冷启动
    if total_times[0] > 10:  # 如果第一次超过10ms，可能是冷启动
        print(f"⚠️  Detected cold start: first inference = {total_times[0]:.1f} ms")
        
        # 提供两种统计：包含和排除冷启动
        times_without_coldstart = total_times[1:]
        print(f"  Statistics including cold start:")
        print(f"    Average: {np.mean(total_times):.3f} ms")
        print(f"  Statistics excluding cold start:")
        print(f"    Average: {np.mean(times_without_coldstart):.3f} ms")
        
        # 使用排除冷启动的统计作为主要结果
        main_times = times_without_coldstart
    else:
        main_times = total_times
    
    # 计算统计信息
    stats = {
        'avg_time_ms': np.mean(main_times),
        'std_time_ms': np.std(main_times),
        'min_time_ms': np.min(main_times),
        'max_time_ms': np.max(main_times),
        'avg_pre_ms': np.mean(pre_times[1:] if len(pre_times) > 1 else pre_times),
        'avg_infer_ms': np.mean(infer_times[1:] if len(infer_times) > 1 else infer_times),
        'avg_post_ms': np.mean(post_times[1:] if len(post_times) > 1 else post_times),
        'total_runs': len(main_times),
        'cold_start_detected': total_times[0] > 10 if total_times else False,
        'cold_start_time': total_times[0] if total_times else 0,
        'pre_times': pre_times,
        'infer_times': infer_times,
        'post_times': post_times,
        'total_times': total_times
    }
    
    print(f"Results:")
    print(f"  Average total time: {stats['avg_time_ms']:.3f} ± {stats['std_time_ms']:.3f} ms")
    print(f"  Average pre time: {stats['avg_pre_ms']:.3f} ms")
    print(f"  Average inference time: {stats['avg_infer_ms']:.3f} ms")
    print(f"  Average post time: {stats['avg_post_ms']:.3f} ms")
    print(f"  Min/Max time: {stats['min_time_ms']:.3f} / {stats['max_time_ms']:.3f} ms")
    
    return stats

def plot_tpu_inference_segments(pre_times, infer_times, post_times, model_name):
    """绘制TPU推理各阶段时间分析图"""
    os.makedirs("./results", exist_ok=True)
    
    # 保存详细数据
    df = pd.DataFrame({
        "pre_ms": pre_times,
        "infer_ms": infer_times, 
        "post_ms": post_times
    })
    df.to_csv(f"./results/{model_name}_tpu_segments.csv", index=False)
    
    # 绘制各阶段时间图
    plt.figure(figsize=(12, 8))
    
    # 子图1: 时间序列
    plt.subplot(2, 2, 1)
    plt.plot(df.index, df["pre_ms"], label="Pre (load)", alpha=0.7)
    plt.plot(df.index, df["infer_ms"], label="Inference", alpha=0.7)
    plt.plot(df.index, df["post_ms"], label="Post (save)", alpha=0.7)
    plt.xlabel("Inference #")
    plt.ylabel("Time (ms)")
    plt.title(f"{model_name} - TPU Inference Segments")
    plt.legend()
    plt.grid(True)
    
    # 子图2: 时间分布直方图
    plt.subplot(2, 2, 2)
    plt.hist([pre_times, infer_times, post_times], 
             bins=30, alpha=0.7, 
             label=['Pre', 'Inference', 'Post'])
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency")
    plt.title("Time Distribution")
    plt.legend()
    plt.grid(True)
    
    # 子图3: 总时间分布
    total_times = [p + i + o for p, i, o in zip(pre_times, infer_times, post_times)]
    plt.subplot(2, 2, 3)
    plt.hist(total_times, bins=30, alpha=0.7, color='green')
    plt.xlabel("Total Time (ms)")
    plt.ylabel("Frequency")
    plt.title("Total Inference Time Distribution")
    plt.grid(True)
    
    # 子图4: 各阶段平均时间对比
    plt.subplot(2, 2, 4)
    avg_times = [np.mean(pre_times), np.mean(infer_times), np.mean(post_times)]
    stages = ['Pre', 'Inference', 'Post']
    bars = plt.bar(stages, avg_times, alpha=0.8, color=['blue', 'orange', 'green'])
    plt.ylabel("Average Time (ms)")
    plt.title("Average Time by Stage")
    plt.grid(True)
    
    # 添加数值标签
    for bar, avg_time in zip(bars, avg_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg_time:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"./results/{model_name}_tpu_detailed.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed analysis saved to ./results/{model_name}_tpu_detailed.png")

def visualize_tpu_results(all_results):
    """可视化TPU测试结果的函数 - 暂时不实现"""
    pass

def main():
    # TPU模型配置
    model_configs = [
        "conv2d_tpu.tflite",
        "depthwise_conv2d_tpu.tflite",
        "separable_conv_tpu.tflite",
        "max_pool_tpu.tflite",
        "avg_pool_tpu.tflite",
        "dense_tpu.tflite",
        "feature_pyramid_tpu.tflite",
        "detection_head_tpu.tflite"
    ]
    
    models_dir = "./tpu"
    results = []
    
    print("Starting TPU benchmarking...")
    print("=" * 50)
    print("Note: You will be asked to enter power consumption after each test")
    print("Power consumption unit: Watts (W)")
    print("Example: If your power meter shows 2.5W during test, enter: 2.5")
    print("=" * 50)
    
    for model_name in model_configs:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            print(f"\n{'='*20}")
            print(f"Testing: {model_name}")
            print(f"{'='*20}")
            
            # 等待用户确认开始测试
            input("Press Enter to start testing this model (prepare your power meter)...")
            
            stats = test_model_tpu(model_path)
            
            if stats is not None:
                avg_time = stats['avg_time_ms']
                # 测试完成后输入功耗数据
                while True:
                    try:
                        power_input = input(f"Please enter the power consumption you observed for {model_name} (in Watts, e.g., 2.5): ")
                        avg_power = float(power_input)
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number (e.g., 2.5).")
                results.append({
                    "model": model_name,
                    "layer_type": model_name.replace("_tpu.tflite", ""),
                    "avg_time_ms": avg_time,
                    "avg_power_w": avg_power,
                    "platform": "TPU"
                })
                
                print(f"Recorded - Time: {avg_time:.2f} ms, Power: {avg_power:.2f} W")
            else:
                print(f"Failed to test {model_name}")
            
            print(f"Completed: {model_name}")
            print("-" * 40)
        else:
            print(f"Model not found: {model_path}")
    
    # 保存结果
    os.makedirs("./results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("./results/tpu_benchmark_results.csv", index=False)
    print(f"\nResults saved to ./results/tpu_benchmark_results.csv")
    
    # 简单可视化
    if results:  # 只有在有结果的情况下才进行可视化
        plt.figure(figsize=(14, 10))
        
        plt.subplot(2, 1, 1)
        layer_types = [r["layer_type"] for r in results]
        times = [r["avg_time_ms"] for r in results]
        plt.bar(layer_types, times, color='orange', alpha=0.8)
        plt.xlabel("Layer Type")
        plt.ylabel("Average Time (ms)")
        plt.title("TPU Inference Time by Layer Type")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(times):
            plt.text(i, v + max(times)*0.01, f'{v:.2f}', ha='center', va='bottom')
        
        plt.subplot(2, 1, 2)
        powers = [r["avg_power_w"] for r in results]
        plt.bar(layer_types, powers, color='green', alpha=0.8)
        plt.xlabel("Layer Type")
        plt.ylabel("Average Power (W)")
        plt.title("TPU Power Consumption by Layer Type")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(powers):
            plt.text(i, v + max(powers)*0.01, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig("./results/tpu_benchmark.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印总结
        print("\n" + "="*50)
        print("TPU BENCHMARK SUMMARY")
        print("="*50)
        for result in results:
            print(f"{result['layer_type']:20s}: {result['avg_time_ms']:8.2f} ms, {result['avg_power_w']:6.2f} W")
    else:
        print("No results to display.")
    
    print("TPU benchmarking completed!")

if __name__ == "__main__":
    main()