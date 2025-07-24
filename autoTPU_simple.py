#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EdgeTPU基准测试脚本 - 简化版本（无功耗测量）
"""

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

def prepare_input_data(input_details):
    """
    根据模型输入要求准备正确的输入数据
    """
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    print(f"Model input shape: {input_shape}")
    print(f"Model input dtype: {input_dtype}")
    
    # 根据数据类型准备输入
    if input_dtype == np.uint8:
        dummy_input = np.random.randint(0, 256, input_shape, dtype=np.uint8)
    elif input_dtype == np.int8:
        dummy_input = np.random.randint(-128, 128, input_shape, dtype=np.int8)
    elif input_dtype == np.float32:
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
    elif input_dtype == np.float16:
        dummy_input = np.random.rand(*input_shape).astype(np.float16)
    else:
        print(f"Warning: Unsupported input dtype {input_dtype}, using float32")
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
    
    print(f"Generated input shape: {dummy_input.shape}, dtype: {dummy_input.dtype}")
    return dummy_input

def prepare_tpu_interpreter(model_path, warmup=10):
    """准备TPU解释器并进行预热"""
    try:
        interpreter = build_tpu_interpreter(model_path)
        if interpreter is None:
            return None, None, None, None
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error loading TPU model {model_path}: {e}")
        return None, None, None, None
    
    # 获取输入输出信息
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    inp_idx = input_details[0]['index']
    out_idx = output_details[0]['index']
    
    print(f"Model has {len(input_details)} inputs and {len(output_details)} outputs")
    
    # 准备输入数据
    dummy_input = prepare_input_data(input_details)
    
    # 预热 - 确保TPU完全预热
    print(f"Warming up with {warmup} runs...")
    warmup_times = []
    try:
        for i in range(warmup):
            start_time = time.perf_counter()
            interpreter.set_tensor(inp_idx, dummy_input)
            interpreter.invoke()
            _ = interpreter.get_tensor(out_idx)
            end_time = time.perf_counter()
            warmup_time = (end_time - start_time) * 1000
            warmup_times.append(warmup_time)
            
            if i == 0:
                output_shape = interpreter.get_tensor(out_idx).shape
                print(f"  First warmup run: {warmup_time:.1f} ms (cold start)")
                print(f"  Output shape: {output_shape}")
            elif i == warmup - 1:
                print(f"  Last warmup run: {warmup_time:.1f} ms (warmed up)")
    except Exception as e:
        print(f"Error during warmup: {e}")
        return None, None, None, None
    
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
    测试单个TPU模型，返回详细的推理统计信息（无功耗测量）
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
    
    # 执行推理测试
    for i in range(num_runs):
        if i % 100 == 0:
            print(f"Progress: {i}/{num_runs}")
        
        try:
            t_pre, t_infer, t_post, t_total = run_tpu_inference_once(
                interpreter, inp_idx, out_idx, dummy_input
            )
            
            pre_times.append(t_pre)
            infer_times.append(t_infer)
            post_times.append(t_post)
            total_times.append(t_total)
        except Exception as e:
            print(f"Error during inference {i}: {e}")
            break
    
    if not total_times:
        print("No successful inferences recorded")
        return None
    
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
        'cold_start_time': total_times[0] if total_times else 0
    }
    
    print(f"Results:")
    print(f"  Average total time: {stats['avg_time_ms']:.3f} ± {stats['std_time_ms']:.3f} ms")
    print(f"  Average pre time: {stats['avg_pre_ms']:.3f} ms")
    print(f"  Average inference time: {stats['avg_infer_ms']:.3f} ms")
    print(f"  Average post time: {stats['avg_post_ms']:.3f} ms")
    print(f"  Min/Max time: {stats['min_time_ms']:.3f} / {stats['max_time_ms']:.3f} ms")
    print(f"  Successful runs: {len(main_times)}/{num_runs}")
    
    return stats

def visualize_tpu_results(all_results, output_dir):
    """
    可视化EdgeTPU测试结果（简化版本）
    """
    if not all_results:
        print("No results to visualize")
        return
    
    layer_types = [r['layer_type'] for r in all_results]
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 推理时间
    times = [r['avg_time_ms'] for r in all_results]
    time_stds = [r['std_time_ms'] for r in all_results]
    bars = ax.bar(layer_types, times, yerr=time_stds, capsize=5, 
                  alpha=0.8, color='lightgreen', edgecolor='darkgreen')
    ax.set_title('EdgeTPU - Inference Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (ms)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    for bar, time_val, std_val in zip(bars, times, time_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std_val,
                f'{time_val:.2f} ± {std_val:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('EdgeTPU Benchmark Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, "tpu_benchmark.png"), dpi=300, bbox_inches='tight')
    print(f"EdgeTPU visualization saved to {os.path.join(output_dir, 'tpu_benchmark.png')}")
    plt.show()

def main():
    """
    主函数
    """
    # 使用外面tpu文件夹中的模型
    model_configs = [
        "conv2d_tpu.tflite",
        "depthwise_conv2d_tpu.tflite",
        "separable_conv_tpu.tflite", 
        "dense_tpu.tflite",
        "max_pool_tpu.tflite",
        "avg_pool_tpu.tflite",
        "feature_pyramid_tpu.tflite",
        "detection_head_tpu.tflite"
    ]
    
    models_dir = "./tpu"
    results = []
    
    print("=" * 70)
    print("⚡ EdgeTPU Benchmark Test (Simplified)")
    print("=" * 70)
    print("Testing models from ./tpu folder (真正的复合操作块)")
    print()
    
    for model_name in model_configs:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            print(f"\n📊 Testing: {model_name}")
            print("-" * 50)
            
            result = test_model_tpu(model_path, num_runs=1000)
            if result is not None:
                result.update({
                    "model": model_name,
                    "layer_type": model_name.replace("_tpu.tflite", ""),
                    "platform": "TPU"
                })
                results.append(result)
                
                print(f"✅ Completed: {model_name}")
            else:
                print(f"❌ Failed: {model_name}")
            print("-" * 50)
        else:
            print(f"❌ Model not found: {model_path}")
    
    if results:
        # 创建带时间戳的独立文件夹
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./results/tpu_test_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        import pandas as pd
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "tpu_benchmark_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved to: {csv_path}")
        
        # 显示汇总
        print("\n" + "=" * 70)
        print("📈 EDGETPU BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Layer Type':<20} | {'Time (ms)':<15} | {'Success':<8}")
        print("-" * 70)
        for result in results:
            print(f"{result['layer_type']:<20} | "
                  f"{result['avg_time_ms']:6.2f} ± {result['std_time_ms']:4.2f} | "
                  f"{result['total_runs']:4d}/1000")
        
        # 生成可视化到独立文件夹
        visualize_tpu_results(results, output_dir)
        
        print(f"\n🎉 EdgeTPU benchmarking completed!")
        print(f"📁 All results saved to: {output_dir}/")
        
    else:
        print("❌ No models found to test. Please check the ./tpu directory.")

if __name__ == "__main__":
    main()