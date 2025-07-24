#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树莓派CPU基准测试脚本 - 简化版本（无功耗测量）
"""

import os
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
from tflite_runtime.interpreter import Interpreter

def set_rpi_normal_mode():
    """
    设置树莓派为普通模式，避免超频，使用CPU核心1-3（避免核心0）
    """
    try:
        # 设置CPU调度器为ondemand（普通模式，非超频）
        subprocess.run(['sudo', 'cpufreq-set', '-g', 'ondemand'], 
                      capture_output=True)
        print("Set CPU governor to ondemand mode (normal frequency scaling)")
        
        # 设置CPU亲和性到核心1（避免使用核心0）
        available_cores = list(range(1, os.cpu_count()))  # 使用核心1开始的可用核心
        if available_cores:
            os.sched_setaffinity(0, {available_cores[0]})
            print(f"Process bound to CPU core {available_cores[0]} (avoiding core 0)")
        else:
            print("Warning: Only one CPU core available, using default scheduling")
        
        return True
    except Exception as e:
        print(f"Could not set normal mode: {e}")
        return False

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

def test_model_rpi(model_path, num_runs=1000):
    """
    树莓派优化的模型测试函数（无功耗测量）
    """
    print(f"Testing {os.path.basename(model_path)} on Raspberry Pi...")
    
    # 设置普通模式（非超频）
    set_rpi_normal_mode()
    
    try:
        # 加载模型
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Model has {len(input_details)} inputs and {len(output_details)} outputs")
        
        # 准备输入数据
        dummy_input = prepare_input_data(input_details)
        
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None
    
    # 预热
    print("Warming up (20 runs)...")
    try:
        for i in range(20):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            if i == 0:
                output_shape = interpreter.get_tensor(output_details[0]['index']).shape
                print(f"Output shape: {output_shape}")
    except Exception as e:
        print(f"Error during warmup: {e}")
        return None
    
    # 等待系统稳定
    time.sleep(1.0)
    
    # 正式测试
    inference_times = []
    
    print(f"Running {num_runs} inferences...")
    
    for i in range(num_runs):
        if i % 100 == 0:
            print(f"Progress: {i}/{num_runs}")
            
        try:
            start_time = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
            end_time = time.perf_counter()
            
            inference_times.append((end_time - start_time) * 1000)
        except Exception as e:
            print(f"Error during inference {i}: {e}")
            break
    
    if not inference_times:
        print("No successful inferences recorded")
        return None
    
    # 计算统计数据
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    results = {
        'avg_time_ms': avg_inference_time,
        'std_time_ms': std_inference_time,
        'min_time_ms': np.min(inference_times),
        'max_time_ms': np.max(inference_times),
        'total_runs': len(inference_times)
    }
    
    print(f"Results:")
    print(f"  Average inference time: {avg_inference_time:.3f} ± {std_inference_time:.3f} ms")
    print(f"  Min/Max time: {np.min(inference_times):.3f} / {np.max(inference_times):.3f} ms")
    print(f"  Successful runs: {len(inference_times)}/{num_runs}")
    
    return results

def visualize_rpi_results(all_results, output_dir):
    """
    可视化树莓派测试结果（简化版本）
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
                  alpha=0.8, color='lightblue', edgecolor='navy')
    ax.set_title('Raspberry Pi 5 - Inference Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (ms)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    for bar, time_val, std_val in zip(bars, times, time_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std_val,
                f'{time_val:.2f} ± {std_val:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Raspberry Pi 5 CPU Benchmark Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, "cpu_benchmark.png"), dpi=300, bbox_inches='tight')
    print(f"Raspberry Pi visualization saved to {os.path.join(output_dir, 'cpu_benchmark.png')}")
    plt.show()

def main():
    """
    主函数 - 运行树莓派CPU基准测试
    """
    # 使用外面cpu文件夹中的模型
    model_configs = [
        "conv2d_cpu.tflite",
        "depthwise_conv2d_cpu.tflite", 
        "separable_conv_cpu.tflite",
        "dense_cpu.tflite",
        "max_pool_cpu.tflite",
        "avg_pool_cpu.tflite",
        "feature_pyramid_cpu.tflite",
        "detection_head_cpu.tflite"
    ]
    
    models_dir = "./cpu"
    results = []
    
    print("=" * 70)
    print("🍓 Raspberry Pi 5 CPU Benchmark Test (Simplified)")
    print("=" * 70)
    print("Testing models from ./cpu folder (真正的复合操作块)")
    print()
    
    for model_name in model_configs:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            print(f"\n📊 Testing: {model_name}")
            print("-" * 50)
            
            result = test_model_rpi(model_path, num_runs=1000)
            if result is not None:
                result.update({
                    "model": model_name,
                    "layer_type": model_name.replace("_cpu.tflite", ""),
                    "platform": "CPU"
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
        output_dir = f"./results/cpu_test_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "cpu_benchmark_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved to: {csv_path}")
        
        # 显示汇总
        print("\n" + "=" * 70)
        print("📈 RASPBERRY PI 5 CPU BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Layer Type':<20} | {'Time (ms)':<15} | {'Success':<8}")
        print("-" * 70)
        for result in results:
            print(f"{result['layer_type']:<20} | "
                  f"{result['avg_time_ms']:6.2f}±{result['std_time_ms']:4.2f} | "
                  f"{result['total_runs']:4d}/1000")
        
        # 生成可视化到独立文件夹
        visualize_rpi_results(results, output_dir)
        
        print(f"\n🎉 Raspberry Pi 5 CPU benchmarking completed!")
        print(f"📁 All results saved to: {output_dir}/")
        
        # 显示系统信息
        try:
            temp = subprocess.run(['vcgencmd', 'measure_temp'], 
                                capture_output=True, text=True).stdout.strip()
            volt = subprocess.run(['vcgencmd', 'measure_volts'], 
                                capture_output=True, text=True).stdout.strip()
            print(f"\n🌡️  Final system state: {temp}, {volt}")
        except:
            print("\n🌡️  Could not read system temperature/voltage")
        
    else:
        print("❌ No models found to test. Please check the ./cpu directory.")

if __name__ == "__main__":
    main()