#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycoral.utils.edgetpu import make_interpreter
from tflite_runtime.interpreter import Interpreter

def set_cpu_affinity(cpu_id):
    """设置进程的CPU亲和性"""
    os.sched_setaffinity(0, {cpu_id})
    print(f"Process bound to CPU {cpu_id}")

def run_tpu_inference(tpu_id, cpu_id, model_path, num_runs, delay_seconds, results_queue):
    """在指定TPU和CPU上运行推理测试"""
    try:
        # 设置CPU亲和性
        set_cpu_affinity(cpu_id)
        
        # 延迟启动
        if delay_seconds > 0:
            print(f"TPU {tpu_id}: Waiting {delay_seconds} seconds before starting...")
            time.sleep(delay_seconds)
        
        print(f"TPU {tpu_id}: Starting inference on CPU {cpu_id}")
        
        # 首先加载并invoke一次6m模型（用于缓存预热，不计入统计）
        cache_model_path = "./model/test for cache/7m.tflite"
        if os.path.exists(cache_model_path):
            print(f"TPU {tpu_id}: Pre-loading cache model (6m.tflite)...")
            cache_interpreter = make_interpreter(cache_model_path, device=f':{tpu_id}')
            cache_interpreter.allocate_tensors()
            
            cache_input_details = cache_interpreter.get_input_details()
            cache_input_shape = cache_input_details[0]['shape']
            if cache_input_details[0]['dtype'] == np.uint8:
                cache_dummy_input = np.random.randint(0, 256, cache_input_shape, dtype=np.uint8)
            else:
                cache_dummy_input = np.random.rand(*cache_input_shape).astype(np.float32)
            
            # 执行一次6m模型invoke（不计入统计）
            cache_interpreter.set_tensor(cache_input_details[0]['index'], cache_dummy_input)
            cache_interpreter.invoke()
            print(f"TPU {tpu_id}: Cache model (6m.tflite) pre-loaded successfully")
            
            # 释放缓存模型解释器
            del cache_interpreter
        else:
            print(f"TPU {tpu_id}: Warning - cache model not found: {cache_model_path}")
        
        # 加载主要测试模型到指定TPU  
        interpreter = make_interpreter(model_path, device=f':{tpu_id}')
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # 准备输入数据
        input_shape = input_details[0]['shape']
        if input_details[0]['dtype'] == np.uint8:
            dummy_input = np.random.randint(0, 256, input_shape, dtype=np.uint8)
        else:
            dummy_input = np.random.rand(*input_shape).astype(np.float32)
        
        # 预热主要测试模型
        print(f"TPU {tpu_id}: Warming up main model...")
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
        
        # 正式测试
        inference_times = []
        timestamps = []
        
        print(f"TPU {tpu_id}: Starting {num_runs} inferences...")
        start_time = time.time()
        
        for i in range(num_runs):
            if i % 100 == 0:
                print(f"TPU {tpu_id}: Progress {i}/{num_runs}")
            
            # 记录开始时间
            iter_start = time.perf_counter()
            
            # 执行推理
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            invoke_start = time.perf_counter()
            interpreter.invoke()
            invoke_end = time.perf_counter()
            _ = interpreter.get_tensor(output_details[0]['index'])
            
            iter_end = time.perf_counter()
            
            # 记录时间（毫秒）
            invoke_time = (invoke_end - invoke_start) * 1000
            total_time = (iter_end - iter_start) * 1000
            
            inference_times.append({
                'iteration': i,
                'invoke_time_ms': invoke_time,
                'total_time_ms': total_time,
                'timestamp': time.time() - start_time
            })
            timestamps.append(time.time() - start_time)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # 计算统计数据
        invoke_times = [t['invoke_time_ms'] for t in inference_times]
        total_times = [t['total_time_ms'] for t in inference_times]
        
        avg_invoke_time = np.mean(invoke_times)
        avg_total_time = np.mean(total_times)
        std_invoke_time = np.std(invoke_times)
        std_total_time = np.std(total_times)
        
        results = {
            'tpu_id': tpu_id,
            'cpu_id': cpu_id,
            'num_runs': num_runs,
            'total_duration': total_duration,
            'avg_invoke_time_ms': avg_invoke_time,
            'avg_total_time_ms': avg_total_time,
            'std_invoke_time_ms': std_invoke_time,
            'std_total_time_ms': std_total_time,
            'min_invoke_time_ms': np.min(invoke_times),
            'max_invoke_time_ms': np.max(invoke_times),
            'inference_times': inference_times,
            'timestamps': timestamps
        }
        
        print(f"TPU {tpu_id}: Completed!")
        print(f"TPU {tpu_id}: Average invoke time: {avg_invoke_time:.2f} ms")
        print(f"TPU {tpu_id}: Average total time: {avg_total_time:.2f} ms")
        
        results_queue.put(results)
        
    except Exception as e:
        print(f"TPU {tpu_id}: Error occurred: {e}")
        results_queue.put(None)

def plot_results(results_list):
    """绘制测试结果"""
    if not results_list or any(r is None for r in results_list):
        print("Some processes failed, cannot generate plots")
        return
    
    os.makedirs("./results/dual_tpu", exist_ok=True)
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # 1. Invoke时间对比
    ax1 = axes[0, 0]
    for i, result in enumerate(results_list):
        invoke_times = [t['invoke_time_ms'] for t in result['inference_times']]
        ax1.plot(invoke_times, label=f"TPU {result['tpu_id']}", 
                color=colors[i], alpha=0.7)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Invoke Time (ms)")
    ax1.set_title("Invoke Time Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 时间序列对比
    ax2 = axes[0, 1]
    for i, result in enumerate(results_list):
        invoke_times = [t['invoke_time_ms'] for t in result['inference_times']]
        timestamps = [t['timestamp'] for t in result['inference_times']]
        ax2.scatter(timestamps, invoke_times, label=f"TPU {result['tpu_id']}", 
                   color=colors[i], alpha=0.6, s=1)
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Invoke Time (ms)")
    ax2.set_title("Invoke Time vs Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 直方图对比
    ax3 = axes[0, 2]
    for i, result in enumerate(results_list):
        invoke_times = [t['invoke_time_ms'] for t in result['inference_times']]
        ax3.hist(invoke_times, bins=50, alpha=0.6, label=f"TPU {result['tpu_id']}", 
                color=colors[i])
    ax3.set_xlabel("Invoke Time (ms)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Invoke Time Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 总时间对比
    ax4 = axes[1, 0]
    for i, result in enumerate(results_list):
        total_times = [t['total_time_ms'] for t in result['inference_times']]
        ax4.plot(total_times, label=f"TPU {result['tpu_id']}", 
                color=colors[i], alpha=0.7)
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Total Time (ms)")
    ax4.set_title("Total Time Comparison")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 统计对比
    ax5 = axes[1, 1]
    tpu_ids = [r['tpu_id'] for r in results_list]
    avg_invoke_times = [r['avg_invoke_time_ms'] for r in results_list]
    std_invoke_times = [r['std_invoke_time_ms'] for r in results_list]
    
    x = range(len(tpu_ids))
    bars = ax5.bar(x, avg_invoke_times, yerr=std_invoke_times, 
                   color=[colors[i] for i in range(len(tpu_ids))], 
                   alpha=0.7, capsize=5)
    ax5.set_xlabel("TPU")
    ax5.set_ylabel("Average Invoke Time (ms)")
    ax5.set_title("Average Invoke Time with Std Dev")
    ax5.set_xticks(x)
    ax5.set_xticklabels([f"TPU {tid}" for tid in tpu_ids])
    ax5.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (bar, avg_time) in enumerate(zip(bars, avg_invoke_times)):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_invoke_times[i],
                f'{avg_time:.2f}', ha='center', va='bottom')
    
    # 6. 性能变化趋势
    ax6 = axes[1, 2]
    window_size = 50  # 滑动窗口大小
    for i, result in enumerate(results_list):
        invoke_times = [t['invoke_time_ms'] for t in result['inference_times']]
        # 计算滑动平均
        rolling_avg = []
        for j in range(len(invoke_times)):
            start_idx = max(0, j - window_size + 1)
            end_idx = j + 1
            rolling_avg.append(np.mean(invoke_times[start_idx:end_idx]))
        
        ax6.plot(rolling_avg, label=f"TPU {result['tpu_id']}", 
                color=colors[i], alpha=0.8)
    ax6.set_xlabel("Iteration")
    ax6.set_ylabel(f"Rolling Average ({window_size} samples)")
    ax6.set_title("Performance Trend")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./results/dual_tpu/dual_tpu_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存详细数据
    for result in results_list:
        tpu_id = result['tpu_id']
        df = pd.DataFrame(result['inference_times'])
        df.to_csv(f"./results/dual_tpu/tpu_{tpu_id}_detailed.csv", index=False)

def print_summary(results_list):
    """打印测试摘要"""
    if not results_list or any(r is None for r in results_list):
        print("Some processes failed")
        return
    
    print("\n" + "="*60)
    print("DUAL TPU TEST SUMMARY")
    print("="*60)
    
    for result in results_list:
        print(f"\nTPU {result['tpu_id']} (CPU {result['cpu_id']}):")
        print(f"  Total runs: {result['num_runs']}")
        print(f"  Total duration: {result['total_duration']:.2f} seconds")
        print(f"  Average invoke time: {result['avg_invoke_time_ms']:.2f} ± {result['std_invoke_time_ms']:.2f} ms")
        print(f"  Average total time: {result['avg_total_time_ms']:.2f} ± {result['std_total_time_ms']:.2f} ms")
        print(f"  Min invoke time: {result['min_invoke_time_ms']:.2f} ms")
        print(f"  Max invoke time: {result['max_invoke_time_ms']:.2f} ms")
        print(f"  Throughput: {result['num_runs']/result['total_duration']:.2f} inferences/second")
    
    # 对比分析
    if len(results_list) >= 2:
        print(f"\nCOMPARISON:")
        tpu0_avg = results_list[0]['avg_invoke_time_ms']
        tpu1_avg = results_list[1]['avg_invoke_time_ms']
        diff_percent = abs(tpu0_avg - tpu1_avg) / min(tpu0_avg, tpu1_avg) * 100
        print(f"  Average invoke time difference: {diff_percent:.2f}%")
        
        tpu0_std = results_list[0]['std_invoke_time_ms']
        tpu1_std = results_list[1]['std_invoke_time_ms']
        print(f"  TPU 0 variability (CV): {tpu0_std/tpu0_avg*100:.2f}%")
        print(f"  TPU 1 variability (CV): {tpu1_std/tpu1_avg*100:.2f}%")

def main():
    # 配置
    model_path = "./model/test for cache/mn7.tflite"  # 使用mn6模型进行测试
    # model_path = "./layered models/tpu/conv2d_3x3_stride2_edgetpu.tflite"
    num_runs = 1000
    tpu_configs = [
        {'tpu_id': 0, 'cpu_id': 1, 'delay': 0},    # TPU 0, CPU 1, 立即开始
        {'tpu_id': 1, 'cpu_id': 2, 'delay': 1}     # TPU 1, CPU 2, 延迟1秒
    ]
    
    print("Starting dual TPU interference test...")
    print(f"Cache pre-load model: ./model/test for cache/6m.tflite")
    print(f"Main test model: {model_path}")
    print(f"Runs per TPU: {num_runs}")
    print("="*50)
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    # 创建进程和队列
    processes = []
    results_queue = multiprocessing.Queue()
    
    # 启动进程
    for config in tpu_configs:
        process = multiprocessing.Process(
            target=run_tpu_inference,
            args=(config['tpu_id'], config['cpu_id'], model_path, 
                  num_runs, config['delay'], results_queue)
        )
        process.start()
        processes.append(process)
        print(f"Started process for TPU {config['tpu_id']} on CPU {config['cpu_id']}")
    
    # 等待所有进程完成
    results_list = []
    for _ in range(len(processes)):
        result = results_queue.get()
        if result is not None:
            results_list.append(result)
    
    for process in processes:
        process.join()
    
    # 按TPU ID排序结果
    results_list.sort(key=lambda x: x['tpu_id'])
    
    # 分析和绘图
    print_summary(results_list)
    plot_results(results_list)
    
    # 保存汇总数据
    if results_list:
        os.makedirs("./results/dual_tpu", exist_ok=True)
        summary_data = []
        for result in results_list:
            summary_data.append({
                'tpu_id': result['tpu_id'],
                'cpu_id': result['cpu_id'],
                'avg_invoke_time_ms': result['avg_invoke_time_ms'],
                'std_invoke_time_ms': result['std_invoke_time_ms'],
                'avg_total_time_ms': result['avg_total_time_ms'],
                'std_total_time_ms': result['std_total_time_ms'],
                'total_duration': result['total_duration'],
                'throughput': result['num_runs']/result['total_duration']
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv("./results/dual_tpu/summary.csv", index=False)
        print(f"\nResults saved to ./results/dual_tpu/")

if __name__ == "__main__":
    main()
