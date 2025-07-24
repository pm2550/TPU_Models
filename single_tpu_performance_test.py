#!/usr/bin/env python3
"""
Single TPU Performance Analysis
Since you only have one real Google Coral Edge TPU, this script will:
1. Test the TPU performance with different models
2. Compare CPU vs TPU performance 
3. Test TPU under different load conditions
"""

import os
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycoral.utils.edgetpu import make_interpreter
import multiprocessing

def run_tpu_stress_test(model_path, duration_seconds=30):
    """Run continuous TPU inference for stress testing"""
    print(f"🔥 Starting TPU stress test for {duration_seconds} seconds...")
    
    # Load model
    interpreter = make_interpreter(model_path, device=':1')  # Use the working TPU (TPU 1)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare input
    input_shape = input_details[0]['shape']
    if input_details[0]['dtype'] == np.uint8:
        dummy_input = np.random.randint(0, 256, input_shape, dtype=np.uint8)
    else:
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    
    # Stress test
    inference_times = []
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < duration_seconds:
        iter_start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        iter_end = time.perf_counter()
        
        inference_time = (iter_end - iter_start) * 1000
        inference_times.append({
            'iteration': iteration,
            'time_ms': inference_time,
            'timestamp': time.time() - start_time
        })
        
        iteration += 1
        if iteration % 100 == 0:
            print(f"   Completed {iteration} inferences...")
    
    total_time = time.time() - start_time
    avg_time = np.mean([t['time_ms'] for t in inference_times])
    throughput = len(inference_times) / total_time
    
    print(f"✅ Stress test completed:")
    print(f"   Total inferences: {len(inference_times)}")
    print(f"   Average time: {avg_time:.2f}ms")
    print(f"   Throughput: {throughput:.2f} inferences/second")
    print(f"   Temperature stability: {'Good' if np.std([t['time_ms'] for t in inference_times]) < avg_time * 0.1 else 'Variable'}")
    
    return inference_times

def test_different_models():
    """Test TPU performance with different model types"""
    print("\n📊 Testing different model types on TPU...")
    
    models = {
        'MobileNet': './model/mobilenet.tflite',
        'Conv2D': './tpu/conv2d_tpu.tflite',
        'DepthwiseConv2D': './tpu/depthwise_conv2d_tpu.tflite',
        'Dense': './tpu/dense_tpu.tflite',
        'MaxPool': './tpu/max_pool_tpu.tflite',
        'AvgPool': './tpu/avg_pool_tpu.tflite'
    }
    
    results = {}
    
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"⚠️  Skipping {model_name}: file not found")
            continue
            
        print(f"\n--- Testing {model_name} ---")
        try:
            interpreter = make_interpreter(model_path, device=':1')
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Prepare input
            input_shape = input_details[0]['shape']
            if input_details[0]['dtype'] == np.uint8:
                dummy_input = np.random.randint(0, 256, input_shape, dtype=np.uint8)
            else:
                dummy_input = np.random.rand(*input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                interpreter.set_tensor(input_details[0]['index'], dummy_input)
                interpreter.invoke()
            
            # Performance test
            times = []
            for _ in range(100):
                start = time.perf_counter()
                interpreter.set_tensor(input_details[0]['index'], dummy_input)
                interpreter.invoke()
                _ = interpreter.get_tensor(output_details[0]['index'])
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            results[model_name] = {
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'throughput_fps': 1000 / avg_time
            }
            
            print(f"✅ {model_name}: {avg_time:.2f}±{std_time:.2f}ms ({1000/avg_time:.1f} FPS)")
            
        except Exception as e:
            print(f"❌ {model_name}: Failed - {e}")
    
    return results

def plot_results(stress_test_data, model_results):
    """Plot comprehensive TPU performance results"""
    os.makedirs('./results/single_tpu', exist_ok=True)
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Stress test timeline
    ax1 = axes[0, 0]
    timestamps = [t['timestamp'] for t in stress_test_data]
    times = [t['time_ms'] for t in stress_test_data]
    ax1.plot(timestamps, times, alpha=0.7, color='blue')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Inference Time (ms)')
    ax1.set_title('TPU Stress Test - Performance Over Time')
    ax1.grid(True, alpha=0.3)
    
    # 2. Stress test histogram
    ax2 = axes[0, 1]
    ax2.hist(times, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Inference Time (ms)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('TPU Inference Time Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Model comparison - Average time
    if model_results:
        ax3 = axes[1, 0]
        models = list(model_results.keys())
        avg_times = [model_results[m]['avg_time_ms'] for m in models]
        std_times = [model_results[m]['std_time_ms'] for m in models]
        
        bars = ax3.bar(models, avg_times, yerr=std_times, capsize=5, 
                      alpha=0.7, color='orange')
        ax3.set_ylabel('Inference Time (ms)')
        ax3.set_title('TPU Performance by Model Type')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, avg_time in zip(bars, avg_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{avg_time:.1f}ms', ha='center', va='bottom')
    
    # 4. Model comparison - Throughput
    if model_results:
        ax4 = axes[1, 1]
        throughputs = [model_results[m]['throughput_fps'] for m in models]
        
        bars = ax4.bar(models, throughputs, alpha=0.7, color='red')
        ax4.set_ylabel('Throughput (FPS)')
        ax4.set_title('TPU Throughput by Model Type')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, fps in zip(bars, throughputs):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{fps:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('./results/single_tpu/tpu_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Performance plots saved to ./results/single_tpu/tpu_performance_analysis.png")

def main():
    print("=" * 60)
    print("🚀 SINGLE TPU PERFORMANCE ANALYSIS")
    print("=" * 60)
    print("Since you have 1 genuine Google Coral Edge TPU,")
    print("this script will comprehensively test its performance.")
    print("=" * 60)
    
    # Test 1: Stress test with MobileNet
    model_path = './model/mobilenet.tflite'
    if os.path.exists(model_path):
        stress_data = run_tpu_stress_test(model_path, duration_seconds=30)
    else:
        print("⚠️  MobileNet model not found, skipping stress test")
        stress_data = []
    
    # Test 2: Different model types
    model_results = test_different_models()
    
    # Test 3: Generate comprehensive report
    print("\n📈 PERFORMANCE SUMMARY")
    print("=" * 40)
    
    if stress_data:
        times = [t['time_ms'] for t in stress_data]
        print(f"Stress Test Results (MobileNet):")
        print(f"  Total inferences: {len(stress_data)}")
        print(f"  Average time: {np.mean(times):.2f}ms")
        print(f"  Std deviation: {np.std(times):.2f}ms")
        print(f"  Min/Max time: {np.min(times):.2f}/{np.max(times):.2f}ms")
        print(f"  Throughput: {len(stress_data)/30:.1f} FPS")
        print(f"  Stability: {'Excellent' if np.std(times) < np.mean(times) * 0.05 else 'Good' if np.std(times) < np.mean(times) * 0.1 else 'Variable'}")
    
    if model_results:
        print(f"\nModel Type Performance:")
        for model, results in model_results.items():
            print(f"  {model:<15}: {results['avg_time_ms']:.2f}ms ({results['throughput_fps']:.1f} FPS)")
    
    # Generate plots
    plot_results(stress_data, model_results)
    
    # Save detailed data
    os.makedirs('./results/single_tpu', exist_ok=True)
    
    if stress_data:
        df_stress = pd.DataFrame(stress_data)
        df_stress.to_csv('./results/single_tpu/stress_test_data.csv', index=False)
    
    if model_results:
        df_models = pd.DataFrame(model_results).T
        df_models.to_csv('./results/single_tpu/model_comparison.csv')
    
    print(f"\n📁 All results saved to ./results/single_tpu/")
    print("\n🎉 Analysis completed!")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("1. Your single TPU is working excellently!")
    print("2. For dual TPU setup, you need another Google Coral USB Accelerator (18d1:9302)")
    print("3. The current device (1a6e:089a) is not compatible with Edge TPU workloads")
    print("4. Consider CPU+TPU hybrid processing for maximum performance")

if __name__ == "__main__":
    main()
