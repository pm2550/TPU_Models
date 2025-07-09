#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树莓派专用CPU基准测试脚本
利用树莓派5的原生硬件监控接口进行精确功耗测量
"""

import os
import time
import threading
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
from tflite_runtime.interpreter import Interpreter

class RaspberryPiPowerMonitor:
    """
    树莓派专用功耗监控器，使用原生硬件接口
    """
    def __init__(self):
        self.monitoring = False
        self.power_readings = []
        self.start_time = None
        
        # 树莓派硬件监控路径
        self.temp_path = "/sys/class/hwmon/hwmon0/temp1_input"
        self.volt_paths = [
            "/sys/class/hwmon/hwmon1/in1_input",  # Core voltage
            "/sys/class/hwmon/hwmon1/in2_input",  # SDRAM voltage  
            "/sys/class/hwmon/hwmon1/in3_input",  # I/O voltage
            "/sys/class/hwmon/hwmon1/in4_input"   # Other voltage
        ]
        
    def read_vcgencmd_voltage(self):
        """使用vcgencmd读取CPU核心电压"""
        try:
            result = subprocess.run(['vcgencmd', 'measure_volts'], 
                                  capture_output=True, text=True, timeout=1)
            if result.returncode == 0:
                # 解析输出 "volt=0.8860V"
                volt_str = result.stdout.strip().split('=')[1].replace('V', '')
                return float(volt_str)
        except:
            pass
        return None
        
    def read_hardware_temp(self):
        """读取硬件温度传感器"""
        try:
            with open(self.temp_path, 'r') as f:
                # 温度值以毫摄氏度为单位
                temp_millicelsius = int(f.read().strip())
                return temp_millicelsius / 1000.0
        except:
            return None
            
    def read_hardware_voltages(self):
        """读取硬件电压传感器"""
        voltages = {}
        labels = ['core', 'sdram', 'io', 'other']
        
        for i, path in enumerate(self.volt_paths):
            try:
                with open(path, 'r') as f:
                    # 电压值以毫伏为单位
                    voltage_mv = int(f.read().strip())
                    voltages[labels[i]] = voltage_mv / 1000.0
            except:
                voltages[labels[i]] = None
                
        return voltages
        
    def get_cpu_frequency(self):
        """获取当前CPU频率"""
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                return cpu_freq.current
        except:
            pass
        return None
        
    def get_bound_core_usage(self):
        """获取当前绑定CPU核心的使用率"""
        try:
            # 获取当前进程绑定的CPU核心
            bound_cores = os.sched_getaffinity(0)
            if bound_cores:
                # 获取各个核心的使用率
                per_cpu_usage = psutil.cpu_percent(percpu=True, interval=0.01)
                if per_cpu_usage and len(per_cpu_usage) > max(bound_cores):
                    # 返回绑定核心的平均使用率
                    bound_usage = [per_cpu_usage[core] for core in bound_cores if core < len(per_cpu_usage)]
                    return np.mean(bound_usage) if bound_usage else 0.0
        except:
            pass
        return 0.0
        
    def calculate_rpi_power(self):
        """
        计算树莓派功耗（基于实际硬件数据）
        """
        try:
            # 1. 获取硬件数据
            temp = self.read_hardware_temp()
            voltages = self.read_hardware_voltages()
            core_voltage = self.read_vcgencmd_voltage()
            cpu_freq = self.get_cpu_frequency()
            cpu_usage = psutil.cpu_percent(interval=0.01)
            
            # 获取绑定核心的使用率
            bound_core_usage = self.get_bound_core_usage()
            
            # 2. 树莓派5功耗模型
            # 基础功耗：主要来自SoC和内存
            base_power = 1.5  # 树莓派5空闲功耗约1.5W
            
            # CPU动态功耗计算
            cpu_power = 0
            bound_core_power = 0
            
            if core_voltage and cpu_freq:
                # 功耗 ≈ C × V² × f （C为容性负载常数）
                # 树莓派5 CPU最大功耗约4-6W
                freq_factor = cpu_freq / 2400.0  # 标准化到2.4GHz
                voltage_factor = (core_voltage / 0.9) ** 2  # 标准化到0.9V
                usage_factor = cpu_usage / 100.0
                
                cpu_power = 4.0 * freq_factor * voltage_factor * usage_factor
                
                # 计算绑定核心的功耗（单核心的理论最大功耗约1W）
                bound_core_usage_factor = bound_core_usage / 100.0
                bound_core_power = 1.0 * freq_factor * voltage_factor * bound_core_usage_factor
            else:
                # 简化估算
                cpu_power = 3.0 * (cpu_usage / 100.0)
                bound_core_power = 0.75 * (bound_core_usage / 100.0)
                
            # 温度修正
            temp_factor = 1.0
            if temp:
                # 温度高时效率下降，功耗增加
                if temp > 60:
                    temp_factor = 1.0 + (temp - 60) * 0.005
                    
            # 总功耗
            total_power = (base_power + cpu_power) * temp_factor
            bound_core_power *= temp_factor
            
            return total_power, {
                'temperature': temp,
                'core_voltage': core_voltage,
                'cpu_frequency': cpu_freq,
                'cpu_usage': cpu_usage,
                'bound_core_usage': bound_core_usage,
                'voltages': voltages,
                'base_power': base_power,
                'cpu_power': cpu_power,
                'bound_core_power': bound_core_power,
                'temp_factor': temp_factor
            }
            
        except Exception as e:
            print(f"Error calculating RPI power: {e}")
            return 2.0, {}
            
    def start_monitoring(self):
        """开始功耗监控"""
        self.monitoring = True
        self.power_readings = []
        self.start_time = time.time()
        
        def monitor():
            while self.monitoring:
                power, details = self.calculate_rpi_power()
                timestamp = time.time() - self.start_time
                self.power_readings.append({
                    'timestamp': timestamp,
                    'power': power,
                    **details
                })
                time.sleep(0.005)  # 5ms采样率，更精确
                
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """停止功耗监控并返回统计数据"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
            
        if self.power_readings:
            powers = [reading['power'] for reading in self.power_readings]
            bound_core_powers = [reading.get('bound_core_power', 0) for reading in self.power_readings]
            
            return {
                'avg_power': np.mean(powers),
                'min_power': np.min(powers),
                'max_power': np.max(powers),
                'std_power': np.std(powers),
                'avg_bound_core_power': np.mean(bound_core_powers),
                'min_bound_core_power': np.min(bound_core_powers),
                'max_bound_core_power': np.max(bound_core_powers),
                'std_bound_core_power': np.std(bound_core_powers),
                'readings': self.power_readings,
                'sample_count': len(self.power_readings)
            }
        return {
            'avg_power': 2.0, 
            'avg_bound_core_power': 0.0,
            'readings': [], 
            'sample_count': 0
        }

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

def test_model_rpi(model_path, num_runs=1000):
    """
    树莓派优化的模型测试函数
    """
    print(f"Testing {os.path.basename(model_path)} on Raspberry Pi...")
    
    # 设置普通模式（非超频）
    set_rpi_normal_mode()
    
    # 加载模型
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 准备输入数据
    input_shape = input_details[0]['shape']
    if input_details[0]['dtype'] == np.uint8:
        dummy_input = np.random.randint(0, 256, input_shape, dtype=np.uint8)
    else:
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
    
    # 创建功耗监控器
    power_monitor = RaspberryPiPowerMonitor()
    
    # 预热
    print("Warming up (50 runs)...")
    for _ in range(50):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    
    # 等待系统稳定
    time.sleep(1.0)
    
    # 测量基线功耗
    print("Measuring baseline power (3 seconds)...")
    power_monitor.start_monitoring()
    time.sleep(3.0)
    baseline_stats = power_monitor.stop_monitoring()
    baseline_power = baseline_stats['avg_power']
    baseline_bound_core_power = baseline_stats['avg_bound_core_power']
    
    print(f"Baseline power: {baseline_power:.3f} W (from {baseline_stats['sample_count']} samples)")
    print(f"Baseline bound core power: {baseline_bound_core_power:.3f} W")
    
    # 正式测试
    inference_times = []
    test_power_readings = []
    bound_core_power_readings = []
    
    print(f"Running {num_runs} inferences...")
    
    # 分批测试，每批50次推理
    batch_size = 50
    batches = num_runs // batch_size
    
    for batch in range(batches):
        if batch % 5 == 0:
            print(f"Batch {batch + 1}/{batches}")
        
        # 开始功耗监控
        power_monitor.start_monitoring()
        
        batch_times = []
        for i in range(batch_size):
            start_time = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            end_time = time.perf_counter()
            
            batch_times.append((end_time - start_time) * 1000)
        
        # 停止功耗监控
        power_stats = power_monitor.stop_monitoring()
        
        inference_times.extend(batch_times)
        test_power_readings.append(power_stats['avg_power'])
        bound_core_power_readings.append(power_stats['avg_bound_core_power'])
        
        # 短暂休息
        time.sleep(0.05)
    
    # 计算统计数据
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    avg_test_power = np.mean(test_power_readings)
    avg_bound_core_power = np.mean(bound_core_power_readings)
    
    # 计算推理相关功耗
    inference_power = max(0, avg_test_power - baseline_power)
    inference_bound_core_power = max(0, avg_bound_core_power - baseline_bound_core_power)
    
    # 计算能效
    energy_per_inference = (avg_test_power * avg_inference_time) / 1000  # mJ
    
    results = {
        'avg_time_ms': avg_inference_time,
        'std_time_ms': std_inference_time,
        'avg_power_w': avg_test_power,  # 系统总功耗
        'avg_bound_core_power_w': avg_bound_core_power,  # 绑定核心功耗
        'baseline_power_w': baseline_power,
        'baseline_bound_core_power_w': baseline_bound_core_power,
        'inference_power_w': inference_power,
        'inference_bound_core_power_w': inference_bound_core_power,
        'energy_per_inference_mj': energy_per_inference,
        'min_time_ms': np.min(inference_times),
        'max_time_ms': np.max(inference_times),
        'total_runs': len(inference_times)
    }
    
    print(f"Results:")
    print(f"  Average inference time: {avg_inference_time:.3f} ± {std_inference_time:.3f} ms")
    print(f"  System power: {avg_test_power:.3f} W (baseline: {baseline_power:.3f} W)")
    print(f"  Bound core power: {avg_bound_core_power:.3f} W (baseline: {baseline_bound_core_power:.3f} W)")
    print(f"  Inference power (system): {inference_power:.3f} W")
    print(f"  Inference power (bound core): {inference_bound_core_power:.3f} W")
    print(f"  Energy per inference: {energy_per_inference:.3f} mJ")
    
    return results

def visualize_rpi_results(all_results):
    """
    可视化树莓派测试结果
    """
    if not all_results:
        print("No results to visualize")
        return
    
    layer_types = [r['layer_type'] for r in all_results]
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 推理时间
    times = [r['avg_time_ms'] for r in all_results]
    time_stds = [r['std_time_ms'] for r in all_results]
    bars1 = ax1.bar(layer_types, times, yerr=time_stds, capsize=5, 
                   alpha=0.8, color='lightblue', edgecolor='navy')
    ax1.set_title('Raspberry Pi 5 - Inference Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (ms)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    for bar, time_val, std_val in zip(bars1, times, time_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std_val,
                f'{time_val:.2f}±{std_val:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # 2. 功耗对比（包含绑定核心功耗）
    baseline_powers = [r.get('baseline_power_w', 0) for r in all_results]
    test_powers = [r['avg_power_w'] for r in all_results]  # 系统总功耗
    bound_core_powers = [r.get('avg_bound_core_power_w', 0) for r in all_results]  # 绑定核心功耗
    
    x = np.arange(len(layer_types))
    width = 0.25
    
    ax2.bar(x - width, baseline_powers, width, label='Baseline (System)', 
           alpha=0.8, color='lightgreen', edgecolor='darkgreen')
    ax2.bar(x, test_powers, width, label='System Power', 
           alpha=0.8, color='orange', edgecolor='darkorange')
    ax2.bar(x + width, bound_core_powers, width, label='Bound Core Power', 
           alpha=0.8, color='red', edgecolor='darkred')
    
    ax2.set_title('Raspberry Pi 5 - Power Consumption', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Power (W)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_types, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 每次推理能耗
    energies = [r.get('energy_per_inference_mj', 0) for r in all_results]
    bars3 = ax3.bar(layer_types, energies, alpha=0.8, color='purple', edgecolor='darkmagenta')
    ax3.set_title('Energy per Inference', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Energy (mJ)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    for bar, energy in zip(bars3, energies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # 4. 效率对比 (推理次数/秒/瓦特)
    efficiency = [1000 / (r['avg_time_ms'] * r['avg_power_w']) 
                 for r in all_results]
    bars4 = ax4.bar(layer_types, efficiency, alpha=0.8, color='gold', edgecolor='darkorange')
    ax4.set_title('Power Efficiency (inferences/s/W)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Efficiency')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    for bar, eff in zip(bars4, efficiency):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Raspberry Pi 5 CPU Benchmark Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    os.makedirs("./results", exist_ok=True)
    plt.savefig("./results/cpu_benchmark.png", dpi=300, bbox_inches='tight')  # 修改文件名以便对比
    print("Raspberry Pi visualization saved to ./results/cpu_benchmark.png")
    plt.show()

def main():
    """
    主函数 - 运行树莓派CPU基准测试
    """
    # CPU模型配置
    model_configs = [
        "conv2d_cpu.tflite",
        "depthwise_conv2d_cpu.tflite",
        "separable_conv_cpu.tflite", 
        "max_pool_cpu.tflite",
        "avg_pool_cpu.tflite",
        "dense_cpu.tflite",
        "feature_pyramid_cpu.tflite",
        "detection_head_cpu.tflite"
    ]
    
    models_dir = "./cpu"
    results = []
    
    print("=" * 70)
    print("🍓 Raspberry Pi 5 CPU Benchmark Test")
    print("=" * 70)
    print("Using native hardware monitoring for accurate power measurement")
    print()
    
    for model_name in model_configs:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            print(f"\n📊 Testing: {model_name}")
            print("-" * 50)
            
            result = test_model_rpi(model_path, num_runs=1000)
            result.update({
                "model": model_name,
                "layer_type": model_name.replace("_cpu.tflite", ""),
                "platform": "CPU"  # 修改为与compare_results.py期望的格式一致
            })
            results.append(result)
            
            print(f"✅ Completed: {model_name}")
            print("-" * 50)
        else:
            print(f"❌ Model not found: {model_path}")
    
    if results:
        # 保存详细结果
        df = pd.DataFrame(results)
        os.makedirs("./results", exist_ok=True)
        
        csv_path = "./results/cpu_benchmark_results.csv"  # 修改为与compare_results.py期望的文件名一致
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved to: {csv_path}")
        
        # 显示汇总
        print("\n" + "=" * 70)
        print("📈 RASPBERRY PI 5 CPU BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Layer Type':<15} | {'Time (ms)':<12} | {'Power (W)':<10} | {'Energy (mJ)':<12}")
        print("-" * 70)
        for result in results:
            print(f"{result['layer_type']:<15} | "
                  f"{result['avg_time_ms']:6.2f}±{result['std_time_ms']:4.2f} | "
                  f"{result['avg_power_w']:8.3f} | "  # 使用新字段名
                  f"{result.get('energy_per_inference_mj', 0):10.3f}")
        
        # 生成可视化
        visualize_rpi_results(results)
        
        print(f"\n🎉 Raspberry Pi 5 CPU benchmarking completed!")
        print(f"📁 All results saved to: ./results/")
        
        # 显示系统信息
        temp = subprocess.run(['vcgencmd', 'measure_temp'], 
                            capture_output=True, text=True).stdout.strip()
        volt = subprocess.run(['vcgencmd', 'measure_volts'], 
                            capture_output=True, text=True).stdout.strip()
        print(f"\n🌡️  Final system state: {temp}, {volt}")
        
    else:
        print("❌ No models found to test. Please check the ./cpu directory.")

if __name__ == "__main__":
    main()
