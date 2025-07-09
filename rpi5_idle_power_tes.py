#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树莓派5静息功耗测试脚本
用于测量系统在静息状态下的功耗情况
"""

import time
import psutil
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

class RPi5PowerMonitor:
    def __init__(self):
        self.data = []
        self.start_time = time.time()
        
    def get_vcgencmd_data(self):
        """获取树莓派特有的系统信息"""
        try:
            # 温度
            temp_result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                       capture_output=True, text=True)
            temp = float(temp_result.stdout.strip().replace('temp=', '').replace("'C", ''))
            
            # 电压
            volt_result = subprocess.run(['vcgencmd', 'measure_volts'], 
                                       capture_output=True, text=True)
            voltage = float(volt_result.stdout.strip().replace('volt=', '').replace('V', ''))
            
            # CPU频率
            freq_result = subprocess.run(['vcgencmd', 'measure_clock', 'arm'], 
                                       capture_output=True, text=True)
            freq_hz = int(freq_result.stdout.strip().replace('frequency(48)=', ''))
            freq_mhz = freq_hz / 1000000
            
            # GPU频率
            gpu_freq_result = subprocess.run(['vcgencmd', 'measure_clock', 'core'], 
                                           capture_output=True, text=True)
            gpu_freq_hz = int(gpu_freq_result.stdout.strip().replace('frequency(1)=', ''))
            gpu_freq_mhz = gpu_freq_hz / 1000000
            
            return temp, voltage, freq_mhz, gpu_freq_mhz
            
        except Exception as e:
            print(f"Error getting vcgencmd data: {e}")
            return None, None, None, None
    
    def get_system_stats(self):
        """获取系统统计信息"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        
        # 内存使用
        memory = psutil.virtual_memory()
        
        # 磁盘IO
        disk_io = psutil.disk_io_counters()
        
        # 网络IO
        network_io = psutil.net_io_counters()
        
        # 进程数
        process_count = len(psutil.pids())
        
        return {
            'cpu_percent': cpu_percent,
            'cpu_freq_current': cpu_freq.current if cpu_freq else 0,
            'memory_percent': memory.percent,
            'memory_used_mb': memory.used / 1024 / 1024,
            'disk_read_mb': disk_io.read_bytes / 1024 / 1024,
            'disk_write_mb': disk_io.write_bytes / 1024 / 1024,
            'network_sent_mb': network_io.bytes_sent / 1024 / 1024,
            'network_recv_mb': network_io.bytes_recv / 1024 / 1024,
            'process_count': process_count
        }
    
    def collect_sample(self):
        """收集一个样本"""
        timestamp = time.time()
        relative_time = timestamp - self.start_time
        
        # 获取vcgencmd数据
        temp, voltage, cpu_freq_vc, gpu_freq = self.get_vcgencmd_data()
        
        # 获取系统统计
        sys_stats = self.get_system_stats()
        
        sample = {
            'timestamp': timestamp,
            'relative_time': relative_time,
            'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'temperature_c': temp,
            'voltage_v': voltage,
            'cpu_freq_vcgencmd_mhz': cpu_freq_vc,
            'gpu_freq_mhz': gpu_freq,
            **sys_stats
        }
        
        self.data.append(sample)
        return sample
    
    def monitor_idle_power(self, duration_minutes=10, interval_seconds=5):
        """监测静息功耗"""
        total_samples = int(duration_minutes * 60 / interval_seconds)
        
        print("="*60)
        print("🔋 树莓派5静息功耗测试")
        print("="*60)
        print(f"测试时长: {duration_minutes} 分钟")
        print(f"采样间隔: {interval_seconds} 秒")
        print(f"预计样本数: {total_samples}")
        print("="*60)
        print("⚠️  请确保在测试期间:")
        print("   1. 不要运行其他程序")
        print("   2. 不要连接鼠标键盘(如果可能)")
        print("   3. 关闭不必要的服务")
        print("   4. 记录功率计显示的功耗值")
        print("="*60)
        
        input("准备好后按回车开始测试...")
        
        print(f"\n开始监测静息功耗 - {datetime.now().strftime('%H:%M:%S')}")
        print("Time\t\tTemp\tVolt\tCPU%\tCPU_MHz\tMem%\tProc")
        print("-" * 60)
        
        for i in range(total_samples):
            sample = self.collect_sample()
            
            # 显示当前状态
            if sample['temperature_c'] and sample['voltage_v']:
                print(f"{sample['relative_time']:6.0f}s\t"
                      f"{sample['temperature_c']:.1f}°C\t"
                      f"{sample['voltage_v']:.3f}V\t"
                      f"{sample['cpu_percent']:.1f}%\t"
                      f"{sample['cpu_freq_vcgencmd_mhz']:.0f}\t"
                      f"{sample['memory_percent']:.1f}%\t"
                      f"{sample['process_count']}")
            
            # 进度显示
            if (i + 1) % 12 == 0:  # 每分钟显示一次进度
                elapsed_min = (i + 1) * interval_seconds / 60
                print(f"--- {elapsed_min:.1f}/{duration_minutes} 分钟完成 ---")
            
            time.sleep(interval_seconds)
        
        print(f"\n✅ 监测完成 - {datetime.now().strftime('%H:%M:%S')}")
        
        # 手动输入功耗数据
        print("\n" + "="*60)
        print("📊 请输入功率计读数")
        print("="*60)
        
        power_readings = {}
        try:
            power_readings['idle_power_w'] = float(input("静息状态平均功耗 (W): "))
            power_readings['peak_power_w'] = float(input("测试期间最高功耗 (W): "))
            power_readings['min_power_w'] = float(input("测试期间最低功耗 (W): "))
        except ValueError:
            print("⚠️  功耗数据输入错误，将跳过功耗分析")
            power_readings = {}
        
        return power_readings
    
    def analyze_and_save(self, power_readings=None):
        """分析数据并保存结果"""
        if not self.data:
            print("❌ 没有数据可分析")
            return
        
        df = pd.DataFrame(self.data)
        
        # 创建结果目录
        os.makedirs("./results/rpi5_idle_test", exist_ok=True)
        
        # 保存原始数据
        df.to_csv("./results/rpi5_idle_test/idle_power_raw_data.csv", index=False)
        
        # 计算统计信息
        print("\n" + "="*60)
        print("📈 静息状态分析结果")
        print("="*60)
        
        stats = {
            'duration_minutes': df['relative_time'].max() / 60,
            'avg_temperature_c': df['temperature_c'].mean(),
            'avg_voltage_v': df['voltage_v'].mean(),
            'avg_cpu_percent': df['cpu_percent'].mean(),
            'avg_cpu_freq_mhz': df['cpu_freq_vcgencmd_mhz'].mean(),
            'avg_memory_percent': df['memory_percent'].mean(),
            'avg_process_count': df['process_count'].mean(),
            'temp_range_c': df['temperature_c'].max() - df['temperature_c'].min(),
            'cpu_usage_std': df['cpu_percent'].std()
        }
        
        print(f"测试时长: {stats['duration_minutes']:.1f} 分钟")
        print(f"平均温度: {stats['avg_temperature_c']:.1f}°C")
        print(f"平均电压: {stats['avg_voltage_v']:.3f}V")
        print(f"平均CPU使用率: {stats['avg_cpu_percent']:.2f}%")
        print(f"平均CPU频率: {stats['avg_cpu_freq_mhz']:.0f} MHz")
        print(f"平均内存使用率: {stats['avg_memory_percent']:.1f}%")
        print(f"平均进程数: {stats['avg_process_count']:.0f}")
        print(f"温度波动范围: {stats['temp_range_c']:.1f}°C")
        print(f"CPU使用率标准差: {stats['cpu_usage_std']:.2f}%")
        
        # 如果有功耗数据，添加到统计中
        if power_readings:
            stats.update(power_readings)
            print(f"\n💡 功耗分析:")
            print(f"静息平均功耗: {power_readings.get('idle_power_w', 'N/A')} W")
            print(f"最高功耗: {power_readings.get('peak_power_w', 'N/A')} W")
            print(f"最低功耗: {power_readings.get('min_power_w', 'N/A')} W")
            
            if 'idle_power_w' in power_readings:
                power_efficiency = power_readings['idle_power_w'] / 4  # 4核心
                print(f"每核心平均功耗: {power_efficiency:.3f} W/core")
        
        # 保存统计信息
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv("./results/rpi5_idle_test/idle_power_summary.csv", index=False)
        
        # 生成图表
        self.generate_plots(df)
        
        print(f"\n📁 所有结果已保存到: ./results/rpi5_idle_test/")
        
        return stats
    
    def generate_plots(self, df):
        """生成分析图表"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 温度变化
        axes[0, 0].plot(df['relative_time'] / 60, df['temperature_c'], 'r-', linewidth=1)
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('Temperature (°C)')
        axes[0, 0].set_title('Temperature Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. CPU使用率
        axes[0, 1].plot(df['relative_time'] / 60, df['cpu_percent'], 'b-', linewidth=1)
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('CPU Usage (%)')
        axes[0, 1].set_title('CPU Usage Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. CPU频率
        axes[0, 2].plot(df['relative_time'] / 60, df['cpu_freq_vcgencmd_mhz'], 'g-', linewidth=1)
        axes[0, 2].set_xlabel('Time (minutes)')
        axes[0, 2].set_ylabel('CPU Frequency (MHz)')
        axes[0, 2].set_title('CPU Frequency Over Time')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 电压
        axes[1, 0].plot(df['relative_time'] / 60, df['voltage_v'], 'orange', linewidth=1)
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('Voltage (V)')
        axes[1, 0].set_title('Voltage Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 内存使用率
        axes[1, 1].plot(df['relative_time'] / 60, df['memory_percent'], 'purple', linewidth=1)
        axes[1, 1].set_xlabel('Time (minutes)')
        axes[1, 1].set_ylabel('Memory Usage (%)')
        axes[1, 1].set_title('Memory Usage Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 进程数
        axes[1, 2].plot(df['relative_time'] / 60, df['process_count'], 'brown', linewidth=1)
        axes[1, 2].set_xlabel('Time (minutes)')
        axes[1, 2].set_ylabel('Process Count')
        axes[1, 2].set_title('Process Count Over Time')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("./results/rpi5_idle_test/idle_power_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成温度和CPU使用率的分布直方图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(df['temperature_c'], bins=30, alpha=0.7, color='red', edgecolor='black')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Temperature Distribution')
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(df['cpu_percent'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2.set_xlabel('CPU Usage (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('CPU Usage Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("./results/rpi5_idle_test/idle_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 图表已生成:")
        print("   - idle_power_analysis.png (时间序列分析)")
        print("   - idle_distributions.png (分布统计)")

def main():
    monitor = RPi5PowerMonitor()
    
    # 进行静息功耗测试 (默认10分钟)
    power_readings = monitor.monitor_idle_power(duration_minutes=10, interval_seconds=5)
    
    # 分析并保存结果
    stats = monitor.analyze_and_save(power_readings)
    
    print("\n" + "="*60)
    print("🎉 树莓派5静息功耗测试完成!")
    print("="*60)
    print("💡 建议:")
    print("   1. 对比不同负载状态下的功耗")
    print("   2. 测试TPU工作时的功耗变化")
    print("   3. 优化系统服务以降低静息功耗")

if __name__ == "__main__":
    main()
