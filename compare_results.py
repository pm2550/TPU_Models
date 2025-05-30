#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_merge_results():
    """
    加载CPU和TPU的测试结果并合并
    """
    try:
        cpu_df = pd.read_csv("./results/cpu_benchmark_results.csv")
        cpu_df['platform'] = 'CPU'
        print(f"Loaded CPU results: {len(cpu_df)} entries")
    except FileNotFoundError:
        print("CPU results not found!")
        cpu_df = pd.DataFrame()
    
    try:
        tpu_df = pd.read_csv("./results/tpu_benchmark_results.csv")
        tpu_df['platform'] = 'TPU'
        print(f"Loaded TPU results: {len(tpu_df)} entries")
    except FileNotFoundError:
        print("TPU results not found!")
        tpu_df = pd.DataFrame()
    
    if cpu_df.empty and tpu_df.empty:
        print("No results found!")
        return pd.DataFrame()
    
    # 合并数据
    combined_df = pd.concat([cpu_df, tpu_df], ignore_index=True)
    return combined_df

def create_comparison_plots(df):
    """
    创建CPU vs TPU对比图表
    """
    if df.empty:
        print("No data to plot!")
        return
    
    # 获取共同的层类型
    cpu_layers = set(df[df['platform'] == 'CPU']['layer_type'])
    tpu_layers = set(df[df['platform'] == 'TPU']['layer_type'])
    common_layers = sorted(cpu_layers.intersection(tpu_layers))
    
    if not common_layers:
        print("No common layers found for comparison!")
        print(f"CPU layers: {cpu_layers}")
        print(f"TPU layers: {tpu_layers}")
        return
    
    # 准备对比数据
    cpu_times = []
    tpu_times = []
    cpu_system_powers = []  # CPU系统功耗
    cpu_core_powers = []    # CPU绑定核心功耗
    tpu_powers = []         # TPU功耗
    
    for layer in common_layers:
        cpu_data = df[(df['platform'] == 'CPU') & (df['layer_type'] == layer)]
        tpu_data = df[(df['platform'] == 'TPU') & (df['layer_type'] == layer)]
        
        if not cpu_data.empty and not tpu_data.empty:
            cpu_times.append(cpu_data['avg_time_ms'].iloc[0])
            tpu_times.append(tpu_data['avg_time_ms'].iloc[0])
            cpu_system_powers.append(cpu_data['avg_power_w'].iloc[0])
            cpu_core_powers.append(cpu_data.get('avg_bound_core_power_w', pd.Series([0])).iloc[0])
            tpu_powers.append(tpu_data['avg_power_w'].iloc[0])
    
    # 创建对比图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    x = np.arange(len(common_layers))
    width = 0.35
    
    # 推理时间对比
    bars1 = ax1.bar(x - width/2, cpu_times, width, label='CPU', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, tpu_times, width, label='TPU', alpha=0.8, color='orange')
    ax1.set_xlabel("Layer Type")
    ax1.set_ylabel("Inference Time (ms)")
    ax1.set_title("Inference Time Comparison: CPU vs TPU")
    ax1.set_xticks(x)
    ax1.set_xticklabels(common_layers, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    # 功耗对比 - 三种功耗柱状图
    x = np.arange(len(common_layers))
    width = 0.25
    
    bars3 = ax2.bar(x - width, cpu_system_powers, width, label='CPU System Power', 
                   alpha=0.8, color='lightcoral', edgecolor='darkred')
    bars4 = ax2.bar(x, cpu_core_powers, width, label='CPU Core Power', 
                   alpha=0.8, color='orange', edgecolor='darkorange')
    bars5 = ax2.bar(x + width, tpu_powers, width, label='TPU Power', 
                   alpha=0.8, color='green', edgecolor='darkgreen')
    ax2.set_xlabel("Layer Type")
    ax2.set_ylabel("Power Consumption (W)")
    ax2.set_title("Power Consumption Comparison: CPU vs TPU")
    ax2.set_xticks(x)
    ax2.set_xticklabels(common_layers, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, power in zip(bars3, cpu_system_powers):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{power:.2f}', ha='center', va='bottom', fontsize=8)
    for bar, power in zip(bars4, cpu_core_powers):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{power:.2f}', ha='center', va='bottom', fontsize=8)
    for bar, power in zip(bars5, tpu_powers):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{power:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 加速比
    speedup = [cpu/tpu if tpu > 0 else 0 for cpu, tpu in zip(cpu_times, tpu_times)]
    bars6 = ax3.bar(common_layers, speedup, alpha=0.8, color='purple')
    ax3.set_xlabel("Layer Type")
    ax3.set_ylabel("Speedup (CPU Time / TPU Time)")
    ax3.set_title("TPU Speedup over CPU")
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    ax3.legend()
    
    # 添加数值标签
    for bar in bars6:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom')
    
    # 能效比 (时间/功耗，越小越好)
    cpu_efficiency = [time/power if power > 0 else 0 for time, power in zip(cpu_times, cpu_system_powers)]
    tpu_efficiency = [time/power if power > 0 else 0 for time, power in zip(tpu_times, tpu_powers)]
    
    bars7 = ax4.bar(x - width/2, cpu_efficiency, width, label='CPU', alpha=0.8, color='lightblue')
    bars8 = ax4.bar(x + width/2, tpu_efficiency, width, label='TPU', alpha=0.8, color='lightgreen')
    ax4.set_xlabel("Layer Type")
    ax4.set_ylabel("ms/W (lower is better)")
    ax4.set_title("Energy Efficiency Comparison")
    ax4.set_xticks(x)
    ax4.set_xticklabels(common_layers, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars7:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    for bar in bars8:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    os.makedirs("./results", exist_ok=True)
    plt.savefig('./results/cpu_vs_tpu_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计摘要
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    for i, layer in enumerate(common_layers):
        print(f"\n{layer}:")
        print(f"  CPU: {cpu_times[i]:8.2f} ms")
        print(f"    System Power: {cpu_system_powers[i]:6.2f} W")
        print(f"    Core Power:   {cpu_core_powers[i]:6.2f} W") 
        print(f"  TPU: {tpu_times[i]:8.2f} ms, {tpu_powers[i]:6.2f} W")
        print(f"  Speedup: {speedup[i]:6.2f}x")
        if cpu_system_powers[i] > 0:
            power_ratio = tpu_powers[i]/cpu_system_powers[i]
            print(f"  Power ratio (TPU/CPU System): {power_ratio:.2f}x")
        if cpu_core_powers[i] > 0:
            core_power_ratio = tpu_powers[i]/cpu_core_powers[i]
            print(f"  Power ratio (TPU/CPU Core):   {core_power_ratio:.2f}x")
        efficiency_ratio = tpu_efficiency[i]/cpu_efficiency[i] if cpu_efficiency[i] > 0 else 0
        print(f"  Energy efficiency ratio (TPU/CPU): {efficiency_ratio:.2f}x")

def main():
    print("Loading benchmark results...")
    df = load_and_merge_results()
    
    if not df.empty:
        print(f"Found {len(df)} total results")
        print("Creating comparison plots...")
        create_comparison_plots(df)
        print("Comparison completed! Check './results/cpu_vs_tpu_comparison.png'")
    else:
        print("No data available for comparison")

if __name__ == "__main__":
    main()
