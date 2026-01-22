models_local 模型所在位置
five_models 模型参数信息和结果文件位置

模型usbmon测算结果放置位置
results/models_local_batch_usbmon
results/models_local_combo_chain
results/models_local_combo_sim

analyze_usbmon_active.py 总分析文件

tools/update_pure_from_offchip_span.py 根据平均速度调整off-chip pure invoke 大小
tools/offline_align_usbmon_ref.py 离线对齐脚本

analyze_host_time.py 提取host 处理时间长度
run_models_local_combo_sim.py 交错跑模型combos
run_models_local_batch_usbmon.py 连续运行单个模型
tools/run_full_models_alternate.sh 交错跑模型临时脚本


tools/compute_theory_chain_times.py 计算统一速度的theory time脚本
tools/run_model_theory_sweeps.py 计算每一个模型不同速度的theroy time脚本

tools/plot_combo_speeds.py 根据tools/compute_theory_chain_times.py产生的结果five_models/results/theory_chain_times.csv画theory invoke time和实际分布图
tools/plot_sweep_boxplots.py 根据tools/run_model_theory_sweeps.py产生的结果five_models/results/theory_sweeps 画theory invoke time和实际分布图


附录：流程/脚本涉及的文件清单（输入/输出/依赖）

总体目录
- models_local：本地模型目录（例如 `models_local/public/<model>/full_split_pipeline_local/tpu/segN_*.tflite`）
- five_models：基线参数与汇总结果（例如 `five_models/baselines/*.json`，`five_models/results/*`）
- results/models_local_batch_usbmon：单段批量测试结果根目录
- results/models_local_combo_chain：真实链式（combo）结果根目录
- results/models_local_combo_sim：模拟链式（combo）结果根目录

捕获/对齐/基础分析
- run_usbmon_capture_offline.sh（最原始脚本，无策略）
  - 输入：模型文件 `<model_path>.tflite`；系统 `/sys/kernel/debug/usb/usbmon/<bus>u`；密码文件 `password.text`
  - 输出（在 `<out_dir>`）：`usbmon.txt`，`time_map.json`，`invokes.json`，`io_split_bt.json`（回退口径），`active_analysis_strict.json`
  - 依赖/调用：`tools/correct_per_invoke_stats.py`，`analyze_usbmon_active.py`
- run_usbmon_chain_offline.sh（多模型连续）
  - 输入：`<model_tpu_dir>`（含 seg1..seg8 或 tail_* 模型）；`/sys/kernel/debug/usb/usbmon/<bus>u`；`password.text`
  - 输出：`usbmon.txt`，`time_map.json`，以及每段 `seg*/invokes.json`、`seg*/io_split_bt.json`
  - 依赖/调用：内嵌 Python 生成各段 `invokes.json`
- run_usbmon_chain_offline_sim.sh（多模型连续+模拟输入输出数据））
  - 输入：`<tpu_dir>`；`/sys/kernel/debug/usb/usbmon/<bus>u`；`password.text`
  - 输出：`usbmon.txt`，`time_map.json`，以及 `*/invokes.json`、`*/io_split_bt.json`
  - 依赖/调用：内嵌 Python 生成各段 `invokes.json`
- tools/offline_align_usbmon_ref.py
  - 输入：`usbmon.txt`，`invokes.json`，`time_map.json`
  - 输出：回写 `time_map.json`（补齐/修正 `usbmon_ref`）
- analyze_usbmon_active.py
  - 输入：`usbmon.txt`，`invokes.json`，`time_map.json`
  - 输出：标准输出 JSON（常由上层脚本保存为 `active_analysis*.json`）

批量/组合运行
- run_models_local_batch_usbmon.py（单模型连续运行）
  - 输入：`models_local/public/<model>/full_split_pipeline_local/tpu/segN_*.tflite`；`list_usb_buses.py`；捕获脚本 `run_usbmon_capture_offline.sh`（单段）/`run_usbmon_chain_offline.sh`/`run_usbmon_chain_offline_sim.sh`；`tools/offline_align_usbmon_ref.py`
  - 输出：
    - 每段目录：`results/models_local_batch_usbmon/(single|chain|chain_sim)/<model>/segN/`
      - `usbmon.txt`，`time_map.json`，`invokes.json`，`run_env.json`，`active_analysis_strict.json`，`performance_summary.json`
      - 可能的诊断：`analysis_meta.json`，`analysis_error.txt`
    - 汇总：`results/models_local_batch_usbmon/<mode>/<model>/model_summary.json`，`results/models_local_batch_usbmon/<mode>/batch_summary.json`
- run_models_local_combo_sim.py
  - 输入：`models_local/public/<model>/.../tpu/`；`run_usbmon_chain_offline_sim.sh`/`run_usbmon_chain_offline.sh`；`tools/offline_align_usbmon_ref.py`；`tools/correct_per_invoke_stats.py`；`analyze_usbmon_active.py`
  - 输出：
    - 组合根：`usbmon.txt`，`time_map.json`，`merged_invokes.json`，`correct_per_invoke_stdout.txt`
    - 各段：`seg*/invokes.json`，`seg*/active_analysis.json`，`seg*/performance_summary.json`
- tools/run_full_models_alternate.sh
  - 输入：`models_local/public/full_models/*/*.tflite`；`password.text`；`/sys/kernel/debug/usb/usbmon/<bus>u`；`tools/prepend_epoch.py`
  - 输出：`usbmon.txt`，`time_map.json`，`merged_invokes.json`，（可选）`bo_envelope_summary.json`

理论/绘图
- tools/compute_theory_chain_times.py
  - 输入：
    - 基线：`five_models/baselines/theory_io_combos.json`，`five_models/baselines/theory_io_seg.json`
    - 纯时间：`five_models/results/single_pure_invoke_times.csv`（读/写，生成 `.bak` 备份）
    - 单段汇总：`results/models_local_batch_usbmon/single/combined_pure_gap_seg1-8_summary.csv`，`results/models_local_batch_usbmon/single/combined_summary_span.json`，`results/models_local_batch_usbmon/single/combined_summary.json`
    - 其他：`results/offchip_usbmon_avg_times.csv`，`five_models/results/theory_chain_source_data.csv`（读/写）
  - 输出：
    - `five_models/results/theory_chain_times.csv`
    - 回写：`five_models/results/theory_chain_source_data.csv`
    - 回写单段汇总：`results/models_local_batch_usbmon/single/{offchip_summary.json,onchip_summary.json,combined_summary_span.json}`
- tools/run_model_theory_sweeps.py
  - 输入：`five_models/results/theory_chain_times.csv`，`five_models/results/combo_cycle_times.csv`（由 `tools/generate_combo_csvs.py` 生成）
  - 输出：`five_models/results/theory_sweeps/<model>.csv`，`five_models/results/theory_sweeps/all_models.csv`
- tools/plot_combo_speeds.py
  - 输入：`five_models/results/combo_cycle_times.csv`，`five_models/results/theory_chain_times.csv`
  - 输出：`five_models/results/plots/<short>_time_vs_K.png`
- tools/plot_sweep_boxplots.py
  - 输入：`five_models/results/theory_sweeps/all_models.csv`，`five_models/results/combo_cycle_times.csv`
  - 输出：`five_models/results/theory_sweeps/plots/*_box_vs_K.png|.pdf`，`five_models/results/theory_sweeps/plots/metrics_per_model.csv`，`five_models/results/theory_sweeps/plots/metrics_per_model_K.csv`

主机时间分析
- analyze_host_time.py
  - 输入：
    - 单段：`results/models_local_batch_usbmon/single/<model>/seg1/{usbmon.txt,invokes.json,time_map.json}`
    - K2 链式：`results/models_local_combo_chain/<model>/K2/{usbmon.txt,time_map.json}` 与 `results/models_local_combo_chain/<model>/K2/seg1/invokes.json`
  - 输出：
    - `five_models/results/host_time/<model>/single_seg1_host_times.csv`，`five_models/results/host_time/<model>/single_seg1_host_summary.json`
    - `five_models/results/host_time/<model>_K2/k2_seg1_host_times.csv`，`five_models/results/host_time/<model>_K2/k2_seg1_host_summary.json`
    - 汇总：`five_models/results/host_time/summary.json`

更新纯时间（off‑chip 修正）
- tools/update_pure_from_offchip_span.py
  - 输入：`results/models_local_batch_usbmon/single/combined_summary_span.json`，`results/models_local_batch_usbmon/single/combined_pure_gap_seg1-8_summary.csv`，`five_models/results/single_pure_invoke_times.csv`，`five_models/baselines/theory_io_seg.json`
  - 输出：回写 `five_models/results/single_pure_invoke_times.csv`（生成 `*.measured.bak` 备份）
