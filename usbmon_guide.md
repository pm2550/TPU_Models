Directory Layout
- `models_local/` — local model assets (e.g., `models_local/public/<model>/full_split_pipeline_local/tpu/segN_*.tflite`).
- `five_models/` — baselines and consolidated outputs.
  - `five_models/baselines/*.json` — IO sizes and weight splits for segments/combos.
  - `five_models/results/*` — CSVs and plots produced by tools.
- `results/models_local_batch_usbmon/` — single‑segment batch runs.
- `results/models_local_combo_chain/` — real chain (combo) runs.
- `results/models_local_combo_sim/` — simulated chain (combo) runs.

Captured Artifacts
- `usbmon.txt` — raw usbmon dump captured from `/sys/kernel/debug/usb/usbmon/<bus>u`.
- `time_map.json` — mapping between CLOCK_BOOTTIME and usbmon times; contains `boottime_ref`, `usbmon_ref`, optionally `epoch_ref`.
- `invokes.json` — per‑invoke windows `{begin,end[,set_begin,get_end]}` in boottime seconds.
- `active_analysis*.json` — analyzer outputs with per‑invoke IO union and pure compute metrics.
- `io_split_bt.json` — fallback IO statistics from the quick/overlap path.
- `merged_invokes.json` — cross‑segment merged windows (combo workflows).

Core Workflows
- Single‑segment offline capture (`run_usbmon_capture_offline.sh`)
  - Inputs: model path, bus number, duration.
  - Outputs (under the chosen directory): `usbmon.txt`, `time_map.json`, `invokes.json`, `active_analysis_strict.json`.
  - Notes: also runs `tools/correct_per_invoke_stats.py` (fallback stats) and `analyze_usbmon_active.py` (authoritative analysis).
- Chain capture — real (`run_usbmon_chain_offline.sh`)
  - Inputs: TPU directory with seg models, bus, duration.
  - Outputs: `usbmon.txt`, `time_map.json`, per‑segment `seg*/invokes.json`, `seg*/io_split_bt.json`.
- Chain capture — simulated (`run_usbmon_chain_offline_sim.sh`)
  - Same structure as real chain; uses randomized inputs per segment instead of true chaining.
- Batch runner (`run_models_local_batch_usbmon.py`)
  - Discovers models under `models_local`, runs capture, ensures alignment, and writes per‑segment `performance_summary.json` plus per‑model summaries.
- Combo runner (`run_models_local_combo_sim.py`)
  - Produces `merged_invokes.json`, runs global per‑invoke corrections, and saves per‑segment summaries (supports both sim and real chain modes).

Alignment and Analyzer
- Offline alignment (`tools/offline_align_usbmon_ref.py`)
  - Inputs: `usbmon.txt`, `invokes.json`, `time_map.json`.
  - Effect: sets/repairs `usbmon_ref` in `time_map.json` by anchoring to the earliest large OUT.
- Authoritative analyzer (`analyze_usbmon_active.py`)
  - Inputs: `usbmon.txt`, `invokes.json`, `time_map.json`.
  - Defaults (env‑tunable):
    - `STRICT_INVOKE_WINDOW=1`
    - `SHIFT_POLICY=tail_last_BiC_guard_BoS`
    - `SEARCH_TAIL_MS=40`, `SEARCH_HEAD_MS=40`, `EXTRA_HEAD_EXPAND_MS=10`, `MAX_SHIFT_MS=50`
    - `SPAN_STRICT_PAIR=1`, `MIN_URB_BYTES=512`, `CLUSTER_GAP_MS=0.1`
  - Output: JSON to stdout; callers save as `active_analysis*.json`.

Theory and Visualization
- `tools/compute_theory_chain_times.py`
  - Reads: `five_models/baselines/theory_io_combos.json`, `five_models/baselines/theory_io_seg.json`, single‑mode summaries under `results/models_local_batch_usbmon/single/`, and `five_models/results/single_pure_invoke_times.csv` (read/write with backups).
  - Writes: `five_models/results/theory_chain_times.csv`, updates `five_models/results/theory_chain_source_data.csv`, and patches selected single‑mode summaries.
- `tools/run_model_theory_sweeps.py`
  - Reads theory times and measured combo cycles; writes `five_models/results/theory_sweeps/<model>.csv` and `five_models/results/theory_sweeps/all_models.csv`.
- `tools/plot_combo_speeds.py`
  - Inputs: `five_models/results/combo_cycle_times.csv`, `five_models/results/theory_chain_times.csv`.
  - Output: `five_models/results/plots/<short>_time_vs_K.png`.
- `tools/plot_sweep_boxplots.py`
  - Inputs: `five_models/results/theory_sweeps/all_models.csv`, `five_models/results/combo_cycle_times.csv`.
  - Outputs: `five_models/results/theory_sweeps/plots/*_box_vs_K.png|.pdf`, plus coverage metrics CSVs.
- `tools/generate_combo_csvs.py`
  - Aggregates per‑segment summaries into combo cycle and per‑segment metrics under `five_models/results/`.

Host‑Time Analysis
- `analyze_host_time.py`
  - Single mode: reads `results/models_local_batch_usbmon/single/<model>/seg1/{usbmon.txt,invokes.json,time_map.json}`.
  - K2 chain mode: reads `results/models_local_combo_chain/<model>/K2/{usbmon.txt,time_map.json}` and `K2/seg1/invokes.json`.
  - Writes: CSVs per model and `five_models/results/host_time/summary.json`.

Off‑Chip Pure Time Update
- `tools/update_pure_from_offchip_span.py`
  - Reads: `results/models_local_batch_usbmon/single/combined_summary_span.json`, `results/models_local_batch_usbmon/single/combined_pure_gap_seg1-8_summary.csv`, `five_models/results/single_pure_invoke_times.csv`, `five_models/baselines/theory_io_seg.json`.
  - Writes: in‑place update to `five_models/results/single_pure_invoke_times.csv` with `*.measured.bak` backup.

Notes and Defaults
- Bus detection: `list_usb_buses.py` finds EdgeTPU USB bus IDs for capture scripts.
- Sensitive files: `password.text` is required by capture scripts; keep it out of version control and archives.
- Environment knobs: runners accept `COUNT`, `INVOKE_GAP_MS` (ms), `CAP_DUR` (s), and analyzer tuning vars listed above.

Quick Reference (Script → Inputs → Outputs)
- `run_usbmon_capture_offline.sh` → model path, bus, duration → `usbmon.txt`, `time_map.json`, `invokes.json`, `active_analysis_strict.json`.
- `run_usbmon_chain_offline.sh` → TPU dir, bus, duration → `usbmon.txt`, `time_map.json`, `seg*/invokes.json`, `seg*/io_split_bt.json`.
- `run_usbmon_chain_offline_sim.sh` → TPU dir, bus, duration → same outputs with simulated inputs.
- `run_models_local_batch_usbmon.py` → model roots → per‑segment summaries + per‑model summaries.
- `run_models_local_combo_sim.py` → model roots → `merged_invokes.json`, per‑segment summaries.
- `analyze_usbmon_active.py` → `usbmon.txt`, `invokes.json`, `time_map.json` → JSON to stdout.
- `tools/offline_align_usbmon_ref.py` → `usbmon.txt`, `invokes.json`, `time_map.json` → updates `time_map.json`.
- `tools/compute_theory_chain_times.py` → baselines + single summaries → `five_models/results/theory_chain_times.csv`.
- `tools/run_model_theory_sweeps.py` → theory times + measured combos → `five_models/results/theory_sweeps/*`.
- `tools/plot_combo_speeds.py` → combo cycles + theory times → plots under `five_models/results/plots/`.
- `tools/plot_sweep_boxplots.py` → sweeps + measured combos → plots + metrics under `five_models/results/theory_sweeps/plots/`.
- `tools/update_pure_from_offchip_span.py` → span summaries + CSV → updated CSV.
