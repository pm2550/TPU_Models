#!/usr/bin/env python3
import os, json, subprocess, shlex
from pathlib import Path

BASE = Path('/home/10210/Desktop/OS')
TOOLS = BASE/'tools'
RES_DIR = BASE/'five_models/results'
PLOTS_DIR = RES_DIR/'plots_newBound'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Prefer project virtualenv python
VENV_PY = BASE/'.venv/bin/python'
PY = str(VENV_PY) if VENV_PY.exists() else 'python3'

# Define requested bounds per model: (UB, B_IN)
REQUESTS = {
    'densenet201_8seg_uniform_local': [(310.0, 355.0), (325.0, 355.0)],
    'inceptionv3_8seg_uniform_local': [(323.5, 368.5), (338.5, 368.5)],
    'resnet50_8seg_uniform_local': [(324.5, 369.5), (339.5, 369.5)],
    'resnet101_8seg_uniform_local': [(329.0, 374.0), (344.0, 374.0)],
    'xception_8seg_uniform_local': [(272.0, 317.0), (287.0, 317.0)],
}

def run_cmd(cmd: str, env: dict):
    print('>>', cmd)
    subprocess.run(cmd, shell=True, check=True, env=env)

def main():
    report_paths = []
    for model, ranges in REQUESTS.items():
        for ub, binv in ranges:
            env = os.environ.copy()
            env['B_IN'] = str(binv)
            env['UB'] = str(ub)
            env['FILTER_MODEL'] = model
            # compute theory CSV with these bounds
            run_cmd(f"{PY} {TOOLS/'compute_theory_chain_times.py'}", env)
            # plot and build coverage report JSON for this pair
            run_cmd(f"{PY} {TOOLS/'plot_combo_speeds_newBound.py'}", env)
            # After plotting, extract only this model section to a model-specific JSON
            src = RES_DIR/'plots_newBound'/'combo_bounds_report.json'
            dst = RES_DIR/'plots_newBound'/f"{model.split('_')[0]}_bounds_{ub:.1f}-{binv:.1f}.json"
            if src.exists():
                try:
                    rep = json.loads(src.read_text())
                    one = {model: rep.get(model)} if model in rep else {}
                    # include the explicit range for traceability
                    one['_range'] = {'UB': ub, 'B_IN': binv}
                    dst.write_text(json.dumps(one, ensure_ascii=False, indent=2))
                except Exception:
                    # fallback: copy raw
                    dst.write_text(src.read_text())
                report_paths.append(dst)
                print('Saved:', dst)
            # Do not delete other figures; allow all requested images to accumulate
    print('All reports:')
    for p in report_paths:
        print(p)

if __name__ == '__main__':
    main()
