#!/usr/bin/env bash
set -euo pipefail

# Run five full models alternately in a single usbmon capture, with per-invoke gap, and compute Bo envelope H2D speeds.
# Usage: tools/run_full_models_alternate.sh <out_dir> <bus> [count] [gap_ms]
# - out_dir: output directory to write usbmon.txt, time_map.json, merged_invokes.json, summaries
# - bus: usbmon bus number (e.g., 0 for 0u)
# - count: number of cycles per model (default 5)
# - gap_ms: sleep between invokes in milliseconds (default 100)

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <out_dir> <bus> [count] [gap_ms]" >&2
  exit 1
fi

ROOT="/home/10210/Desktop/OS"
OUTDIR="$1"
BUS="$2"
COUNT="${3:-5}"
GAP_MS="${4:-100}"

PW="$ROOT/password.text"
USBMON_NODE="/sys/kernel/debug/usb/usbmon/${BUS}u"
CAP="$OUTDIR/usbmon.txt"
TM="$OUTDIR/time_map.json"
MERGED="$OUTDIR/merged_invokes.json"

if [[ ! -f "$PW" ]]; then
  echo "缺少密码文件: $PW" >&2; exit 1
fi

mkdir -p "$OUTDIR"

# Resolve model paths (EdgeTPU compiled full models)
FM_BASE="$ROOT/models_local/public/full_models"
declare -A MODELS
MODELS[densenet201]="${FM_BASE}/densenet201/full_int8_edgetpu.tflite"
MODELS[inceptionv3]="${FM_BASE}/inceptionv3/full_int8_edgetpu.tflite"
MODELS[resnet101]="${FM_BASE}/resnet101/full_int8_edgetpu.tflite"
MODELS[resnet50]="${FM_BASE}/resnet50/full_int8_edgetpu.tflite"
MODELS[xception]="${FM_BASE}/xception/full_int8_edgetpu.tflite"

ORDER=(densenet201 inceptionv3 resnet101 resnet50 xception)

for key in "${ORDER[@]}"; do
  mdl="${MODELS[$key]}"
  if [[ ! -f "$mdl" ]]; then
    echo "模型不存在: $key -> $mdl" >&2
    exit 2
  fi
done

# Start usbmon capture with sudo
cat "$PW" | sudo -S -p '' modprobe usbmon || true
cat "$PW" | sudo -S -p '' sh -c ": > '$CAP'" || true
# Use prepend_epoch to add wall-clock epoch in the first column for each usbmon line
cat "$PW" | sudo -S -p '' sh -c "/usr/bin/env python3 $ROOT/tools/prepend_epoch.py < '$USBMON_NODE' > '$CAP'" &
CATPID=$!
sleep 0.1

# Build time_map.json aligned to the first usbmon line (epoch_ref + boottime_ref)
python3 - "$CAP" "$TM" <<'PY'
import json,sys,time
cap, tm_path = sys.argv[1:]
epoch_ref=None
deadline=time.time()+15.0
while time.time()<deadline and epoch_ref is None:
    try:
        with open(cap,'r',errors='ignore') as f:
            for ln in f:
                parts=ln.split()
                if not parts:
                    continue
                # prepend_epoch.py outputs: <epoch> ...
                try:
                    epoch_ref = float(parts[0])
                    break
                except Exception:
                    epoch_ref=None
    except FileNotFoundError:
        pass
    if epoch_ref is None:
        time.sleep(0.02)
try:
    bt_ref=time.clock_gettime(time.CLOCK_BOOTTIME)
except Exception:
    bt_ref=float(open('/proc/uptime').read().split()[0])
open(tm_path,'w').write(json.dumps({'epoch_ref': epoch_ref, 'boottime_ref': bt_ref}))
print('time_map_ready=', epoch_ref is not None)
PY

# Prepare and run Python to alternate models and record merged_invokes.json with seg labels
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYRUN="$ROOT/.venv/bin/python"
else
  PYRUN="$(command -v python3)"
fi

"$PYRUN" - <<PY > "$MERGED.tmp"
import os, sys, time, json, numpy as np

ROOT = r"$ROOT"
COUNT = int(r"$COUNT")
GAP_MS = float(r"$GAP_MS")

order = json.loads(r'''${ORDER[@]}''') if False else [
    'densenet201','inceptionv3','resnet101','resnet50','xception']
models = {
    'densenet201': r"${MODELS[densenet201]}",
    'inceptionv3': r"${MODELS[inceptionv3]}",
    'resnet101': r"${MODELS[resnet101]}",
    'resnet50': r"${MODELS[resnet50]}",
    'xception': r"${MODELS[xception]}"
}

def make_itp(model_path: str):
    try:
        from pycoral.utils.edgetpu import make_interpreter
        it = make_interpreter(model_path)
    except Exception:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        it = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1')])
    it.allocate_tensors()
    return it

# Pre-create interpreters and inputs
its = {}
inputs = {}
for name in order:
    it = make_itp(models[name])
    inp = it.get_input_details()[0]
    dt = getattr(inp['dtype'], '__name__', str(inp['dtype']))
    if dt == 'uint8':
        x = np.random.randint(0,256, size=inp['shape'], dtype=np.uint8)
    elif dt == 'int8':
        x = np.random.randint(-128,128, size=inp['shape'], dtype=np.int8)
    else:
        x = np.random.random_sample(inp['shape']).astype(np.float32)
    its[name] = it
    inputs[name] = (inp['index'], x)

# Optional warmup for each model (WARMUP env or default 0)
WARMUP = int(os.environ.get('WARMUP', '0'))
for name in order:
    it = its[name]
    idx, x = inputs[name]
    for _ in range(WARMUP):
        it.set_tensor(idx, x)
        it.invoke()
        _ = it.get_output_details()[0]['index']

def bt_now():
    try:
        import time
        return time.clock_gettime(time.CLOCK_BOOTTIME)
    except Exception:
        import time
        return float(open('/proc/uptime').read().split()[0])

spans = []
mutate = os.environ.get('MUTATE_INPUT','0') == '1'
for c in range(COUNT):
    for name in order:
        it = its[name]
        idx, x = inputs[name]
        if mutate:
            try:
                x.flat[0] = np.bitwise_xor(x.flat[0], np.array((c & 0x7), dtype=x.dtype))
            except Exception:
                pass
        t0 = bt_now()
        it.set_tensor(idx, x)
        it.invoke()
        t1 = bt_now()
        # force read
        out = it.get_output_details()[0]
        _ = it.get_tensor(out['index'])
        spans.append({'begin': t0, 'end': t1, 'seg_label': name})
        if GAP_MS > 0:
            time.sleep(GAP_MS/1000.0)

print(json.dumps({'spans': spans}, ensure_ascii=False))
PY

mv -f "$MERGED.tmp" "$MERGED" 2>/dev/null || true

# Stop capture and fix permissions
sleep 0.2 || true
kill "$CATPID" 2>/dev/null || true
sleep 0.1 || true
cat "$PW" | sudo -S -p '' chown "$USER:$USER" "$CAP" "$TM" "$MERGED" 2>/dev/null || true
cat "$PW" | sudo -S -p '' chmod 0644 "$CAP" "$TM" "$MERGED" 2>/dev/null || true

# Compute Bo envelope speeds per model using existing analyzer
ANA="$ROOT/tools/bo_envelope_span.py"
if [[ -f "$ANA" ]]; then
  echo "运行 Bo envelope 分析器: $ANA"
  python3 "$ANA" "$OUTDIR" > "$OUTDIR/bo_envelope_summary.json" || true
  echo "摘要保存: $OUTDIR/bo_envelope_summary.json"
else
  echo "警告: 未找到 $ANA，跳过 Bo envelope 分析"
fi

echo "完成，输出: $CAP, $TM, $MERGED"
