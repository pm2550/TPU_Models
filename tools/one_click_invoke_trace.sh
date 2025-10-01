#!/usr/bin/env bash
set -euo pipefail

# One-click: run a model with LD_PRELOAD probe and collect function names
# Usage: tools/one_click_invoke_trace.sh --model <model.tflite> --out <out_dir> [--count 5]

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LIB="$ROOT_DIR/hook_and_track/libldprobe_gate.so"

MODEL=""
OUT=""
COUNT=5
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --count) COUNT="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$MODEL" || -z "$OUT" ]]; then
  echo "Usage: $0 --model <model.tflite> --out <out_dir> [--count N]" >&2
  exit 1
fi

mkdir -p "$OUT"

echo "[1/5] Building probe libs" >&2
make -C "$ROOT_DIR/hook_and_track" >/dev/null

# Prefer system python3 for better delegate compatibility; fallback to venv
PYBIN="$(command -v python3 || echo /usr/bin/python3)"
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYBIN_SYS="$PYBIN"
  PYBIN_VENV="${ROOT_DIR}/.venv/bin/python"
  # choose system first
  PYBIN="$PYBIN_SYS"
fi

RUNPY="$OUT/run.py"
cat > "$RUNPY" <<'PY'
import numpy as np, time, os, sys
m = os.environ.get('MODEL') or r"$MODEL"
it = None
err = None
try:
    from pycoral.utils.edgetpu import make_interpreter
    it = make_interpreter(m)
except Exception as e:
    err = e
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        cands = [
            'libedgetpu.so.1',
            '/opt/edgetpu/std/libedgetpu.so.1',
            '/opt/edgetpu/max/libedgetpu.so.1',
            '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1',
            '/usr/lib/arm-linux-gnueabihf/libedgetpu.so.1',
            '/usr/lib/arm64-linux-gnu/libedgetpu.so.1',
            '/usr/local/lib/libedgetpu.so.1',
            '/usr/lib/libedgetpu.so.1',
        ]
        for c in cands:
            try:
                it = Interpreter(model_path=m, experimental_delegates=[load_delegate(c)])
                break
            except Exception:
                it = None
        if it is None:
            sys.stderr.write(f'EDGETPU_DELEGATE_LOAD_FAIL after pycoral err={err}\n')
            raise
    except Exception:
        raise
it.allocate_tensors()
inp = it.get_input_details()[0]
if inp['dtype'].__name__=='uint8':
    x = np.random.randint(0,256, size=inp['shape'], dtype=np.uint8)
else:
    x = np.random.randint(-128,128, size=inp['shape'], dtype=np.int8)
import ctypes
libpath = os.environ.get('LD_PRELOAD','').split(':')[0]
lib = ctypes.CDLL(libpath) if libpath else ctypes.CDLL(None)
begin = getattr(lib, 'ldprobe_begin_invoke', None)
end = getattr(lib, 'ldprobe_end_invoke', None)
if begin is not None: begin.restype=None
if end is not None: end.argtypes=[ctypes.c_int]; end.restype=None
try:
    tail_ms = float(os.environ.get('INV_TAIL_MS','0') or '0')
except Exception:
    tail_ms = 0.0
for _ in range(2):
    if begin is not None: begin()
    it.set_tensor(inp['index'], x)
    it.invoke()
    _ = it.get_tensor(it.get_output_details()[0]['index'])
    if tail_ms>0: time.sleep(tail_ms/1000.0)
    if end is not None: end(0)
for i in range(int(os.environ.get('COUNT', '$COUNT'))):
    if begin is not None: begin()
    it.set_tensor(inp['index'], x)
    it.invoke()
    _ = it.get_tensor(it.get_output_details()[0]['index'])
    if tail_ms>0: time.sleep(tail_ms/1000.0)
    if end is not None: end(0)
    time.sleep(0.01)
print('done')
PY

ABS="$ROOT_DIR"
LOG="$OUT/ldprobe.jsonl"
export LDP_LOG="$LOG"
export LDP_MEM_THRESHOLD=64
export LDP_MEM_ONLY_IN_INVOKE=1

# Ensure EdgeTPU shared libs are discoverable for both pycoral and tflite_runtime
if [[ -d "/opt/edgetpu/std" ]]; then
  export LD_LIBRARY_PATH="/opt/edgetpu/std:${LD_LIBRARY_PATH:-}"
fi
if [[ -d "/opt/edgetpu/max" ]]; then
  export LD_LIBRARY_PATH="/opt/edgetpu/max:${LD_LIBRARY_PATH:-}"
fi

echo "[2/5] Trying Frida tracing (if available)" >&2
if command -v frida-trace >/dev/null 2>&1; then
  set +e
  frida-trace -f /usr/bin/env \
    -o "$OUT/frida_trace.txt" \
    -i 'libldprobe_gate.so*!ldprobe_*' \
    -i 'libedgetpu.so*!*' -i 'libusb-1.0.so*!*' -i 'libtensorflowlite*.so*!*' -i 'libtflite*.so*!*' -- \
    LD_PRELOAD="$LIB" LDP_LOG="$LOG" LDP_MEM_THRESHOLD=64 LDP_MEM_ONLY_IN_INVOKE=1 "$PYBIN" "$RUNPY"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then echo "Frida failed (rc=$rc), will fallback." >&2; fi
else
  echo "Frida not found; skipping." >&2
fi

echo "[3/5] Running with LD_PRELOAD to produce timeline" >&2
LD_PRELOAD="$LIB" "$PYBIN" "$RUNPY"

echo "[4/5] Building timeline.json" >&2
"$PYBIN" "$ROOT_DIR/tools/build_invoke_timeline.py" "$LOG" -o "$OUT/timeline.json"

echo "[5/5] If Frida not available, using LD_DEBUG to list symbols" >&2
if [[ ! -s "$OUT/frida_trace.txt" ]]; then
  # capture symbol resolutions and bindings
  LD_DEBUG=libs,symbols,bindings LD_DEBUG_OUTPUT="$OUT/lddbg" LD_PRELOAD="$LIB" "$PYBIN" "$RUNPY" || true
  # extract symbol names related to our target libs from loader logs
  (
    grep -ho "binding.*symbol '.*'" "$OUT"/lddbg* 2>/dev/null || true
    grep -ho "symbol=\w\+" "$OUT"/lddbg* 2>/dev/null || true
  ) | sed "s/.*symbol '\(.*\)'.*/\1/; s/.*symbol=\(.*\)$/\1/" | \
    sort -u > "$OUT/calls_lddebug_all.txt" || true
  # further narrow by libs hints, if present in the same lines
  grep -E 'edgetpu|usb|tflite|tensorflowlite' "$OUT/calls_lddebug_all.txt" > "$OUT/calls_lddebug.txt" || true
fi

# If frida trace exists, post-process into per-invoke calls
if [[ -s "$OUT/frida_trace.txt" ]]; then
  "$PYBIN" "$ROOT_DIR/tools/parse_frida_calls_per_invoke.py" "$OUT/frida_trace.txt" -o "$OUT/calls_per_invoke.json" || true
fi
"$PYBIN" "$ROOT_DIR/tools/parse_ldprobe_calls_per_invoke.py" "$LOG" -o "$OUT/calls_per_invoke_ldprobe.json" || true

echo "Done. Outputs:"
echo "  timeline: $OUT/timeline.json"
if [[ -s "$OUT/frida_trace.txt" ]]; then echo "  calls(frida): $OUT/frida_trace.txt"; fi
if [[ -s "$OUT/calls_lddebug.txt" ]]; then echo "  calls(lddebug): $OUT/calls_lddebug.txt"; fi
if [[ -s "$OUT/calls_per_invoke.json" ]]; then echo "  calls_per_invoke(frida): $OUT/calls_per_invoke.json"; fi
if [[ -s "$OUT/calls_per_invoke_ldprobe.json" ]]; then echo "  calls_per_invoke(ldprobe): $OUT/calls_per_invoke_ldprobe.json"; fi
