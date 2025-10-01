#!/usr/bin/env bash
set -euo pipefail

# Usage:
#  tools/run_with_ldprobe.sh --model <model.tflite> --out <out_dir> [--count 10] [--audit 1]
# Or: tools/run_with_ldprobe.sh --out <out_dir> [--audit 1] -- cmd args...

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LIB="$ROOT_DIR/hook_and_track/libldprobe_gate.so"
AUD="$ROOT_DIR/hook_and_track/liblgaudit.so"

function build_lib() {
  make -C "$ROOT_DIR/hook_and_track" >/dev/null
}

function choose_python() {
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    echo "$ROOT_DIR/.venv/bin/python"
  else
    command -v python3 || echo /usr/bin/python3
  fi
}

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 --model <model> --out <dir> [--count N] | --out <dir> -- <cmd>..." >&2
  exit 1
fi

MODEL=""
OUT=""
COUNT=10
AUDIT=1
CMD=("")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --count) COUNT="$2"; shift 2;;
    --audit) AUDIT="$2"; shift 2;;
    --) shift; CMD=("$@"); break;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$OUT" ]]; then
  echo "--out is required" >&2; exit 1
fi

mkdir -p "$OUT"
LOG="$OUT/ldprobe.jsonl"

build_lib

export LDP_LOG="$LOG"
export LDP_MEM_THRESHOLD="64"    # ignore tiny copies
export LDP_MEM_ONLY_IN_INVOKE="1" # capture memcpy only inside invoke brackets
if [[ "$AUDIT" == "1" ]]; then
  export LD_AUDIT="$AUD"
  export LDP_AUDIT_LOG="$OUT/audit.jsonl"
  # default include patterns (comma-separated substrings)
  export LDP_AUDIT_INCLUDE="libedgetpu,libusb,libtflite,libtensorflowlite"
fi

if [[ -n "$MODEL" ]]; then
  PY=$(choose_python)
  set +e
  MODEL="$MODEL" COUNT="$COUNT" LD_PRELOAD="$LIB" "$PY" - <<PY
import numpy as np, time, os
m = r"$MODEL"
it=None
try:
    from pycoral.utils.edgetpu import make_interpreter
    it=make_interpreter(m)
except Exception:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    it=Interpreter(model_path=m, experimental_delegates=[load_delegate('libedgetpu.so.1')])
it.allocate_tensors()
inp=it.get_input_details()[0]
if inp['dtype'].__name__=='uint8':
    x=np.random.randint(0,256, size=inp['shape'], dtype=np.uint8)
else:
    x=np.random.randint(-128,128, size=inp['shape'], dtype=np.int8)
# call ldprobe brackets via ctypes to ensure INV_* events even if C++ path bypasses TfLite C API
import ctypes
lib = ctypes.CDLL(None)
begin = getattr(lib, 'ldprobe_begin_invoke', None)
end = getattr(lib, 'ldprobe_end_invoke', None)
if begin is not None:
    begin.restype=None
if end is not None:
    end.argtypes=[ctypes.c_int]
    end.restype=None
for _ in range(3): it.set_tensor(inp['index'], x); it.invoke()
for i in range(int($COUNT)):
    # place INV window to include set_tensor (Host-pre) and out readback (Host-post)
    if begin is not None: begin()
    it.set_tensor(inp['index'], x)
    it.invoke()
    out=it.get_tensor(it.get_output_details()[0]['index'])
    _chk=int(np.sum(out) % 2**32)
    if end is not None: end(0)
    time.sleep(0.01)
print('done')
PY
  rc=$?
  set -e
  if [[ $rc -ne 0 && "$AUDIT" == "1" ]]; then
  echo "AUDIT run crashed (rc=$rc). Falling back to --audit 0." >&2
    unset LD_AUDIT
    export AUDIT=0
  MODEL="$MODEL" COUNT="$COUNT" LD_PRELOAD="$LIB" "$PY" - <<'PY'
import numpy as np, time, os
m = os.environ['MODEL']
it=None
try:
    from pycoral.utils.edgetpu import make_interpreter
    it=make_interpreter(m)
except Exception:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    it=Interpreter(model_path=m, experimental_delegates=[load_delegate('libedgetpu.so.1')])
it.allocate_tensors()
inp=it.get_input_details()[0]
if inp['dtype'].__name__=='uint8':
    x = np.random.randint(0,256, size=inp['shape'], dtype=np.uint8)
else:
    x = np.random.randint(-128,128, size=inp['shape'], dtype=np.int8)
import ctypes
lib = ctypes.CDLL(None)
begin = getattr(lib, 'ldprobe_begin_invoke', None)
end = getattr(lib, 'ldprobe_end_invoke', None)
if begin is not None: begin.restype=None
if end is not None: end.argtypes=[ctypes.c_int]; end.restype=None
for _ in range(2): it.set_tensor(inp['index'], x); it.invoke()
for i in range(int(os.environ.get('COUNT','5'))):
    if begin is not None: begin()
    it.set_tensor(inp['index'], x)
    it.invoke()
    out=it.get_tensor(it.get_output_details()[0]['index'])
    _chk=int(np.sum(out) % 2**32)
    if end is not None: end(0)
    time.sleep(0.01)
print('done')
PY
    # Try ltrace to capture symbol names if available
    if command -v ltrace >/dev/null 2>&1; then
      RUNPY="$OUT/run_invoke.py"
      cat > "$RUNPY" <<'PY'
import numpy as np, time, os
m = os.environ['MODEL']
cnt = int(os.environ.get('COUNT','5'))
it=None
try:
    from pycoral.utils.edgetpu import make_interpreter
    it=make_interpreter(m)
except Exception:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    it=Interpreter(model_path=m, experimental_delegates=[load_delegate('libedgetpu.so.1')])
it.allocate_tensors()
inp=it.get_input_details()[0]
if inp['dtype'].__name__=='uint8':
    x = np.random.randint(0,256, size=inp['shape'], dtype=np.uint8)
else:
    x = np.random.randint(-128,128, size=inp['shape'], dtype=np.int8)
for _ in range(1):
    it.set_tensor(inp['index'], x)
    it.invoke()
for i in range(cnt):
    it.set_tensor(inp['index'], x)
    it.invoke()
    _=it.get_tensor(it.get_output_details()[0]['index'])
PY
      LTRACE_OUT="$OUT/calls_ltrace.txt"
      echo "ltrace fallback is running to capture symbol names..." >&2
      MODEL="$MODEL" COUNT="$COUNT" LD_PRELOAD="$LIB" ltrace -f -tt -o "$LTRACE_OUT" -e 'libedgetpu*+libusb*+libtensorflowlite*+libtflite*' "$PY" "$RUNPY" || true
    else
      echo "ltrace not found; to install: sudo apt-get install -y ltrace" >&2
    fi
  fi
else
  if [[ "${#CMD[@]}" -eq 1 && -z "${CMD[0]}" ]]; then
    echo "Provide either --model or an explicit command after --" >&2
    exit 1
  fi
  LD_PRELOAD="$LIB" "${CMD[@]}"
fi

# Build timeline and callgraph
python3 "$ROOT_DIR/tools/build_invoke_timeline.py" "$LOG" -o "$OUT/timeline.json" || true
if [[ -f "$OUT/audit.jsonl" ]]; then
  python3 "$ROOT_DIR/tools/build_invoke_callgraph.py" --ldp "$LOG" --audit "$OUT/audit.jsonl" -o "$OUT/calls.json" || true
fi
echo "LDPROBE log: $LOG"
echo "Timeline: $OUT/timeline.json"
if [[ -f "$OUT/calls.json" ]]; then echo "Calls: $OUT/calls.json"; fi
if [[ -f "$OUT/calls_ltrace.txt" ]]; then echo "Calls(ltrace): $OUT/calls_ltrace.txt"; fi
