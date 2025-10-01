#!/usr/bin/env bash
set -euo pipefail

# Trace function calls during invoke windows using bpftrace uprobes gated by ldprobe_begin/end.
# Requirements: bpftrace, sudo privilege (uses password.text), libusb-1.0 present.
# Usage: tools/trace_invoke_calls_bt.sh --model <model.tflite> --out <out_dir> [--count 10] [--tail_ms 120]

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LIB="$ROOT_DIR/hook_and_track/libldprobe_gate.so"
PW_FILE="$ROOT_DIR/password.text"

MODEL=""; OUT=""; COUNT=10; TAIL_MS=120
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --count) COUNT="$2"; shift 2;;
    --tail_ms) TAIL_MS="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done
[[ -n "$MODEL" && -n "$OUT" ]] || { echo "Usage: $0 --model <model> --out <out_dir> [--count N] [--tail_ms MS]" >&2; exit 1; }

mkdir -p "$OUT"

make -C "$ROOT_DIR/hook_and_track" >/dev/null

# Prefer system python3; fallback to venv
PYBIN="$(command -v python3 || echo /usr/bin/python3)"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYBIN="$PYBIN"
fi

# Resolve absolute libusb path
LIBUSB=$(ldconfig -p | awk '/libusb-1.0.so/{print $4; exit}')
if [[ -z "${LIBUSB:-}" || ! -f "$LIBUSB" ]]; then
  echo "Cannot resolve libusb-1.0.so path; ldconfig -p failed" >&2; exit 1
fi

RUNPY="$OUT/run.py"
cat > "$RUNPY" <<PY
import numpy as np, time, os
m = r"$MODEL"
from pycoral.utils.edgetpu import make_interpreter
it = make_interpreter(m)
it.allocate_tensors()
inp=it.get_input_details()[0]
x = np.random.randint(0,256, size=inp['shape'], dtype=np.uint8 if inp['dtype'].__name__=='uint8' else np.int8)
import ctypes
libpath = os.environ.get('LD_PRELOAD','').split(':')[0]
lib = ctypes.CDLL(libpath) if libpath else ctypes.CDLL(None)
begin = getattr(lib, 'ldprobe_begin_invoke', None)
end = getattr(lib, 'ldprobe_end_invoke', None)
if begin is not None: begin.restype=None
if end is not None: end.argtypes=[ctypes.c_int]; end.restype=None
tail_ms = float(os.environ.get('INV_TAIL_MS','0') or '0')
for _ in range(2):
    if begin is not None: begin()
    it.set_tensor(inp['index'], x)
    it.invoke()
    _=it.get_tensor(it.get_output_details()[0]['index'])
    time.sleep(tail_ms/1000.0)
    if end is not None: end(0)
for i in range(int(os.environ.get('COUNT','10'))):
    if begin is not None: begin()
    it.set_tensor(inp['index'], x)
    it.invoke()
    _=it.get_tensor(it.get_output_details()[0]['index'])
    time.sleep(tail_ms/1000.0)
    if end is not None: end(0)
PY

BT_OUT="$OUT/invoke_calls_bt.txt"
LOG="$OUT/ldprobe.jsonl"

BT_PROG=$(mktemp)
cat > "$BT_PROG" <<BT
#!/usr/bin/env bpftrace
BEGIN { printf("start\\n"); }
uprobe:"$LIB":ldprobe_begin_invoke { @inv[pid] = 1; }
uprobe:"$LIB":ldprobe_end_invoke   { @inv[pid] = 0; }
uprobe:"$LIBUSB":libusb_submit_transfer     { if (@inv[pid]) { time("%s "); printf("LIBUSB_SUBMIT pid=%d\\n", pid); } }
uprobe:"$LIBUSB":libusb_cancel_transfer     { if (@inv[pid]) { time("%s "); printf("LIBUSB_CANCEL pid=%d\\n", pid); } }
uprobe:"$LIBUSB":libusb_bulk_transfer       { if (@inv[pid]) { time("%s "); printf("LIBUSB_BULK pid=%d\\n", pid); } }
uprobe:"$LIBUSB":libusb_interrupt_transfer  { if (@inv[pid]) { time("%s "); printf("LIBUSB_INTR pid=%d\\n", pid); } }
uprobe:"$LIBUSB":libusb_control_transfer    { if (@inv[pid]) { time("%s "); printf("LIBUSB_CTRL pid=%d\\n", pid); } }
END { printf("done\\n"); }
BT

echo "Starting bpftrace (requires sudo)…" >&2
PW=$(cat "$PW_FILE")
set +e
echo "$PW" | sudo -S -p '' bpftrace "$BT_PROG" > "$BT_OUT" 2>"$OUT/invoke_calls_bt.err" &
BT_PID=$!
# wait up to 5s for bpftrace to start
for i in $(seq 1 50); do
  if [ -s "$BT_OUT" ] && grep -q start "$BT_OUT" 2>/dev/null; then break; fi
  sleep 0.1
done
INV_TAIL_MS="$TAIL_MS" COUNT="$COUNT" LDP_LOG="$LOG" LDP_MEM_THRESHOLD=64 LDP_MEM_ONLY_IN_INVOKE=1 LD_PRELOAD="$LIB" "$PYBIN" "$RUNPY"
APP_RC=$?
sleep 0.3
kill "$BT_PID" >/dev/null 2>&1 || true
wait "$BT_PID" 2>/dev/null || true
set -e

echo "bpftrace log: $BT_OUT"
echo "ldprobe log: $LOG"
