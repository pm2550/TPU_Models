#!/usr/bin/env bash
set -euo pipefail

# 用法: ./run_usbmon_invoke_boottime.sh <model_path> <name> <out_dir>

if [[ $# -lt 3 ]]; then
  echo "用法: $0 <model_path> <name> <out_dir>" >&2
  exit 1
fi

MODEL_PATH="$1"
NAME="$2"
OUTDIR="$3"
PW="/home/10210/Desktop/OS/password.text"
USBMON_PATH="/sys/kernel/debug/usb/usbmon/0u"

mkdir -p "$OUTDIR"

if [[ ! -f "$PW" ]]; then
  echo "缺少密码文件: $PW" >&2
  exit 1
fi

CAP="$OUTDIR/usbmon.txt"
INVOKES_JSON="$OUTDIR/invokes.json"
TIME_MAP_JSON="$OUTDIR/time_map.json"

# 启动 usbmon 采集（聚合 0u），不注入 epoch，直接记录原始时间（第二列：boottime 微秒）
cat "$PW" | sudo -S -p '' modprobe usbmon || true
cat "$PW" | sudo -S -p '' sh -c ": > '$CAP'" || true
cat "$PW" | sudo -S -p '' sh -c "cat $USBMON_PATH > '$CAP'" &
CATPID=$!
sleep 0.05

# 建立 usbmon 内部时基与 CLOCK_BOOTTIME 的映射
python3 - <<'PY' "$CAP" "$TIME_MAP_JSON"
import json, sys, time, re
cap=sys.argv[1]; out=sys.argv[2]
usb_ts=None
deadline=time.time()+3.0
pat=re.compile(r"^\s*\S+\s+([0-9]+(?:\.[0-9]+)?)\s+")
while time.time()<deadline and usb_ts is None:
    try:
        with open(cap,'r',errors='ignore') as f:
            for ln in f:
                m=pat.match(ln)
                if m:
                    v=float(m.group(1))
                    usb_ts = v/1e6 if v>1e6 else v
                    break
    except FileNotFoundError:
        pass
    time.sleep(0.01)
try:
    bt_ref = time.clock_gettime(time.CLOCK_BOOTTIME)
except Exception:
    bt_ref = float(open('/proc/uptime').read().split()[0])
open(out,'w').write(json.dumps({'usbmon_ref': usb_ts, 'boottime_ref': bt_ref}, ensure_ascii=False))
PY

# Python 仅记录 invoke 的 CLOCK_BOOTTIME 窗口
"/home/10210/Desktop/OS/.venv/bin/python" - "$MODEL_PATH" <<'PY' > "$INVOKES_JSON"
import json, time, numpy as np, sys, os
from pycoral.utils.edgetpu import make_interpreter

def now_boottime() -> float:
    try:
        import time as _t
        return _t.clock_gettime(_t.CLOCK_BOOTTIME)
    except Exception:
        # 回退：用 /proc/uptime
        return float(open('/proc/uptime').read().split()[0])

m = sys.argv[1]
it = make_interpreter(m)
it.allocate_tensors()
inp = it.get_input_details()[0]

if inp['dtype'].__name__=='uint8':
    x = np.random.randint(0,256, size=inp['shape'], dtype=np.uint8)
else:
    x = np.random.randint(-128,128, size=inp['shape'], dtype=np.int8)

# 预热
for _ in range(50):
    it.set_tensor(inp['index'], x); it.invoke()

spans=[]
for i in range(10):
    it.set_tensor(inp['index'], x)
    t0 = now_boottime()
    it.invoke()
    t1 = now_boottime()
    spans.append({'begin': t0, 'end': t1})

print(json.dumps({'name': r"${NAME}", 'clock': 'boottime', 'spans': spans}))
PY

sleep 0.05
kill "$CATPID" 2>/dev/null || true
sleep 0.05 || true

cat "$PW" | sudo -S -p '' chown "$USER:$USER" "$CAP" "$INVOKES_JSON" 2>/dev/null || true
cat "$PW" | sudo -S -p '' chmod 0644 "$CAP" "$INVOKES_JSON" 2>/dev/null || true

echo "已保存: $INVOKES_JSON, $CAP"


