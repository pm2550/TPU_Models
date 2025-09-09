#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "用法: $0 <model_path> <name> <out_dir> <bus>" >&2
  exit 1
fi

MODEL_PATH="$1"
NAME="$2"
OUTDIR="$3"
BUS="$4"
mkdir -p "$OUTDIR"

# 加载 usbmon 并开始 ASCII 采集 (按指定 BUS)
CAP="$OUTDIR/usbmon.txt"

# 一律使用绝对路径密码文件 sudo -S
PW="/home/10210/Desktop/OS/password.text"
if [[ ! -f "$PW" ]]; then
  echo "缺少密码文件: $PW" >&2; exit 1
fi
cat "$PW" | sudo -S -p '' modprobe usbmon || true
cat "$PW" | sudo -S -p '' sh -c ": > '$CAP'" || true
cat "$PW" | sudo -S -p '' sh -c "/usr/bin/env python3 /home/10210/Desktop/OS/tools/prepend_epoch.py < /sys/kernel/debug/usb/usbmon/${BUS}u > '$CAP'" &
CATPID=$!
sleep 0.05

# 等待捕获文件出现至少一行 ' C ' 完成事件，记录与此刻 epoch 的映射
TIME_MAP_JSON="$OUTDIR/time_map.json"
python3 - "$CAP" <<'PY' > "$TIME_MAP_JSON.tmp" || true
import json, sys, time, re
cap=sys.argv[1]
ts_usb=None
deadline=time.time()+3.0
while time.time()<deadline and ts_usb is None:
    try:
        with open(cap,'r',errors='ignore') as f:
            for ln in f:
                cols=ln.strip().split()
                if not cols:
                    continue
                # 兼容 prepend_epoch: <epoch> <tag> <usec> ...
                v=None
                if len(cols) >= 3 and re.fullmatch(r"\d+", cols[2]):
                    v=float(cols[2])
                else:
                    # 回退：扫描首个数字
                    for tok in cols:
                        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
                            v=float(tok); break
                if v is not None:
                    ts_usb = v/1e6 if v>1e6 else v
                    break
    except FileNotFoundError:
        pass
    time.sleep(0.01)
ts_epoch=time.time()
try:
    up=float(open('/proc/uptime').read().split()[0])
except Exception:
    up=None
bt = (ts_epoch - up) if (up is not None) else None
print(json.dumps({
    'epoch_ref_at_usbmon': ts_epoch,
    'usbmon_ref': ts_usb,
    'uptime_ref_at_usbmon': up,
    'boottime_ref': bt
}, ensure_ascii=False))
PY
mv -f "$TIME_MAP_JSON.tmp" "$TIME_MAP_JSON" 2>/dev/null || true

# 运行模型：记录每次 invoke 的起止绝对时间
INVOKES_JSON="$OUTDIR/invokes.json"
"/home/10210/Desktop/OS/.venv/bin/python" - "$MODEL_PATH" <<'PY' > "$INVOKES_JSON"
import json, time as t, numpy as np, sys

def make_itp(model_path: str):
    try:
        from pycoral.utils.edgetpu import make_interpreter
        return make_interpreter(model_path)
    except Exception:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        return Interpreter(model_path, experimental_delegates=[load_delegate('libedgetpu.so.1')])

m=sys.argv[1]
it=make_itp(m); it.allocate_tensors(); inp=it.get_input_details()[0]
dtype_name = inp['dtype'].__name__ if hasattr(inp['dtype'], '__name__') else str(inp['dtype'])
if dtype_name == 'uint8':
    x = np.random.randint(0,256, inp['shape'], dtype=np.uint8)
elif dtype_name == 'int8':
    x = np.random.randint(-128,128, inp['shape'], dtype=np.int8)
else:
    x = np.random.random_sample(inp['shape']).astype(np.float32)

for _ in range(50):
    it.set_tensor(inp['index'], x); it.invoke()
spans=[]
for i in range(10):
    it.set_tensor(inp['index'], x)
    t0=t.time(); it.invoke(); t1=t.time()
    spans.append({'begin': t0, 'end': t1})
print(json.dumps({'name': r"${NAME}", 'spans': spans}))
PY

sleep 0.05
kill "$CATPID" 2>/dev/null || true
sleep 0.05 || true
cat "$PW" | sudo -S -p '' chown "$USER:$USER" "$CAP" "$INVOKES_JSON" "$TIME_MAP_JSON" 2>/dev/null || true
cat "$PW" | sudo -S -p '' chmod 0644 "$CAP" "$INVOKES_JSON" "$TIME_MAP_JSON" 2>/dev/null || true

echo "已保存: $INVOKES_JSON, $CAP, $TIME_MAP_JSON"


