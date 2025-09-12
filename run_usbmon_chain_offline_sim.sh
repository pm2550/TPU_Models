#!/usr/bin/env bash
set -euo pipefail

# 用法: ./run_usbmon_chain_offline_sim.sh <tpu_dir> <name> <out_dir> <bus> <duration_s>

if [[ $# -lt 5 ]]; then
  echo "用法: $0 <tpu_dir> <name> <out_dir> <bus> <duration_s>" >&2
  exit 1
fi

TPU_DIR="$1"       # seg1..seg8_int8_edgetpu.tflite 所在目录
NAME_BASE="$2"
OUTDIR="$3"
BUS="$4"
DUR="$5"

PW="/home/10210/Desktop/OS/password.text"
USBMON_NODE="/sys/kernel/debug/usb/usbmon/${BUS}u"
CAP="$OUTDIR/usbmon.txt"
TM="$OUTDIR/time_map.json"

# 每段一次，循环100次（无预热）
COUNT=100

PY_SYS="$(command -v python3)"
PY_MAP="${PY_SYS:-/usr/bin/python3}"
PY_ANALYZE="${PY_SYS:-/usr/bin/python3}"

if [[ -x "/home/10210/Desktop/OS/.venv/bin/python" ]]; then
  RUN_PY="/home/10210/Desktop/OS/.venv/bin/python"
elif [[ -n "${PY_SYS}" ]]; then
  RUN_PY="${PY_SYS}"
else
  RUN_PY="/usr/bin/python3"
fi

mkdir -p "$OUTDIR"

if [[ ! -f "$PW" ]]; then
  echo "缺少密码文件: $PW" >&2
  exit 1
fi

# 启动 usbmon 采集
cat "$PW" | sudo -S -p '' modprobe usbmon || true
cat "$PW" | sudo -S -p '' sh -c ": > '$CAP'" || true
cat "$PW" | sudo -S -p '' sh -c "cat '$USBMON_NODE' > '$CAP'" &
CATPID=$!
sleep 0.1

# 启动映射记录器
"$PY_MAP" - "$CAP" "$TM" <<'PY' &
import json,sys,time,re
cap, tm_path = sys.argv[1:]
usb_ts=None
pat=re.compile(r"^\s*\S+\s+([0-9]+(?:\.[0-9]+)?)\s+")
deadline=time.time()+10.0
while time.time()<deadline and usb_ts is None:
    try:
        with open(cap,'r',errors='ignore') as f:
            lines=f.readlines()
            for ln in lines:
                m=pat.match(ln)
                if m:
                    v=float(m.group(1))
                    usb_ts=v/1e6 if v>1e6 else v
                    break
    except FileNotFoundError:
        pass
    if usb_ts is None:
        time.sleep(0.02)
if usb_ts is None:
    open(tm_path,'w').write(json.dumps({'usbmon_ref': None, 'boottime_ref': None}))
    raise SystemExit(0)
try:
    bt_ref=time.clock_gettime(time.CLOCK_BOOTTIME)
except Exception:
    bt_ref=float(open('/proc/uptime').read().split()[0])
open(tm_path,'w').write(json.dumps({'usbmon_ref': usb_ts, 'boottime_ref': bt_ref}))
PY

# 等待 time_map.json 就绪
"$PY_MAP" - "$TM" <<'PY'
import json,sys,time,os
tm_path=sys.argv[1]
deadline=time.time()+5.0
ready=False
while time.time()<deadline and not ready:
    try:
        j=json.load(open(tm_path))
        ready=(j.get('usbmon_ref') is not None) and (j.get('boottime_ref') is not None)
    except Exception:
        ready=False
    if not ready:
        time.sleep(0.02)
print('time_map_ready=', ready)
PY

# 运行链式（模拟输入，不串接输出）
"$RUN_PY" - "$TPU_DIR" "$OUTDIR" "$COUNT" <<'PY'
import sys, os, time, json, glob, numpy as np

def make_itp(model_path: str):
    try:
        from pycoral.utils.edgetpu import make_interpreter
        return make_interpreter(model_path)
    except Exception:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        return Interpreter(model_path, experimental_delegates=[load_delegate('libedgetpu.so.1')])

tpu_dir, out_dir, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
paths = []
for i in range(1, 9):
    cands = sorted(glob.glob(os.path.join(tpu_dir, f"seg{i}_*_edgetpu.tflite")))
    if not cands:
        cands = sorted(glob.glob(os.path.join(tpu_dir, f"seg{i}_int8_edgetpu.tflite")))
    if not cands:
        print(f"MISSING seg{i} in {tpu_dir}")
        sys.exit(1)
    paths.append(cands[0])

for i in range(1, 9):
    os.makedirs(os.path.join(out_dir, f"seg{i}"), exist_ok=True)

itps = [make_itp(p) for p in paths]
for it in itps:
    it.allocate_tensors()

def now():
    return time.clock_gettime(time.CLOCK_BOOTTIME)

seg_spans = {i: [] for i in range(1,9)}
# 外层循环100次，每次按 seg1..seg8 串行各一次
for _ in range(count):
    for si, it in enumerate(itps, start=1):
        inp = it.get_input_details()[0]
        # 随机输入（不串接上一段输出）
        dtype_name = getattr(inp['dtype'], '__name__', str(inp['dtype']))
        if dtype_name == 'uint8':
            x = np.random.randint(0,256, inp['shape'], dtype=np.uint8)
        elif dtype_name == 'int8':
            x = np.random.randint(-128,128, inp['shape'], dtype=np.int8)
        else:
            x = np.random.random_sample(inp['shape']).astype(np.float32)
        it.set_tensor(inp['index'], x)
        t0 = now(); it.invoke(); t1 = now()
        _ = it.get_tensor(it.get_output_details()[0]['index'])  # 触发 TPU 回传
        seg_spans[si].append({'begin': t0, 'end': t1})

for si in range(1,9):
    with open(os.path.join(out_dir, f"seg{si}", "invokes.json"), 'w') as f:
        json.dump({'name': f"sim_chain_seg{si}", 'spans': seg_spans[si]}, f)
PY

# 等待到持续时间结束
sleep "$DUR"

# 停止采集并修正权限
kill "$CATPID" 2>/dev/null || true
sleep 0.1 || true
cat "$PW" | sudo -S -p '' chown "$USER:$USER" "$CAP" "$TM" 2>/dev/null || true
cat "$PW" | sudo -S -p '' chmod 0644 "$CAP" "$TM" 2>/dev/null || true

# 针对每段运行分析，生成 seg*/io_split_bt.json
for i in 1 2 3 4 5 6 7 8; do
  IV="$OUTDIR/seg${i}/invokes.json"
  OUT_JSON="$OUTDIR/seg${i}/io_split_bt.json"
  if [[ -f "$IV" ]]; then
    "$PY_ANALYZE" - "$CAP" "$IV" "$TM" <<'PY' | tee "$OUT_JSON" >/dev/null
import json,sys,re,os
cap, ivp, tmp = sys.argv[1:]
tm=json.load(open(tmp)) if os.path.exists(tmp) else {'usbmon_ref':None,'boottime_ref':None}
iv=json.load(open(ivp))['spans']
usb_ref=tm.get('usbmon_ref'); bt_ref=tm.get('boottime_ref')
recs=[]
with open(cap,'r',errors='ignore') as f:
    for ln in f:
        parts=ln.split()
        if len(parts)<3: continue
        try:
            ts=float(parts[1]); ts=ts/1e6 if ts>1e6 else ts
        except: continue
        tok=None; idx=None
        for i,t in enumerate(parts):
            if re.match(r'^[BC][io]:\d+:', t): tok=t[:2]; idx=i; break
        if not tok: continue
        m=re.search(r'len=(\d+)', ln); nb=0
        if m: nb=int(m.group(1))
        elif idx is not None and idx+2<len(parts):
            try: nb=int(parts[idx+2])
            except: nb=0
        recs.append((ts,tok,nb))

def stat(win, extra=0.0):
    if usb_ref is None or bt_ref is None:
        return {'note':'no_ref','span_s':win['end']-win['begin'],'Bi':0,'Bo':0,'Ci':0,'Co':0,'bytes_in':0,'bytes_out':0,'MBps_in':0.0,'MBps_out':0.0}
    b0 = win['begin'] - bt_ref + usb_ref - extra
    e0 = win['end']   - bt_ref + usb_ref + extra
    bi=bo=ci=co=0; binb=boutb=0
    for ts,d,nb in recs:
        if b0<=ts<=e0:
            if d=='Bi': bi+=1; binb+=nb
            elif d=='Bo': bo+=1; boutb+=nb
            elif d=='Ci': ci+=1; binb+=nb
            elif d=='Co': co+=1; boutb+=nb
    span=e0-b0
    toMB=lambda x: x/(1024*1024.0)
    return dict(span_s=span,Bi=bi,Bo=bo,Ci=ci,Co=co,bytes_in=binb,bytes_out=boutb,MBps_in=(toMB(binb)/span if span>0 else 0.0),MBps_out=(toMB(boutb)/span if span>0 else 0.0))

strict=[stat(iv[i],0.0) for i in range(len(iv))]
loose=[stat(iv[i],0.010) for i in range(len(iv))]
def agg(arr):
    s={'bytes_in':0,'bytes_out':0,'span_s':0.0}
    for x in arr:
        s['bytes_in']+=x['bytes_in']; s['bytes_out']+=x['bytes_out']; s['span_s']+=x['span_s']
    toMB=lambda x: x/(1024*1024.0)
    s['MBps_in']=toMB(s['bytes_in'])/s['span_s'] if s['span_s']>0 else 0.0
    s['MBps_out']=toMB(s['bytes_out'])/s['span_s'] if s['span_s']>0 else 0.0
    return s
print(json.dumps({'strict_first':strict[0] if strict else {},'loose_first':loose[0] if loose else {},'strict_overall':agg(strict),'loose_overall':agg(loose)}, ensure_ascii=False, indent=2))
PY
  fi
done

echo "完成，输出: $OUTDIR"


