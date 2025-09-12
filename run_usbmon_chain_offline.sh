#!/usr/bin/env bash
set -euo pipefail

# 用法: ./run_usbmon_chain_offline.sh <model_tpu_dir> <name> <out_dir> <bus> <duration_s>
# 
# 已回滚：恢复多interpreter重用方式，频繁重建导致EdgeTPU不稳定和性能下降

if [[ $# -lt 5 ]]; then
  echo "用法: $0 <model_tpu_dir> <name> <out_dir> <bus> <duration_s>" >&2
  exit 1
fi

TPU_DIR="$1"           # 包含 seg1..seg8_int8_edgetpu.tflite 的目录
NAME_BASE="$2"
OUTDIR="$3"
BUS="$4"
DUR="$5"

PW="/home/10210/Desktop/OS/password.text"
USBMON_NODE="/sys/kernel/debug/usb/usbmon/${BUS}u"
CAP="$OUTDIR/usbmon.txt"
TM="$OUTDIR/time_map.json"

# 记录次数（默认 10，可通过环境变量 COUNT 覆盖，例如 COUNT=101）
COUNT="${COUNT:-10}"

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
# 先清理同 BUS 上的历史采集进程，避免残留占用导致“卡住”
cat "$PW" | sudo -S -p '' pkill -f "/sys/kernel/debug/usb/usbmon/${BUS}u" 2>/dev/null || true
cat "$PW" | sudo -S -p '' modprobe usbmon || true
cat "$PW" | sudo -S -p '' sh -c ": > '$CAP'" || true
cat "$PW" | sudo -S -p '' sh -c "cat '$USBMON_NODE' > '$CAP'" &
CATPID=$!
# 确保异常退出时也能回收采集进程
trap 'kill "$CATPID" 2>/dev/null || true' INT TERM EXIT
sleep 0.1

# 启动映射记录器：读取首条 usbmon 行第2列时间戳（第1列为URB指针），与 BOOTTIME 建立映射
"$PY_MAP" - "$CAP" "$TM" <<'PY' &
import json,sys,time,re
cap, tm_path = sys.argv[1:]
usb_ts=None
# 典型 usbmon 行格式："ffff88003d8fd680 123456.789012 S Bo:1:002:... len=..."
# 第1列为URB地址（非数字），第2列为时间戳（秒）
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

# 采集前预留窗口（可通过 LEAD_S 指定，默认 0）
sleep "${LEAD_S:-0}"

# 运行链式推理：seg1..seg8，按环境变量 WARMUP 预热（默认10），记录每段invoke窗口（CLOCK_BOOTTIME）
"$RUN_PY" - "$TPU_DIR" "$OUTDIR" "$COUNT" <<'PY'
import sys, os, time, json, glob, numpy as np

def make_itp(model_path: str):
    try:
        from pycoral.utils.edgetpu import make_interpreter
        return make_interpreter(model_path)
    except Exception:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        return Interpreter(model_path, experimental_delegates=[load_delegate('libedgetpu.so.1')])

np.random.seed(123)
tpu_dir, out_dir, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
paths = []
def parse_stage_index(p: str) -> int:
    """从文件名推断阶段顺序：
    - segN_*.tflite -> N
    - tail_segX_to_Y_*.tflite -> X
    兜底返回一个较大值以保持稳定排序。
    """
    bn = os.path.basename(p)
    import re
    m = re.match(r"seg(\d+)_", bn)
    if m:
        return int(m.group(1))
    m = re.match(r"tail_seg(\d+)_?to_?\d+_", bn)
    if m:
        return int(m.group(1))
    return 9999

# 收集可用阶段模型：兼容 K<8 组合目录（含 tail_*）
cand_files = sorted(glob.glob(os.path.join(tpu_dir, "*_edgetpu.tflite")))
if not cand_files:
    print(f"NO TFLITE in {tpu_dir}")
    sys.exit(1)
cand_files.sort(key=parse_stage_index)
paths = cand_files

def stage_label(p: str) -> str:
    bn = os.path.basename(p)
    import re
    m = re.match(r"seg(\d+)_", bn)
    if m:
        return f"seg{int(m.group(1))}"
    m = re.match(r"tail_seg(\d+)_?to_?(\d+)_", bn)
    if m:
        return f"seg{int(m.group(1))}to{int(m.group(2))}"
    return "seg"

# 创建输出目录（按实际阶段标签）
num_stages = len(paths)
labels = [stage_label(p) for p in paths]
for lab in labels:
    os.makedirs(os.path.join(out_dir, lab), exist_ok=True)

# 构建解释器（一次性创建所有，然后重用）
itps = [make_itp(p) for p in paths]
for it in itps:
    it.allocate_tensors()

# 准备初始输入（seg1 的输入，固定随机种子保证复现）
inp0 = itps[0].get_input_details()[0]
dtype_name = getattr(inp0['dtype'], '__name__', str(inp0['dtype']))
if dtype_name == 'uint8':
    x0 = np.random.randint(0,256, inp0['shape'], dtype=np.uint8)
elif dtype_name == 'int8':
    x0 = np.random.randint(-128,128, inp0['shape'], dtype=np.int8)
else:
    x0 = np.random.random_sample(inp0['shape']).astype(np.float32)

def now():
    return time.clock_gettime(time.CLOCK_BOOTTIME)

WARMUP = int(os.environ.get('WARMUP', '10'))
GAP_MS = float(os.environ.get('STAGE_GAP_MS', '0') or 0)
GAP_S = GAP_MS / 1000.0
for _ in range(WARMUP):
    x = x0
    for it in itps:
        inp = it.get_input_details()[0]
        # 类型/形状对齐
        xi = x.astype(inp['dtype'])
        if list(xi.shape) != list(inp['shape']):
            xi = np.resize(xi, inp['shape']).astype(inp['dtype'])
        it.set_tensor(inp['index'], xi)
        it.invoke()
        out = it.get_output_details()[0]
        x = it.get_tensor(out['index'])
        if GAP_S > 0:
            time.sleep(GAP_S)

# 测量 COUNT 次，每阶段记录 invoke 时间窗口
seg_spans = {lab: [] for lab in labels}
for _ in range(count):
    x = x0
    for si, it in enumerate(itps, start=1):
        inp = it.get_input_details()[0]
        xi = x.astype(inp['dtype'])
        if list(xi.shape) != list(inp['shape']):
            xi = np.resize(xi, inp['shape']).astype(inp['dtype'])
        it.set_tensor(inp['index'], xi)
        t0 = now(); it.invoke(); t1 = now()
        lab = labels[si-1]
        seg_spans[lab].append({'begin': t0, 'end': t1})
        out = it.get_output_details()[0]
        x = it.get_tensor(out['index'])
        if GAP_S > 0:
            time.sleep(GAP_S)

# 写出每段 invokes.json（按实际阶段标签）
for lab in labels:
    with open(os.path.join(out_dir, lab, "invokes.json"), 'w') as f:
        json.dump({'name': f"chain_{lab}", 'spans': seg_spans[lab]}, f)
PY

if [[ "${STOP_ON_COUNT:-1}" == "1" ]]; then
  : # 直接结束采集（按 COUNT 完成即停）
else
  # 等待到持续时间结束
  sleep "$DUR"
fi

# 停止采集并修正权限（无论哪种模式都停止）
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


