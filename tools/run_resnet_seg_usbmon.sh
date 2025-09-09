#!/usr/bin/env bash
set -euo pipefail

VENV_PY="/home/10210/Desktop/OS/.venv/bin/python"
SYS_PY="$(command -v python3 || echo /usr/bin/python3)"
MODELS_DIR="/home/10210/Desktop/OS/layered models/resnet101_balanced/tflite"
RESULTS_BASE="/home/10210/Desktop/OS/results/resnet101_balanced_usbmon"

mkdir -p "$RESULTS_BASE"

if [[ ! -x "$VENV_PY" ]]; then
  echo "缺少虚拟环境 Python: $VENV_PY" >&2
  exit 1
fi

BUS_JSON="$($VENV_PY /home/10210/Desktop/OS/list_usb_buses.py)"
BUS="$($SYS_PY - <<'PY'
import sys, json
data = json.loads(sys.stdin.read() or '{}')
buses = data.get('buses') or []
print(buses[0] if buses else '')
PY
)"

if [[ -z "$BUS" ]]; then
  echo "未检测到 USB EdgeTPU 总线号" >&2
  exit 1
fi

echo "使用 BUS=$BUS"

for SEG in 0 1 2 3 4 5 6; do
  MODEL="$MODELS_DIR/resnet101_seg${SEG}_int8.tflite"
  OUTDIR="$RESULTS_BASE/seg${SEG}"
  mkdir -p "$OUTDIR"
  echo "=== 采集 seg${SEG} -> $OUTDIR ==="
  COUNT=100 /home/10210/Desktop/OS/run_usbmon_capture_offline.sh "$MODEL" "resnet101_seg${SEG}" "$OUTDIR" "$BUS" 12

  echo "分析 seg${SEG}（纯活跃时长，IN/OUT 并集）"
  "$SYS_PY" /home/10210/Desktop/OS/analyze_usbmon_out_active.py "$OUTDIR/usbmon.txt" "$OUTDIR/invokes.json" "$OUTDIR/time_map.json" > "$OUTDIR/out_active_union.json" || true
  "$SYS_PY" /home/10210/Desktop/OS/analyze_usbmon_in_active.py  "$OUTDIR/usbmon.txt" "$OUTDIR/invokes.json" "$OUTDIR/time_map.json" > "$OUTDIR/in_active_union.json"  || true
  "$SYS_PY" /home/10210/Desktop/OS/summarize_warm_usbmon.py "$OUTDIR/out_active_union.json" "$OUTDIR/in_active_union.json" > "$OUTDIR/warm_summary.json" || true

  echo "--- seg${SEG} warm 摘要 ---"
  if [[ -f "$OUTDIR/warm_summary.json" ]]; then
    cat "$OUTDIR/warm_summary.json"
  else
    echo "(无 warm_summary.json，可能采集或分析失败)"
  fi
  echo
done

echo "完成，结果目录: $RESULTS_BASE"




