#!/usr/bin/env bash
set -euo pipefail

# 用法: ./run_usbmon_invoke.sh <model_path> <name> <out_dir> <bus> [margins]
# 例子: ./run_usbmon_invoke.sh './layered models/tpu/conv2d_3x3_stride2_edgetpu.tflite' 原单层 results/mn_layers/usbmon_bus0 0 "0.030 0.500 2.000"

if [[ $# -lt 4 ]]; then
  echo "用法: $0 <model_path> <name> <out_dir> <bus> [margins]" >&2
  exit 1
fi

MODEL_PATH="$1"
NAME="$2"
OUTDIR="$3"
BUS="$4"
MARGINS=${5:-"0.030 0.500"}

mkdir -p "$OUTDIR"

echo "[1/3] 采集 usbmon (BUS=$BUS) 并记录 invoke 时间..."
./run_usbmon_model.sh "$MODEL_PATH" "$NAME" "$OUTDIR" "$BUS" | cat

# 同步对齐校验：取 usbmon 头部 epoch 与第一条 invoke begin 的差
USB_EPOCH_HEAD=$(head -n1 "$OUTDIR/usbmon.txt" | awk '{print $1}')
INVOKE_EPOCH_BEGIN=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['spans'][0]['begin'])" "$OUTDIR/invokes.json")
echo "[check] usbmon_head_epoch=$USB_EPOCH_HEAD, invoke_begin=$INVOKE_EPOCH_BEGIN"
if python3 -c "import sys; a=float(sys.argv[1]); b=float(sys.argv[2]); print(abs(a-b)>0.1)" "$USB_EPOCH_HEAD" "$INVOKE_EPOCH_BEGIN" | grep -q true; then
  echo "[warn] 时间轴相差>0.1s，可能未对齐（但继续分析）"
fi

echo "[2/3] 文件头检查:"
wc -l "$OUTDIR/usbmon.txt" | cat
sed -n '1,8p' "$OUTDIR/usbmon.txt" | cat
echo '--- invokes first ---'
sed -n '1p' "$OUTDIR/invokes.json" | cat

echo "[3/3] 分别按多种扩窗做分析: $MARGINS"
source .venv/bin/activate
for M in $MARGINS; do
  tag=${M//./}
  echo "===== margin ${M}s ====="
  python3 analyze_usbmon.py "$OUTDIR/usbmon.txt" "$OUTDIR/invokes.json" "$OUTDIR/time_map.json" "$M" \
    | tee "$OUTDIR/io_split_margin_${tag}ms.json" \
    | sed -n '1,160p'
done

echo "完成，输出目录: $OUTDIR"


