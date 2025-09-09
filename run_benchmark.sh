#!/bin/bash
# 基准测试运行脚本 - 使用 .venv 虚拟环境

set -e  # 出错时退出

echo "=============================================="
echo "🍓 Raspberry Pi 5 Benchmark Test Suite"
echo "=============================================="
echo "使用虚拟环境 .venv 运行测试"
echo

# 检查虚拟环境是否存在
if [ ! -d ".venv" ]; then
    echo "❌ 虚拟环境 .venv 不存在!"
    echo "请先创建虚拟环境: python3 -m venv .venv"
    exit 1
fi

echo "✅ 找到虚拟环境 .venv"

# 激活虚拟环境
echo "🔄 激活虚拟环境..."
source .venv/bin/activate

# 检查必要的包
echo "🔍 检查必要的Python包..."
python3 -c "import numpy, pandas, matplotlib, tflite_runtime" 2>/dev/null || {
    echo "⚠️  缺少必要的包，正在安装..."
    pip install numpy pandas matplotlib
    pip install tflite-runtime
}

# 检查PyCoral（用于TPU测试）
python3 -c "import pycoral" 2>/dev/null || {
    echo "⚠️  PyCoral未安装，TPU测试将不可用"
    echo "如需TPU测试，请安装: pip install pycoral"
}

echo

# 函数：运行CPU测试
run_cpu_test() {
    echo "=============================================="
    echo "🖥️  开始CPU基准测试"
    echo "=============================================="
    
    if [ -f "autoCPU_simple.py" ]; then
        echo "运行 autoCPU_simple.py..."
        python3 autoCPU_simple.py
        echo "✅ CPU测试完成"
    else
        echo "❌ autoCPU_simple.py 文件不存在"
        return 1
    fi
}

# 函数：运行TPU测试
run_tpu_test() {
    echo "=============================================="
    echo "⚡ 开始TPU基准测试"
    echo "=============================================="
    
    if [ -f "autoTPU_simple.py" ]; then
        echo "运行 autoTPU_simple.py..."
        python3 autoTPU_simple.py
        echo "✅ TPU测试完成"
    else
        echo "❌ autoTPU_simple.py 文件不存在"
        return 1
    fi
}

# 函数：比较结果
compare_results() {
    echo "=============================================="
    echo "📊 开始结果对比"
    echo "=============================================="
    
    if [ -f "compare_results_simple.py" ]; then
        echo "运行 compare_results_simple.py..."
        python3 compare_results_simple.py
        echo "✅ 结果对比完成"
    else
        echo "❌ compare_results_simple.py 文件不存在"
        return 1
    fi
}

# 主菜单
while true; do
    echo
    echo "请选择要运行的测试:"
    echo "1) CPU基准测试"
    echo "2) TPU基准测试"
    echo "3) 比较CPU和TPU结果"
    echo "4) 运行所有测试"
    echo "5) 退出"
    echo
    read -p "请输入选择 (1-5): " choice

    case $choice in
        1)
            run_cpu_test
            ;;
        2)
            run_tpu_test
            ;;
        3)
            compare_results
            ;;
        4)
            echo "🚀 运行所有测试..."
            run_cpu_test
            echo
            run_tpu_test
            echo
            compare_results
            echo
            echo "🎉 所有测试完成！"
            echo "📁 结果保存在 ./results/ 目录下"
            ;;
        5)
            echo "👋 退出测试套件"
            break
            ;;
        *)
            echo "❌ 无效选择，请输入 1-5"
            ;;
    esac
done

# 退出虚拟环境
deactivate

echo
echo "=============================================="
echo "✅ 基准测试套件运行完成"
echo "=============================================="