#!/bin/bash
# Wave Reasoning 最小复现脚本 (Linux/macOS)
# 用法: bash scripts/repro_tiny.sh

set -e

echo "Wave Reasoning - Quick Reproduction"
echo "=================================================="

# 检查虚拟环境
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

# 运行复现脚本
$PYTHON scripts/repro_tiny.py

echo ""
echo "Done!"

