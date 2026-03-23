#!/bin/bash

echo "=========================================="
echo "perf 性能分析脚本"
echo "=========================================="

# 先编译程序
echo ""
echo "1. 编译程序..."
g++ -std=c++11 -O2 -g main.cpp mat_vec_bad.cpp mat_vec_fast.cpp arr_sum_slow.cpp arr_sum_fast.cpp -o program

if [ ! -f ./program ]; then
    echo "编译失败！"
    exit 1
fi

echo "编译成功！"

# 1. 统计核心性能指标
echo ""
echo "2. 统计性能指标 (cycles, instructions, cache-misses)..."
echo "----------------------------------------"
sudo perf stat -e cycles,instructions,cache-misses,L1-dcache-load-misses ./program

# 2. 记录缓存缺失事件
echo ""
echo "3. 采样记录 cache-misses 事件..."
sudo perf record -e cache-misses -o perf.data ./program

# 3. 显示报告
echo ""
echo "4. 性能报告摘要..."
echo "----------------------------------------"
sudo perf report -i perf.data --stdio 2>/dev/null | head -50

echo ""
echo "=========================================="
echo "分析完成！"
echo "查看详细报告: sudo perf report -i perf.data"
echo "=========================================="
