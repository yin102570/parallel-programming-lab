#!/bin/bash
# 进入编译目录
cd build || exit
# 1. 统计核心性能指标：周期、指令、缓存缺失
sudo perf stat -e cycles,instructions,cache-misses,L1-dcache-load-misses ./lab
# 2. 记录缓存缺失事件，生成分析报告
sudo perf record -e cache-misses ./lab
# 3. 打开性能分析报告
sudo perf report