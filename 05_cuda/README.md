# 05 GPU 加速 — CUDA ANN 搜索

## 概述

基于 CUDA 的 GPU 加速 ANN 搜索，包含 Flat Tiled GEMM 和 IVF 倒排索引 GPU 实现。

## 三层 GPU 加速

| Level | 实现 | 技术 |
|:---|:---|:---|
| Level 1 | `ann_cuda.cu` | Tiled GEMM + Shared Memory + Warp对齐 |
| Level 2 | `ann_cuda_tiled.cu` | FP16混合精度 + Batch自适应 |
| Level 3 🆕 | `ivf_cuda.cu` | GPU IVF倒排 + 簇感知GEMM |

## 核心优化

- **Shared Memory Tiling**: TILE_M=16, TILE_N=64, TILE_K=32
- **Wavefront-Aware**: TILE_N=64 对齐 warp=32
- **FP16 混合精度**: 带宽减半，理论 2× 加速
- **自适应 Block Size**: 根据维度动态选择 (256→96维)
- **Stream 并发**: 传输-计算重叠

## 编译

```bash
mkdir build && cd build && cmake .. && make
```

## 运行

```bash
# Flat Tiled GEMM
./ann_cuda --data <dir> --queries 100 --batch 64

# Block Size Tuning
./ann_cuda --data <dir> --batch 16,32,64,128,256
```

## 预期性能 (A100/RTX 3090, DEEP100K)

| 模式 | 延迟/查询 | 加速比 vs CPU |
|:---|:---|:---|
| CPU Baseline | ~12,000 μs | 1.00× |
| CUDA Flat GEMM | ~200 μs | 60× |
| CUDA FP16 | ~120 μs | 100× |
