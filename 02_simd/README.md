# 02 SIMD 向量化 — 数据级并行 ANN

## 概述

利用 SSE2/SSE4.1/AVX2/NEON 等 SIMD 指令集，在单个 CPU 核心内实现数据级并行距离计算。包含软件预取优化。

## 核心优化

| 技术 | 说明 |
|:---|:---|
| AVX2 向量化 | 256-bit 一次处理 8 个 float |
| SSE2/SSE4.1 | 128-bit 兼容方案 |
| FMA 指令 | `_mm256_fmadd_ps` 融合乘加 |
| 软件预取 | `__builtin_prefetch` 减少 Cache Miss |
| 平台自适应 | 编译期检测，自动选择最优实现 |

## 距离函数

- `l2_distance_scalar()` — 标量基线
- `l2_distance_sse()` — SSE2 (128-bit, 4-way)
- `l2_distance_avx2()` — AVX2 (256-bit, 8-way)
- `l2_distance_neon()` — ARM NEON
- `l2_distance_auto()` — 自动选择

## 编译

```bash
mkdir build && cd build && cmake .. && make
```

## 预期加速

| 模式 | 预期加速比 |
|:---|:---|
| SSE2 vs Scalar | ~2.0× |
| AVX2 vs Scalar | ~3.3× |
