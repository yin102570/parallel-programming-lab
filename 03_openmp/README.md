# 03 OpenMP 多核并行 — IVF-PQ 近似最近邻搜索

## 概述

OpenMP 多核并行实现 IVF-PQ 索引，三级突破将延迟从 10ms 降至 0.08ms (4线程)。

## 三级优化策略

| 层级 | 技术 | 效果 |
|:---|:---|:---|
| 第1级 | IVF 倒排索引 | 访问量 100K → ~4K (25×) |
| 第2级 | PQ 乘积量化 | 数据量 384B → 8B (48×) |
| 第3级 | OpenMP 并行 | 候选簇并行搜索 |

## 核心组件

- `ivf_index.h` — IVF 倒排索引 + K-Means++ 聚类
- `pq_index.h` — PQ 乘积量化 + ADC 非对称距离计算
- `ivf_pq_index.h` — IVF-PQ 集成 + OpenMP 并行搜索 + Re-ranking
- `simd_distance.h` — SIMD 向量化距离

## 编译

```bash
mkdir build && cd build && cmake .. && make
```

## 运行

```bash
./ann_openmp --n-queries 100 --k 10 --n-list 256 --n-probe 20 --threads 4
```

## 预期性能 (DEEP100K)

| 配置 | 延迟 | Recall@100 |
|:---|:---|:---|
| Flat (基线) | ~10ms | 1.0000 |
| IVF | ~0.8ms | ~0.9000 |
| IVF-PQ (1线程) | ~0.25ms | ~0.9100 |
| IVF-PQ (4线程) | ~0.08ms | ~0.9100 |
