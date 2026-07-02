# 01 串行基线 — 暴力搜索 ANN

## 概述

纯 C++ 串行实现，O(n × k × d) 暴力搜索。作为所有并行版本的正确性基准和性能基线。

## 算法

```
对每条查询向量 q:
  1. 计算 q 到所有数据库向量的距离 (内积/L2/余弦)
  2. partial_sort 选出最小的 k 个
  3. 返回 top-k 索引和距离
```

## 编译

```bash
mkdir build && cd build && cmake .. && make
```

## 运行

```bash
./ann_serial --data <data_dir> --queries 100 --topk 10
```

## 预期性能 (DEEP100K, 100K×96维)

| 指标 | 数值 |
|:---|:---|
| 每查询耗时 | ~2846 μs |
| Recall@100 | 1.0000 |
| GFLOPS | ~0.67 |
