# 04 MPI 分布式 — 集群并行 ANN 训练

## 概述

MPI 分布式并行，将 DEEP100K 数据按行分片到 N 个节点，支持同步/异步两种 SGD 训练模式。

## 两种模式

| 模式 | 文件 | 通信方式 | 特点 |
|:---|:---|:---|:---|
| 同步 SGD | `ann_mpi_sync.cpp` | MPI_Allreduce | 一致性强，有同步等待 |
| **异步 SGD** 🆕 | `ann_mpi_async.cpp` | MPI_Isend/Irecv | 无同步瓶颈，stale梯度 |

## 异步 SGD 创新点 🆕

- **参数服务器架构**: Rank 0 为 Server，其余为 Workers
- **非阻塞通信**: `MPI_Isend` 推送梯度，`MPI_Iprobe` 轮询接收
- **Staleness-aware**: 陈旧梯度自动衰减权重
- **理论优势**: 打破 Amdahl 定律串行瓶颈限制

## 编译

```bash
mkdir build && cd build && cmake .. && make
```

## 运行

```bash
# 同步 SGD
mpirun -np 4 ./ann_mpi_sync --dim 96 --epochs 100

# 异步 SGD 🆕
mpirun -np 4 ./ann_mpi_async --dim 96 --iters 100 --batch 64
```

## 通信模型

```
同步 SGD:                     异步 SGD (参数服务器):
  W1 ──┐                        W1 ──→ Server ←── W2
  W2 ──┼─ Allreduce ── W_all          ↓ param    ↓ param
  W3 ──┤                        W1 ←── Server ──→ W2
  W4 ──┘                             (无同步屏障)
  (屏障等待)
```
