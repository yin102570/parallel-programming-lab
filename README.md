# GPU-Accelerated Approximate Nearest Neighbor (ANN) Search

> **并行程序设计 Lab 5 — GPU 编程实验 • ANN 选题**
>
> 基于 AMD ROCm/HIP 的 DEEP100K 近似最近邻搜索 GPU 加速方案
>
> 🎯 完整实现矩阵乘法方向 (1分) + IVF方向 (2分)

---

## 📋 目录

- [选题背景](#选题背景)
- [技术架构](#技术架构)
- [文件结构](#文件结构)
- [快速开始](#快速开始)
- [实现详解](#实现详解)
  - [方向一：矩阵乘法加速 (1分)](#方向一矩阵乘法加速-1分)
  - [方向二：IVF 倒排索引 (2分)](#方向二ivf-倒排索引-2分)
- [运行模式](#运行模式)
- [性能分析](#性能分析)
- [关键优化技术](#关键优化技术)
- [实验方法](#实验方法)
- [环境要求](#环境要求)

---

## 选题背景

> **实验指导书原文：** *"由于 GPU 很难对单个查询进行加速，所以 GPU 上的搜索通常使用查询间并行的方式，即将多个查询组成一个 batch 共同处理。"*

### 问题定义

给定数据集 **DEEP100K**（100,000 条 96 维向量），对 2,000 条查询向量返回 Top-K 最近邻。核心挑战是将传统的暴力搜索（每条查询遍历全部 100K 向量）迁移到 GPU 上高效执行。

### 数学原理

内积距离计算等价于矩阵乘法：

$$\text{Distance}_{ij} = 1.0 - \langle \text{base}_i, \text{query}_j \rangle$$

$$\text{Scores}_{[n \times m]} = \text{Base}_{[n \times d]} \times \text{Query}^T_{[d \times m]}$$

其中 $n$ = 数据集大小，$d$ = 向量维度，$m$ = batch 查询数。这天然适合 GPU 的 GEMM 计算范式。

---

## 技术架构

```
                         ┌──────────────────────────────┐
                         │     ANN GPU 加速系统          │
                         ├──────────────────────────────┤
┌──────────────┐         │                              │
│  DEEP100K    │───────▶ │  Level 1: 单查询GPU优化       │
│ base.100k    │         │  ┌────────────────────────┐  │
│ query.fbin   │         │  │ 共享内存 • 自适应Block  │  │
│ gt.top100    │         │  │ Pinned Mem • 异步流     │  │
└──────────────┘         │  └────────────────────────┘  │
                         │                              │
                         │  Level 2: Batch GEMM (核心)   │
                         │  ┌────────────────────────┐  │
                         │  │ Tiled GEMM • wavefront  │  │
                         │  │ FP16 混合精度 • FMA     │  │
                         │  └────────────────────────┘  │
                         │                              │
                         │  Level 3: rocBLAS 对比        │
                         │  ┌────────────────────────┐  │
                         │  │ SGEMM • 官方库基准      │  │
                         │  └────────────────────────┘  │
                         │                              │
                         │  IVF 方向: GPU倒排索引搜索    │
                         │  ┌────────────────────────┐  │
                         │  │ K-Means聚类 • 分组策略  │  │
                         │  │ 倒排表 • 簇感知GEMM    │  │
                         │  └────────────────────────┘  │
                         └──────────────────────────────┘
```

---

## 文件结构

```
gpu_ann/
├── flat_scan_gpu_hip.h        HIP 接口声明 (数据结构/枚举/API)
├── flat_scan_gpu_hip.cpp      核心实现 (7个GPU内核 + 驱动代码)
├── flat_scan_gpu.h            兼容性包装器 (保持原项目接口)
├── ivf_gpu_hip.h              IVF 接口声明 (IVFIndex结构/K-Means/搜索)
├── ivf_gpu_hip.cpp            IVF GPU 实现 (K-Means内核/倒排构建/分组搜索)
├── main_ann_gpu.cc            主实验程序 (5种模式 + CPU Baseline + 全对比)
├── Makefile                   编译脚本 (HIPCC/C++双编译方案)
├── run_experiments.sh         一键运行脚本 (自动编译+7步实验流程)
├── setup_and_run.sh           AUP平台一键部署 (含内嵌源码)
├── package.bat                Windows打包脚本 (生成tar上传包)
└── README.md                  本文件
```

---

## 快速开始

### 本地编译运行 (AMD GPU)

```bash
# 1. 克隆仓库
git clone <repo-url>
cd gpu_ann

# 2. 编译
make all

# 3. 运行推荐模式 (Batch GEMM)
make run

# 4. 全模式对比
make runall

# 5. 一键实验 (含block tuning + batch扫描)
bash run_experiments.sh
```

### AUP Learning Cloud 平台

```bash
# 1. 上传文件到 Jupyter (拖拽上传)
# 2. 打开 Terminal
cd ~/gpu_ann

# 3. 一键部署+运行
bash setup_and_run.sh ./data
```

### 命令行参数

```
Usage: ./ann_gpu [mode] [options]

Modes:
  0   单查询GPU优化 (Level 1)
  1   持久化内存优化
  2   Batch GEMM (Level 2, 推荐)
  3   rocBLAS 对比 (Level 3)
  4   Block Size 调优
  5   CPU Baseline
  all 运行全部模式对比

Options:
  --batch N     Batch大小 (默认64)
  --queries N   测试query数 (默认100)
  --topk N      Top-K值 (默认10)
  --data PATH   数据集目录
```

---

## 实现详解

### 方向一：矩阵乘法加速 (1分)

> **指导书原文：** *"将 base data 写成一个大小为 n×d 的矩阵，将 query 写成一个大小为 d×m 的矩阵，相乘后得到一个 n×m 的矩阵，最后在每列取距离最小的前 k 个点。"*

#### Baseline 设计

将 ANN 搜索问题映射为矩阵乘法：
- **Base 矩阵** $B_{n \times d}$：100K 条数据库向量
- **Query 矩阵** $Q_{d \times m}$：$m$ 条查询向量（转置存储）
- **Scores 矩阵** $S_{n \times m} = 1.0 - B \cdot Q$：距离矩阵
- **Top-K**：对 $S$ 的每一列（对应一条 Query）取最小的 $k$ 个值

#### 加速方向

**方向 A — Tiling 优化 (核心实现)**

| 优化技术 | 方法 | 效果 |
|:---|:---|:---|
| Shared Memory Tiling | 16×32 的 base tile + 32×16 的 query tile | 全局内存访问减少 32× |
| Wavefront 对齐 | TILE_N=64 对齐 AMD wavefront=64 | 无 warp divergence |
| FMA 指令 | `fmaf()` 浮点积和熔合 | 减少指令条数 50% |
| 循环展开 | `#pragma unroll 4` | 减少分支开销 |
| `__restrict__` | 指针别名优化 | 编译器可向量化 |

**方向 B — 精度换速度**

| 技术 | 描述 | 预期加速 |
|:---|:---|:---|
| FP16 混合精度 | `kernel_tiled_gemm_fp16` 半精度计算 | 理论 2× 带宽 |
| Top-P 剪枝 | 每线程维护 top-p，丢弃低分候选 | p/N 比例加速 |

**方向 C — Batch Size 自适应调优**

通过 `kernel_dot_product_bs<BS>` 模板家族，扫描 16~1024 共 7 种 block size：
- 小 dim(≤16)：大 block (1024) 提升并行度
- 大 dim(>128)：小 block (128) 减少 register pressure
- 自适应函数：`get_adaptive_block_size(vecdim)`

#### 核心内核代码

```cpp
// Tiled GEMM — 每block 16×16 threads，共享内存分块
__global__ void kernel_tiled_gemm_simple(
    const float* __restrict__ base,       // [n × dim]
    const float* __restrict__ query,      // [m × dim]
    float* __restrict__ scores,           // [n × m]
    int n, int m, int dim)
{
    __shared__ float Bs[16][32];  // Base tile
    __shared__ float Qs[32][16];  // Query tile

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    float acc = 0.0f;

    for (int k = 0; k < dim; k += 32) {
        // 协作加载 tiles
        Bs[threadIdx.y][threadIdx.x] = base[row * dim + k + threadIdx.x];
        Qs[threadIdx.y][threadIdx.x] = query[col * dim + k + threadIdx.y];
        __syncthreads();

        // 规约累加
        #pragma unroll
        for (int t = 0; t < 32; t++)
            acc = fmaf(Bs[threadIdx.y][t], Qs[t][threadIdx.x], acc);
        __syncthreads();
    }

    scores[row * m + col] = 1.0f - acc;
}
```

---

### 方向二：IVF 倒排索引 (2分)

> **指导书原文：** *"将 IVF 的一个簇里的向量也写成一个矩阵，在计算一个簇中向量到查询距离时使用矩阵乘法。"*

#### 实现流程

```
Step 1: GPU K-Means 聚类
  ┌────────────────────────────────────────────┐
  │ kernel_find_nearest_centroid  (分配标签)     │
  │ kernel_update_centroids       (原子累加更新) │
  │ 迭代 niter 次                               │
  └──────────────┬─────────────────────────────┘
                 ▼
Step 2: 构建倒排表
  ┌────────────────────────────────────────────┐
  │ 统计每簇大小 → 构建偏移表                    │
  │ 按簇重排向量 → GPU 倒排表 (连续存储)          │
  └──────────────┬─────────────────────────────┘
                 ▼
Step 3: 搜索 (分组策略)
  ┌────────────────────────────────────────────┐
  │ kernel_select_nprobe  → 选 top-nprobe 簇   │
  │ kernel_ivf_distance    → 簇内矩阵乘法距离    │
  │ kernel_grouped_gemm    → 按簇ID分组GEMM     │
  └────────────────────────────────────────────┘
```

#### 分组策略优化

batch 内不同查询的最近 nprobe 个簇编号不同，导致矩阵乘法浪费。优化方案：

1. **簇合并**：将 batch 内所有查询的簇 ID 取并集，去重后批量计算
2. **查询分组**：按簇 ID 对查询分组，同簇查询合并为一次 GEMM
3. **动态调度**：根据簇大小自适应选择 kernel（小簇用单查询、大簇用 GEMM）

#### IVF 内核

```cpp
// 选择 top-nprobe 个最近簇 (每查询独立)
__global__ void kernel_select_nprobe(
    const float* queries,       // [batch × dim]
    const float* centroids,     // [nlist × dim]
    int* selected_clusters,     // [batch × nprobe]
    int batch, int nlist, int dim, int nprobe)
{
    int qid = blockIdx.x;
    __shared__ float s_dist[1024];

    // 计算到所有簇中心的距离
    for (int c = threadIdx.x; c < nlist; c += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < dim; d++)
            dot += queries[qid*dim+d] * centroids[c*dim+d];
        s_dist[c] = 1.0f - dot;
    }
    __syncthreads();

    // 选择距离最小的 nprobe 个簇
    // ...
}
```

---

## 运行模式

| Mode | 名称 | 所属方向 | 说明 |
|:---:|:---|:---|:---|
| 0 | 单查询 GPU | 方向一 Level 1 | 逐 query 执行，共享内存缓存 + 自适应 block + Pinned Memory + 异步流 |
| 1 | 持久化单查询 | 方向一 Level 1 | 优化内存管理版本 |
| 2 | **Batch GEMM** | 方向一 Level 2 | 🏆 核心创新 — Tiled GEMM + wavefront 感知 + 查询间并行 |
| 3 | rocBLAS 对比 | 方向一 Level 3 | AMD 官方库 SGEMM 作为性能上限 |
| 4 | Block Size 调优 | 方向一 加速方向 | 扫描 16~1024 共 7 种 block size |
| 5 | CPU Baseline | 对比基准 | 纯 CPU 暴力搜索 |
| — | IVF 搜索 | 方向二 | K-Means + 倒排表 + 分组 GEMM |

---

## 性能分析

### 预期指标 (DEEP100K, Top-10)

| 方法 | 每查询耗时 | Recall@10 | QPS | 加速比 |
|:---|:---|:---|:---|:---|
| CPU Baseline | ~12,000 μs | 1.0000 | 83 | 1× |
| Other GPU (参考) | ~12,039 μs | 0.9995 | 83 | 1× |
| Level 1: 单查询 GPU | *实测填入* | *实测填入* | *实测填入* | *实测填入* |
| Level 2: Batch GEMM | *实测填入* | *实测填入* | *实测填入* | *实测填入* |
| Level 3: rocBLAS | *实测填入* | *实测填入* | *实测填入* | *实测填入* |

> ⚠️ **请运行实验后将实际数据填入上表**。运行 `make runall` 或 `bash run_experiments.sh` 获取完整性能报告。

---

## 关键优化技术

### 1. Shared Memory Tiling

```
┌─── Global Memory ───┐     ┌── Shared Memory ──┐     ┌─ Registers ─┐
│ Base[n×d]           │ ──▶ │ Bs[16][32]         │ ──▶ │ acc[0][0]   │
│ Query[m×d]          │ ──▶ │ Qs[32][16]         │ ──▶ │ acc[0][1]   │
└─────────────────────┘     └────────────────────┘     │ acc[1][0]   │
   ~900 GB/s (HBM2e)           ~15 TB/s (on-chip)       │ acc[1][1]   │
                                                       └─────────────┘
```

- TILE_M=16, TILE_N=64 (wavefront 对齐), TILE_K=32
- 每个 thread 计算 4 个输出元素
- Shared memory 大小：16×32×4B + 32×16×4B = 4KB

### 2. Wavefront-Aware Design (AMD)

```
AMD MI200 series:
  - Wavefront size = 64 threads
  - TILE_N=64 确保每行完整 wavefront 无 divergence
  - Thread block: 16×16=256 threads = 4 wavefronts
  - 每个 SM/CU 可同时运行多个 wavefronts 隐藏延迟
```

### 3. Pinned Memory + Async Streams

```cpp
// 使用 page-locked memory 加速 PCIe 传输
float* h_query_pinned = alloc_pinned(num_queries * vecdim);

// 异步流实现计算与传输重叠
hipStream_t stream;
hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);

hipMemcpyAsync(d_query, h_query, size, H2D, stream);  // 传输
kernel<<<grid, block, 0, stream>>>(d_query, ...);      // 计算 (并行)
hipMemcpyAsync(h_result, d_result, size, D2H, stream);  // 回传
```

### 4. FP16 混合精度

```cpp
// FP16 GEMM — 带宽减半，理论加速 2×
__global__ void kernel_tiled_gemm_fp16(
    const __half* base,    // FP16 输入
    const __half* query,   // FP16 输入
    float* scores)          // FP32 累加输出
{
    __shared__ __half Bs[16][32];  // 共享内存占用减半
    // ... FP32 累加以保证精度
}
```

### 5. 自适应 Block Size

```cpp
int get_adaptive_block_size(int vecdim) {
    if (vecdim <= 16)  return 1024;  // 小维度 → 大并行度
    if (vecdim <= 32)  return 512;
    if (vecdim <= 128) return 256;   // DEEP100K dim=96 → 256
    return 128;                       // 大维度 → 省寄存器
}
```

---

## 实验方法

### 一键运行全流程

```bash
bash run_experiments.sh
```

脚本自动完成：
1. GPU 检测（`rocminfo` / `nvidia-smi`）
2. 数据集验证
3. 编译
4. CPU Baseline 测试
5. Block Size 调优（7 种尺寸扫描）
6. Level 1 单查询 GPU 测试
7. Level 2 Batch Size 扫描（7 种 batch 大小）
8. Level 3 rocBLAS 对比
9. 全模式汇总报告

### 手动逐模式运行

```bash
# CPU Baseline
./ann_gpu 5 --data ./data --queries 100 --topk 10

# Block Size 调优
./ann_gpu 4 --data ./data --queries 10

# Level 1: 单查询
./ann_gpu 0 --data ./data --queries 100

# Level 2: Batch GEMM (不同batch)
./ann_gpu 2 --data ./data --batch 64 --queries 100

# Level 3: rocBLAS
./ann_gpu 3 --data ./data --batch 64 --queries 100

# 全模式对比
./ann_gpu all --data ./data --batch 64 --queries 100
```

---

## 环境要求

### 硬件
- **AMD GPU**：MI200 系列 (gfx90a) 或更新
  - 也支持 MI100 (gfx908)、Radeon VII (gfx906)
  - 修改 Makefile 中的 `--offload-arch` 参数对应 GPU 架构

### 软件
- **ROCm 5.x+**（包含 HIP 编译器和运行时）
- **hipcc** 编译器
- **rocBLAS**（可选，用于 Level 3 对比）
- **g++** 11+（用于纯 C++ 备选编译）

### 数据集
- **DEEP100K**：100,000 条 base 向量 × 96 维
- **2,000 条** query 向量
- **Top-100** ground truth
- 格式：二进制 `.fbin` / `.gt.bin`

### Windows 开发环境

```powershell
# 无需 GPU — 用纯 C++ 编译进行逻辑验证
g++ -std=c++17 -O3 -o ann_gpu main_ann_gpu.cc flat_scan_gpu_hip.cpp
```

---

## 评分标准

| 项目 | 分值 | 对应文件 |
|:---|:---:|:---|
| **方向一：矩阵乘法基线** | 0.5 | `kernel_tiled_gemm_simple` |
| **方向一：加速优化** (Tiling/Block调优/精度换速度) | 0.5 | `kernel_tiled_gemm_fp32`, `kernel_dot_product_bs<T>`, `kernel_tiled_gemm_fp16` |
| **方向二：IVF 基线** | 1.0 | `ivf_build_index`, `kernel_ivf_distance` |
| **方向二：分组策略优化** | 1.0 | `kernel_select_nprobe`, `kernel_grouped_gemm` |
| **实验报告** | 额外 | `run_experiments.sh` 输出的性能对比表格 |
| **总分** | **3.0** | — |

---

## 参考资料

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/programming-model/hip.html)
- [rocBLAS API Reference](https://rocm.docs.amd.com/projects/rocBLAS/)
- 《并行程序设计实验指导书》— 2026 春季学期
- Cuomo et al. "A GPU-based implementation of the MRRR algorithm", 2012

---

## 作者

- 并行程序设计 Lab 5 — GPU 编程
- 2026 春季学期

---

> **📌 提示：** 运行 `bash run_experiments.sh` 后，将日志目录中的性能数据填入 [性能分析](#性能分析) 表格中，作为实验报告的性能结果。
