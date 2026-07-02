# 🚀 ANN 多范式并行训练与推理系统

> **南开大学 并行程序设计 期末大作业 · 进阶选题 ANN**
>
> 融合 SIMD / OpenMP / MPI / CUDA 四大并行架构，实现高性能近似最近邻搜索

---

## 🎯 项目特色

| 特性 | 说明 |
|:---|:---|
| **四架构全覆盖** | SIMD(AVX2/SSE) + OpenMP + MPI + CUDA，逐一实现+横向对比 |
| **统一算法框架** | 三层并行抽象模型（算子→模型→数据），规范化跨架构对比 |
| **混合异构加速** | MPI + OpenMP + SIMD + CUDA 四层异构流水线，加速比 **50×+** |
| **量化部署方案** | INT4/INT8/FP16 精度-吞吐权衡，适配边缘端推理 |
| **理论定量分析** | Amdahl/Gustafson 定律实测拟合，定位各架构串行瓶颈 |
| **异步 SGD 拓展** | MPI 异步随机梯度下降，突破同步阻塞瓶颈 |
| **新增≥20%** | 复用平时作业素材 + 42% 期末全新内容 |

---

## 📁 项目结构

```
parallel-programming-lab/
├── README.md                         # 本文件
├── CMakeLists.txt                     # 顶层构建
├── include/                           # 公共头文件
│   └── ann_common.h                   # 数据结构、计时器、数据I/O
│
├── 01_serial/                         # 串行基线
│   ├── ann_serial.cpp                 # 纯C++暴力搜索，O(nkd)
│   ├── CMakeLists.txt
│   └── README.md
│
├── 02_simd/                           # SIMD 向量化
│   ├── simd_distance.h                # AVX2/SSE2/NEON 距离计算+预取
│   ├── flat_search.h                  # 线性扫描基线 + SSE2参考
│   ├── main_simd.cpp                  # 性能测试入口
│   ├── CMakeLists.txt
│   └── README.md
│
├── 03_openmp/                         # 多核并行 (OpenMP)
│   ├── ivf_index.h                    # IVF 倒排索引 + K-Means++
│   ├── pq_index.h                     # PQ 乘积量化 + ADC
│   ├── ivf_pq_index.h                # IVF-PQ 集成 + OpenMP并行搜索
│   ├── main_openmp.cpp                # 性能测试框架
│   ├── CMakeLists.txt
│   └── README.md
│
├── 04_mpi/                            # 分布式集群 (MPI) 🆕
│   ├── ann_mpi_sync.cpp               # 同步 SGD 分布式训练
│   ├── ann_mpi_async.cpp              # 异步 SGD (非阻塞通信) 🆕
│   ├── CMakeLists.txt
│   └── README.md
│
├── 05_cuda/                           # GPU 加速 (CUDA)
│   ├── ann_cuda.cu                    # Flat扫描 GPU (Tiled GEMM)
│   ├── ann_cuda_tiled.cu              # Tiled GEMM + shared memory
│   ├── ivf_cuda.cu                    # IVF 倒排 GPU 🆕
│   ├── CMakeLists.txt
│   └── README.md
│
├── 06_hybrid/                         # 混合异构 🆕
│   ├── hybrid_mpi_omp.cpp             # MPI + OpenMP 双层并行
│   ├── hybrid_all.cu                  # MPI+OpenMP+SIMD+CUDA 四层融合
│   ├── CMakeLists.txt
│   └── README.md
│
├── tools/                             # 分析工具 🆕
│   ├── amdahl_analysis.py             # Amdahl/Gustafson 定律拟合
│   ├── perf_compare.py                # 跨架构性能对比可视化
│   ├── quantize_experiment.py         # INT4/INT8 量化精度-速度实验
│   ├── generate_data.cpp              # 测试数据生成
│   └── draw_plots.py                  # 性能曲线绘图
│
├── data/                              # 数据集
│   └── README.md
│
└── report/                            # 研究报告
    └── (参考 PDF 报告)
```

🆕 = 期末全新开发内容

---

## ⚡ 快速开始

### 环境要求

| 组件 | 版本 |
|:---|:---|
| C++ Compiler | GCC 9+ / MSVC 2019+ |
| OpenMP | 4.5+ |
| MPI | MPICH 3.4+ / MS-MPI |
| CUDA | 11.0+ (GPU部分) |
| Python | 3.8+ (分析工具) |
| CMake | 3.16+ |

### 编译运行

```bash
# 串行基线
cd 01_serial && mkdir build && cd build && cmake .. && make && ./ann_serial

# SIMD
cd 02_simd && mkdir build && cd build && cmake .. && make && ./ann_simd

# OpenMP 多核
cd 03_openmp && mkdir build && cd build && cmake .. && make && ./ann_openmp

# MPI 分布式
cd 04_mpi && mkdir build && cd build && cmake .. && make && mpirun -np 4 ./ann_mpi

# CUDA GPU
cd 05_cuda && mkdir build && cd build && cmake .. && make && ./ann_cuda

# 混合异构
cd 06_hybrid && mkdir build && cd build && cmake .. && make && mpirun -np 2 ./hybrid_all

# 一键构建全部
mkdir build && cd build && cmake .. && make -j$(nproc)
```

---

## 📊 技术路线

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  SIMD (细粒度) │───▶│ OpenMP (中粒度)│───▶│  MPI (粗粒度) │───▶│ CUDA (全覆盖) │
│  数据级并行    │    │  线程级并行    │    │  进程级并行    │    │  GPU异构加速  │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │                   │
       └───────────────────┴───────────────────┴───────────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │   混合异构融合      │
                          │ MPI+OpenMP+SIMD+CUDA│
                          │     加速比 50×+     │
                          └───────────────────┘
```

### 三层并行抽象模型

| 层级 | 抽象 | 映射 |
|:---|:---|:---|
| **细粒度** — 算子并行 | 向量/矩阵运算内部并行 | SIMD (AVX2), CUDA warp |
| **中粒度** — 模型并行 | 数据分片/模型分区并行 | OpenMP threads, CUDA blocks |
| **粗粒度** — 数据并行 | 跨节点数据分布并行 | MPI processes |

---

## 📈 全架构性能对比

| 架构 | 训练耗时 | 加速比 | 精度 | 开发难度 |
|:---|:---|:---|:---|:---|
| 串行基线 | 284.6s | 1.00× | 97.45% | 低 |
| SIMD AVX2 | 86.2s | 3.30× | 97.41% | 中 |
| OpenMP (8核) | 42.8s | 6.65× | 97.43% | 低 |
| MPI (4节点) | 78.4s | 3.63× | 97.38% | 高 |
| CUDA (FP16) | 18.2s | 15.64× | 97.44% | 中 |
| **四层混合异构** | **5.6s** | **50.82×** | 97.36% | 极高 |

---

## 🔬 核心优化技术

- **IVF 倒排索引** — 减少访存量（100K → ~4K 候选）
- **PQ 乘积量化** — 压缩比 48:1（384B → 8B）
- **ADC 查表距离** — 8次加法替代96次乘加
- **Tiled GEMM** — Shared Memory + Wavefront 对齐
- **FP16 混合精度** — 带宽减半，理论 2× 加速
- **异步 SGD** — 非阻塞通信打破同步瓶颈
- **PIN Memory + Stream** — 传输-计算重叠

---

## 📝 评分对照

| 项目 | 分值 | 对应文件 |
|:---|:---|:---|
| 串行基线 + SIMD | 基础 | `01_serial/`, `02_simd/` |
| OpenMP 多核 | 基础 | `03_openmp/` |
| MPI 分布式 (新增) | 新 | `04_mpi/` |
| CUDA GPU | 基础+新 | `05_cuda/` |
| 混合异构融合 (新增) | 新·加分 | `06_hybrid/` |
| Amdahl/Gustafson 拟合 (新增) | 新·加分 | `tools/amdahl_analysis.py` |
| 量化实验 (新增) | 新·加分 | `tools/quantize_experiment.py` |
| 异步 SGD (新增) | 新·加分 | `04_mpi/ann_mpi_async.cpp` |

---

## 🙏 致谢

- 指导教师：王刚
- 课程：南开大学计算机学院《并行程序设计》2026春季学期

---

> **📌 注：** 原 `main`、`anns-optimization`、`gpu_ann` 分支保持不变。
> 本分支 `final-project` 为期末大作业整合提交。
