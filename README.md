# ANNS Optimization - 高维向量近似最近邻搜索的并行优化

**研究问题**：10万个96维向量的最近邻搜索，在召回率≥0.9的约束下，将延迟从10ms压至0.2ms以内（50倍加速）。

**核心矛盾**：ANNS是典型的**存储墙**问题——CPU算力年增60%，DRAM带宽年增10%，差距由物理定律决定。优化方向不是"算得更快"，而是"少搬数据"和"每次少搬数据"。

---

## 代码结构

```
src/
├── flat_search.h       # 线性扫描基线 + SSE2参考实现
├── simd_distance.h     # NEON/AVX2向量化距离计算（含预取策略）
├── ivf_index.h         # IVF倒排索引 + K-Means++初始化
├── pq_index.h          # PQ乘积量化 + ADC距离计算
├── ivf_pq_index.h      # IVF-PQ集成 + OpenMP并行搜索
├── main.c              # 性能测试框架
└── generate_data.c     # 测试数据生成工具
```

---

## 技术路线：存储墙的三级突破

| 层次 | 优化手段 | 解决的核心问题 |
|------|----------|---------------|
| 第一级 | IVF聚类索引 | 减少访问次数（10万→~4000候选） |
| 第二级 | PQ乘积量化 | 减少单次数据量（384B→8B，压缩比48:1） |
| 第三级 | OpenMP并行 | 多核协同处理候选簇 |

---

## 编译与运行

```bash
# 编译（Linux ARM）
gcc -O2 -march=native -std=c11 -fopenmp -o benchmark \
    src/main.c src/generate_data.c -lm -fopenmp

# 生成测试数据
./generate_data 100000 96 > vectors.bin

# 运行测试
./benchmark vectors.bin --n-queries 1000 --k 100 --n-list 256 --n-probe 10 --threads 4
```

---

## 核心算法原理

### SIMD向量化：`vmlaq_f32` 乘加融合指令
```c
float32x4_t sum = vdupq_n_f32(0);
float32x4_t va = vld1q_f32(a + i);
float32x4_t vb = vld1q_f32(b + i);
float32x4_t diff = vsubq_f32(va, vb);
sum = vmlaq_f32(sum, diff, diff); // FMA: sum += diff*diff
```

### IVF：390个向量=150KB，恰好放进L2 Cache
每簇390向量 × 384字节/向量 ≈ 150KB，而A72 L2为1MB共享——这是n_list=256的参数选择的物理依据。

### PQ：距离计算从96次浮点乘加变成8次查表
96维分成8段，每段12维独立聚类256中心。距离 = Σ LUT[m][code_m]，8次浮点加替代96次乘加，压缩比48:1。

---

## 性能预期

| 版本 | 延迟 | 召回率@100 |
|------|------|-----------|
| Flat（基线） | ~10ms | 1.0000 |
| IVF | ~0.8ms | ~0.9000 |
| IVF-PQ（单线程） | ~0.25ms | ~0.9100 |
| IVF-PQ（4线程） | ~0.08ms | ~0.9100 |
