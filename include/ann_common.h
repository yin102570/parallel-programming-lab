/**
 * ann_common.h — ANN 并行系统公共头文件
 *
 * 统一数据结构、高精度计时器、数据I/O、距离函数接口
 * 供 SIMD/OpenMP/MPI/CUDA/Hybrid 各模块共用
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <random>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cassert>

// ============================================================
// 数据类型
// ============================================================
using FloatType = float;
using IntType   = int32_t;

// ============================================================
// 向量数据集
// ============================================================
struct VectorDataset {
    FloatType* data;   // [n × dim] 连续存储
    IntType   n;       // 向量数量
    IntType   dim;     // 向量维度

    VectorDataset() : data(nullptr), n(0), dim(0) {}
    ~VectorDataset() { if (data) free(data); }

    void allocate(IntType n_, IntType dim_) {
        n = n_; dim = dim_;
        data = (FloatType*)aligned_alloc(32, n * dim * sizeof(FloatType));
    }

    FloatType* operator[](IntType i) { return data + i * dim; }
    const FloatType* operator[](IntType i) const { return data + i * dim; }
};

// ============================================================
// 搜索结果
// ============================================================
struct SearchResult {
    IntType*  indices;    // [k]
    FloatType* distances;  // [k]
    IntType   k;

    SearchResult() : indices(nullptr), distances(nullptr), k(0) {}
    ~SearchResult() { if (indices) free(indices); if (distances) free(distances); }

    void allocate(IntType k_) {
        k = k_;
        indices   = (IntType*)malloc(k * sizeof(IntType));
        distances = (FloatType*)malloc(k * sizeof(FloatType));
    }
};

// ============================================================
// 高精度计时器
// ============================================================
class HiResTimer {
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    TimePoint t0;
public:
    void start() { t0 = Clock::now(); }
    double elapsed_us() const {
        auto t1 = Clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    double elapsed_ms() const { return elapsed_us() / 1000.0; }
    double elapsed_s() const { return elapsed_us() / 1e6; }
};

// ============================================================
// 距离函数
// ============================================================

/** L2 距离平方 (不计算sqrt，节省计算) */
inline FloatType l2_squared(const FloatType* a, const FloatType* b, IntType dim) {
    FloatType sum = 0.0f;
    for (IntType i = 0; i < dim; i++) {
        FloatType diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/** 内积距离 (1.0 - dot) */
inline FloatType inner_product_dist(const FloatType* a, const FloatType* b, IntType dim) {
    FloatType dot = 0.0f;
    for (IntType i = 0; i < dim; i++)
        dot += a[i] * b[i];
    return 1.0f - dot;
}

/** 余弦距离 */
inline FloatType cosine_dist(const FloatType* a, const FloatType* b, IntType dim) {
    FloatType dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (IntType i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return 1.0f - dot / (std::sqrt(na) * std::sqrt(nb));
}

// ============================================================
// Top-K 选择
// ============================================================

/** 使用 partial_sort 返回最小的 k 个索引 */
inline void topk_select(const FloatType* distances, IntType n, IntType k,
                         IntType* out_indices, FloatType* out_distances) {
    std::vector<std::pair<FloatType, IntType>> pairs(n);
    for (IntType i = 0; i < n; i++)
        pairs[i] = {distances[i], i};
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end());
    for (IntType i = 0; i < k; i++) {
        out_indices[i]   = pairs[i].second;
        out_distances[i] = pairs[i].first;
    }
}

// ============================================================
// 召回率计算
// ============================================================
inline double compute_recall(const IntType* result, const IntType* ground_truth,
                              IntType nq, IntType k, IntType gt_k) {
    IntType hits = 0;
    IntType total = nq * std::min(k, gt_k);
    for (IntType q = 0; q < nq; q++) {
        for (IntType i = 0; i < std::min(k, gt_k); i++) {
            IntType target = result[q * k + i];
            for (IntType j = 0; j < std::min(k, gt_k); j++) {
                if (ground_truth[q * gt_k + j] == target) {
                    hits++;
                    break;
                }
            }
        }
    }
    return (double)hits / total;
}

// ============================================================
// 数据 I/O
// ============================================================

/** 读取 .fbin 格式文件 (int32 n, int32 dim, float32[n*dim]) */
inline FloatType* load_fbin(const char* path, IntType* out_n, IntType* out_dim) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "[ERROR] Cannot open %s\n", path);
        return nullptr;
    }
    fin.read((char*)out_n, sizeof(IntType));
    fin.read((char*)out_dim, sizeof(IntType));
    size_t total = (size_t)(*out_n) * (*out_dim);
    FloatType* data = (FloatType*)malloc(total * sizeof(FloatType));
    fin.read((char*)data, total * sizeof(FloatType));
    fin.close();
    return data;
}

/** 读取 ground truth .bin 文件 (int32 n, int32 k, int32[n*k]) */
inline IntType* load_gt_bin(const char* path, IntType* out_n, IntType* out_k) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "[ERROR] Cannot open %s\n", path);
        return nullptr;
    }
    fin.read((char*)out_n, sizeof(IntType));
    fin.read((char*)out_k, sizeof(IntType));
    size_t total = (size_t)(*out_n) * (*out_k);
    IntType* data = (IntType*)malloc(total * sizeof(IntType));
    fin.read((char*)data, total * sizeof(IntType));
    fin.close();
    return data;
}

/** 生成随机数据集 */
inline void generate_random_dataset(FloatType* data, IntType n, IntType dim, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<FloatType> dist(-1.0f, 1.0f);
    for (IntType i = 0; i < n * dim; i++)
        data[i] = dist(rng);
}

/** 写入 .fbin 文件 */
inline void save_fbin(const char* path, const FloatType* data, IntType n, IntType dim) {
    std::ofstream fout(path, std::ios::binary);
    fout.write((const char*)&n, sizeof(IntType));
    fout.write((const char*)&dim, sizeof(IntType));
    fout.write((const char*)data, (size_t)n * dim * sizeof(FloatType));
    fout.close();
}
