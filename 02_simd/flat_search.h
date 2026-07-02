/**
 * flat_search.h — 线性扫描基线实现
 *
 * 核心设计思想：
 * 线性扫描是最朴素的ANNS实现，但它建立了所有优化版本的评价基准。
 * 关键洞察：线性扫描的瓶颈不在于浮点运算(ALU)，而在于从DRAM加载向量数据的内存访问。
 * 96维向量 = 384字节，但L1 Cache行只有64字节，意味着每次向量访问都会触发Cache Miss。
 * 这就是"存储墙"问题在ANNS上的具体体现——CPU大部分时间在等待数据到达，而非计算。
 *
 * 性能参考（Cortex-A72 @ 1.5GHz）：
 *   - 单次L2距离计算：约 30-50 ns
 *   - 从DRAM加载384字节：约 100-150 ns
 *   - 结论：计算只占1/3时间，2/3时间在等内存
 */

#ifndef FLAT_SEARCH_H
#define FLAT_SEARCH_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <emmintrin.h>  // SSE2 for comparison

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// 数据结构
//==============================================================================

// 基础向量集结构
typedef struct {
    float* data;       // [n * dim] 连续内存存储
    int n;             // 向量数量
    int dim;           // 维度（本项目固定96）
} VectorDataset;

// 查询结果结构
typedef struct {
    int* indices;      // 最近邻索引 [k]
    float* distances;   // 对应距离值 [k]
    int k;              // 返回近邻数量
} SearchResult;

//==============================================================================
// 核心算法：纯C实现（基线）
//==============================================================================

/**
 * 计算两个向量的L2距离（标量C实现）
 *
 * 为什么不用除法？L2距离比较只需相对顺序，开方是冗余计算
 * 这个细节在10万次调用时会累积出可观的节省
 */
static inline float l2_distance_scalar(const float* a, const float* b, int dim) {
    float dist = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

/**
 * SSE2向量化距离计算（x86基线参考）
 *
 * 关键洞察：SSE2每次处理4个float，理论加速4倍。
 * 但实际只达到2.5-3倍，因为：
 *   1. 水平求和(vhaddps)有依赖链，延迟高
 *   2. 剩余元素(<4个)的串行处理开销
 *   3. 内存带宽仍是瓶颈，SIMD并不能解决"存储墙"
 *
 * 注意：__m128要求16字节对齐，我们用_loadu处理未对齐情况
 */
static inline float l2_distance_sse(const float* a, const float* b, int dim) {
    __m128 sum = _mm_setzero_ps();
    int i = 0;

    // 主循环：4元素并行
    // 注意：SSE的load不要求对齐（_mm_loadu），但对齐的load稍快
    for (; i + 3 < dim; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(va, vb);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    // 水平求和：__m128的4个float累加
    // 手法：(a0+a1) + (a2+a3) 分两步，避免一次水平加的精度问题
    __m128 temp = _mm_movehl_ps(sum, sum);       // temp = [a2,a3,a2,a3]
    sum = _mm_add_ps(sum, temp);                  // sum = [a0+a2, a1+a3, a0+a2, a1+a3]
    temp = _mm_movehdup_ps(sum);                  // temp = [a1+a3, a1+a3, a1+a3, a1+a3]
    sum = _mm_add_ps(sum, temp);                  // sum = [total, total, total, total]

    float result = _mm_cvtss_f32(sum);

    // 剩余元素：dim不总是4的倍数
    for (; i < dim; i++) {
        float d = a[i] - b[i];
        result += d * d;
    }

    return result;
}

/**
 * 线性扫描查找最近邻
 *
 * 朴素实现：O(n)扫描所有向量，保留top-k
 * 优化空间：可引入启发式剪枝（如用三角不等式 early exit）
 *
 * @param dataset  数据库向量集
 * @param query    查询向量 [dim]
 * @param k        返回最近邻数量
 * @return         搜索结果（需调用free_search_result释放）
 */
SearchResult* flat_search(const VectorDataset* dataset, const float* query, int k) {
    SearchResult* result = (SearchResult*)malloc(sizeof(SearchResult));
    result->indices = (int*)malloc(sizeof(int) * k);
    result->distances = (float*)malloc(sizeof(float) * k);
    result->k = k;

    // 初始化为无穷大
    for (int i = 0; i < k; i++) {
        result->indices[i] = -1;
        result->distances[i] = 1e38f;
    }

    // 遍历所有向量，维护top-k小根堆
    // 优化：当前实现的复杂度是O(n*k)，可以优化但此处保持简洁
    for (int i = 0; i < dataset->n; i++) {
        float dist = l2_distance_scalar(dataset->data + i * dataset->dim, query, dataset->dim);

        // 简单线性替换：找到第一个比dist大的位置
        // 适用于k较小场景；k大时应该用堆
        for (int j = 0; j < k; j++) {
            if (dist < result->distances[j]) {
                // 插入位置j，后续元素后移
                for (int m = k - 1; m > j; m--) {
                    result->distances[m] = result->distances[m - 1];
                    result->indices[m] = result->indices[m - 1];
                }
                result->distances[j] = dist;
                result->indices[j] = i;
                break;
            }
        }
    }

    return result;
}

/**
 * 批量查询接口（便于性能测试）
 */
void flat_search_batch(const VectorDataset* dataset, const float* queries,
                       int n_queries, int k, SearchResult* results) {
    for (int q = 0; q < n_queries; q++) {
        SearchResult* r = flat_search(dataset, queries + q * dataset->dim, k);
        results[q] = *r;  // 浅拷贝，指针指向的内存需注意生命周期
        free(r);          // 释放内部结构，但保留结果数据
    }
}

//==============================================================================
// 工具函数
//==============================================================================

/**
 * 从文件加载向量集（二进制格式）
 * 格式：n(int32) + dim(int32) + 数据(float32连续存储)
 */
VectorDataset* load_dataset(const char* filepath) {
    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filepath);
        return NULL;
    }

    int n, dim;
    fread(&n, sizeof(int), 1, fp);
    fread(&dim, sizeof(int), 1, fp);

    VectorDataset* dataset = (VectorDataset*)malloc(sizeof(VectorDataset));
    dataset->n = n;
    dataset->dim = dim;
    dataset->data = (float*)malloc(sizeof(float) * n * dim);
    fread(dataset->data, sizeof(float), (size_t)n * dim, fp);
    fclose(fp);

    return dataset;
}

/**
 * 生成随机向量集（测试用）
 */
VectorDataset* generate_random_dataset(int n, int dim, unsigned int seed) {
    VectorDataset* dataset = (VectorDataset*)malloc(sizeof(VectorDataset));
    dataset->n = n;
    dataset->dim = dim;
    dataset->data = (float*)malloc(sizeof(float) * n * dim);

    srand(seed);
    for (int i = 0; i < n * dim; i++) {
        dataset->data[i] = (float)rand() / RAND_MAX;
    }

    return dataset;
}

void free_dataset(VectorDataset* dataset) {
    if (dataset) {
        free(dataset->data);
        free(dataset);
    }
}

void free_search_result(SearchResult* result) {
    if (result) {
        free(result->indices);
        free(result->distances);
        free(result);
    }
}

#ifdef __cplusplus
}
#endif

#endif // FLAT_SEARCH_H
