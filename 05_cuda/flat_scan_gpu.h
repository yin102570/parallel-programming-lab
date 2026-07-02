/**
 * flat_scan_gpu_hip.h
 * ANN GPU加速 — HIP接口声明
 * 
 * 架构: 3层金字塔优化
 *   Level 0: CPU Baseline (对比基准)
 *   Level 1: 单查询GPU极致优化 (共享内存/自适应Block/Pinned Memory/异步流)
 *   Level 2: Batch矩阵乘法 (Tiled GEMM + wavefront感知)
 *   Level 3: rocBLAS对比
 */

#ifndef FLAT_SCAN_GPU_HIP_H
#define FLAT_SCAN_GPU_HIP_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ==================== 数据结构 ====================

/** 性能统计 */
typedef struct {
    double total_time_us;       // 总耗时 (微秒)
    double compute_time_us;     // 纯计算耗时
    double h2d_time_us;         // Host→Device 传输耗时
    double d2h_time_us;         // Device→Host 传输耗时
    double per_query_us;        // 每条query平均耗时
    double recall_at_k;         // Recall@K
    double qps;                 // Queries Per Second
    double tflops;              // TFLOPS (估算)
    int    num_queries;         // query数量
    int    top_k;               // K值
} GpuBenchResult;

/** 运行模式 */
typedef enum {
    MODE_SINGLE_QUERY = 0,     // 单查询GPU优化
    MODE_PERSISTENT   = 1,     // 持久化内存单查询
    MODE_BATCH_GEMM   = 2,     // Batch矩阵乘法 (推荐)
    MODE_ROCBLAS      = 3,     // rocBLAS对比
    MODE_BLOCK_TUNING = 4      // Block Size调优
} RunMode;

/** 距离度量类型 */
typedef enum {
    DIST_INNER_PRODUCT = 0,    // 内积距离: 1.0 - dot(q,b)
    DIST_L2            = 1     // L2距离
} DistMetric;

// ==================== 核心接口 ====================

/**
 * GPU加速Flat Search (主入口)
 * 
 * @param base_data    base向量 (n × vecdim, 行主序)
 * @param query_data   query向量 (num_queries × vecdim)
 * @param n             base向量数量
 * @param vecdim       向量维度
 * @param num_queries  query数量
 * @param top_k        返回top-k
 * @param result_indices  输出: 结果索引 (num_queries × top_k)
 * @param mode         运行模式
 * @param metric       距离度量
 * @param batch_size   Batch大小 (仅MODE_BATCH_GEMM有效, 0=auto)
 * @param result       输出: 性能统计
 * @return 0=成功
 */
int gpu_flat_search(
    const float* base_data,
    const float* query_data,
    int n,
    int vecdim,
    int num_queries,
    int top_k,
    int* result_indices,
    RunMode mode,
    DistMetric metric,
    int batch_size,
    GpuBenchResult* result
);

/**
 * 单查询模式 — 每条query依次在GPU执行
 * Level 1优化: shared memory缓存query + 自适应block size + pinned memory + 异步流
 */
int gpu_single_query_search(
    const float* d_base,      // GPU端base (已预加载)
    const float* h_query,     // CPU端query (pinned memory)
    int n, int vecdim,
    int num_queries, int top_k,
    int* h_result,            // CPU端结果
    GpuBenchResult* result
);

/**
 * Batch矩阵乘法模式 — 核心创新
 * Level 2优化: Tiled GEMM + wavefront感知 + 查询间并行
 */
int gpu_batch_gemm_search(
    const float* d_base,      // GPU端base (持久化)
    const float* h_query,     // CPU端query (pinned memory)
    int n, int vecdim,
    int num_queries, int top_k,
    int batch_size,            // 每批处理query数
    int* h_result,
    GpuBenchResult* result
);

/**
 * rocBLAS SGEMM 对比模式
 * 使用AMD官方rocBLAS库做baseline对比
 */
int gpu_rocblas_search(
    const float* d_base,
    const float* h_query,
    int n, int vecdim,
    int num_queries, int top_k,
    int batch_size,
    int* h_result,
    GpuBenchResult* result
);

/**
 * Block Size调优模式
 * 测试不同block size对性能的影响
 */
int gpu_block_tuning(
    const float* d_base,
    const float* h_query,
    int n, int vecdim,
    int num_queries, int top_k,
    int* h_result
);

// ==================== 工具函数 ====================

/** 加载 .fbin 格式数据集 */
float* load_fbin_file(const char* filename, int* n_out, int* d_out);

/** 加载 .gt.bin 格式 ground truth */
int* load_gt_bin_file(const char* filename, int* nq_out, int* k_out);

/** 计算 Recall@K */
double compute_recall(
    const int* result_ids,    // 算法结果 (nq × K)
    const int* gt_ids,        // ground truth (nq × gt_K)
    int nq, int K, int gt_K
);

/** 分配pinned (page-locked) 内存 */
float* alloc_pinned(int num_elements);
void   free_pinned(float* ptr);

/** 获取自适应block size (基于vec_dim) */
int get_adaptive_block_size(int vecdim);

/** HIP错误检查宏 */
#define HIP_CHECK(cmd) do { \
    hipError_t _err = cmd; \
    if (_err != hipSuccess) { \
        fprintf(stderr, "HIP Error [%s]: %s at %s:%d\n", \
            hipGetErrorName(_err), hipGetErrorString(_err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/** 安全的cuda/hip内存分配 */
#define GPU_MALLOC(ptr, size) HIP_CHECK(hipMalloc((void**)(ptr), (size)))
#define GPU_FREE(ptr)         HIP_CHECK(hipFree(ptr))

#ifdef __cplusplus
}
#endif

#endif // FLAT_SCAN_GPU_HIP_H
