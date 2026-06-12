/**
 * flat_scan_gpu.h
 * 兼容性包装器 — 保持与原项目接口一致
 * 直接转发到 HIP 实现
 */

#ifndef FLAT_SCAN_GPU_H
#define FLAT_SCAN_GPU_H

#include "flat_scan_gpu_hip.h"

// 别名: 保持原接口命名
typedef GpuBenchResult FlatScanResult;
typedef RunMode        FlatScanMode;

#define FLAT_SCAN_SINGLE   MODE_SINGLE_QUERY
#define FLAT_SCAN_PERSIST  MODE_PERSISTENT
#define FLAT_SCAN_BATCH    MODE_BATCH_GEMM
#define FLAT_SCAN_ROCBLAS  MODE_ROCBLAS
#define FLAT_SCAN_TUNING   MODE_BLOCK_TUNING

inline int flat_scan_gpu(
    const float* base, const float* query,
    int n, int d, int nq, int k,
    int* result, FlatScanMode mode, int batch,
    FlatScanResult* res)
{
    return gpu_flat_search(base, query, n, d, nq, k,
        result, mode, DIST_INNER_PRODUCT, batch, res);
}

#endif // FLAT_SCAN_GPU_H
