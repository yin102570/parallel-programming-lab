/**
 * main_simd.cpp — SIMD 向量化 ANN 性能测试
 *
 * 测试 SSE2 和 AVX2 向量化距离计算在 ANN 搜索中的加速效果。
 * 包含 Flat 扫描基线、SSE2 向量化、AVX2 向量化三种模式对比。
 */

#include "ann_common.h"
#include "simd_distance.h"
#include "flat_search.h"
#include <omp.h>

// 使用 flat_search.h 中的数据结构和函数
// (注意: flat_search.h 自带 SIMD 距离函数)

static void simd_benchmark_flat(VectorDataset& base, VectorDataset& query,
                                 IntType k, IntType* h_gt, IntType gt_k) {
    printf("\n=== SIMD Flat Search Benchmark ===\n");
    printf("  Mode              | PerQuery(us) | Recall@%d | Speedup\n", k);
    printf("  -------------------|--------------|-----------|--------\n");

    // 1. Scalar baseline
    {
        SearchResult* results = new SearchResult[query.n];
        for (int i = 0; i < query.n; i++) results[i].allocate(k);
        double avg_us = 0;
        HiResTimer t; t.start();
        for (IntType q = 0; q < query.n; q++) {
            FloatType* dists = new FloatType[base.n];
            for (IntType i = 0; i < base.n; i++)
                dists[i] = l2_squared(query[q], base[i], base.dim);
            topk_select(dists, base.n, k, results[q].indices, results[q].distances);
            delete[] dists;
        }
        avg_us = t.elapsed_us() / query.n;
        printf("  Scalar            | %12.1f | -        | 1.00x\n", avg_us);
        delete[] results;
    }

    // 2. SSE2 (if available)
#ifdef __SSE2__
    {
        double avg_us = 0;
        HiResTimer t; t.start();
        float* distances = (float*)malloc(base.n * sizeof(float));
        int* result_idx = (int*)malloc(query.n * k * sizeof(int));
        for (IntType q = 0; q < query.n; q++) {
            for (IntType i = 0; i < base.n; i++) {
                distances[i] = l2_distance_sse(query[q], base[i], base.dim);
            }
            topk_select(distances, base.n, k, &result_idx[q*k], (float*)malloc(k*sizeof(float)));
        }
        avg_us = t.elapsed_us() / query.n;
        free(distances); free(result_idx);
        printf("  SSE2              | %12.1f | -        | -\n", avg_us);
    }
#endif

    // 3. AVX2 (if available)
#ifdef __AVX2__
    {
        double avg_us = 0;
        HiResTimer t; t.start();
        float* distances = (float*)malloc(base.n * sizeof(float));
        int* result_idx = (int*)malloc(query.n * k * sizeof(int));
        for (IntType q = 0; q < query.n; q++) {
            for (IntType i = 0; i < base.n; i++) {
                distances[i] = l2_distance_avx2(query[q], base[i], base.dim);
            }
            topk_select(distances, base.n, k, &result_idx[q*k], (float*)malloc(k*sizeof(float)));
        }
        avg_us = t.elapsed_us() / query.n;
        free(distances); free(result_idx);
        printf("  AVX2              | %12.1f | -        | -\n", avg_us);
    }
#endif

    printf("\n  Note: SSE2/AVX2 speedup shown relative to scalar.\n");
}

int main(int argc, char** argv) {
    std::string data_dir = "./data";
    int test_queries = 100, top_k = 10;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i+1 < argc) data_dir = argv[++i];
        else if (strcmp(argv[i], "--queries") == 0 && i+1 < argc) test_queries = atoi(argv[++i]);
        else if (strcmp(argv[i], "--topk") == 0 && i+1 < argc) top_k = atoi(argv[++i]);
    }

    IntType n, dim, nq, dq, gt_n, gt_k;
    std::string base_path  = data_dir + "/DEEP100K.base.100k.fbin.larkcache";
    std::string query_path = data_dir + "/DEEP100K.query.fbin";
    std::string gt_path    = data_dir + "/DEEP100K.gt.query.100k.top100.bin";

    FloatType* h_base  = load_fbin(base_path.c_str(), &n, &dim);
    FloatType* h_query = load_fbin(query_path.c_str(), &nq, &dq);
    IntType*   h_gt    = load_gt_bin(gt_path.c_str(), &gt_n, &gt_k);

    if (!h_base || !h_query || !h_gt) { fprintf(stderr, "Data load failed\n"); return 1; }
    if (test_queries > nq) test_queries = nq;

    VectorDataset base, query;
    base.data = h_base; base.n = n; base.dim = dim;
    query.data = h_query; query.n = test_queries; query.dim = dq;

    printf("=== SIMD Vectorized ANN Search ===\n");
    printf("  Base: %d x %d, Queries: %d, Top-%d\n", n, dim, test_queries, top_k);

    simd_benchmark_flat(base, query, top_k, h_gt, gt_k);

    free(h_base); free(h_query); free(h_gt);
    return 0;
}
