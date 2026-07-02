/**
 * ann_serial.cpp — 串行基线 ANN 暴力搜索
 *
 * O(nkd) 三重循环暴力搜索，作为所有并行方案的正确性基线和性能参照。
 */

#include "ann_common.h"

void serial_flat_search(const VectorDataset& base, const VectorDataset& query,
                         IntType k, SearchResult* results, double* avg_us = nullptr) {
    HiResTimer timer;
    timer.start();

    for (IntType q = 0; q < (IntType)query.n; q++) {
        const FloatType* qvec = query[q];
        std::vector<FloatType> dists(base.n);

        // 计算所有距离
        for (IntType i = 0; i < base.n; i++) {
            dists[i] = inner_product_dist(qvec, base[i], base.dim);
        }

        // Top-K
        topk_select(dists.data(), base.n, k,
                     results[q].indices, results[q].distances);
    }

    if (avg_us)
        *avg_us = timer.elapsed_us() / query.n;
}

// ============================================================
int main(int argc, char** argv) {
    std::string data_dir = "./data";
    int test_queries = 100, top_k = 10;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i+1 < argc) data_dir = argv[++i];
        else if (strcmp(argv[i], "--queries") == 0 && i+1 < argc) test_queries = atoi(argv[++i]);
        else if (strcmp(argv[i], "--topk") == 0 && i+1 < argc) top_k = atoi(argv[++i]);
    }

    // 加载数据
    IntType n, dim, nq, dq, gt_n, gt_k;
    std::string base_path  = data_dir + "/DEEP100K.base.100k.fbin.larkcache";
    std::string query_path = data_dir + "/DEEP100K.query.fbin";
    std::string gt_path    = data_dir + "/DEEP100K.gt.query.100k.top100.bin";

    FloatType* h_base  = load_fbin(base_path.c_str(), &n, &dim);
    FloatType* h_query = load_fbin(query_path.c_str(), &nq, &dq);
    IntType*   h_gt    = load_gt_bin(gt_path.c_str(), &gt_n, &gt_k);

    if (!h_base || !h_query || !h_gt || dim != dq) {
        fprintf(stderr, "[ERROR] Data loading failed or dimension mismatch\n");
        return 1;
    }

    if (test_queries > nq) test_queries = nq;

    VectorDataset base, query;
    base.data  = h_base;  base.n = n;   base.dim  = dim;
    query.data = h_query; query.n = test_queries; query.dim = dq;

    // 分配结果
    std::vector<SearchResult> results(test_queries);
    for (auto& r : results) r.allocate(top_k);

    // 执行搜索
    printf("\n=== 串行基线 ANN 搜索 ===\n");
    printf("  Base: %d x %d vectors\n", n, dim);
    printf("  Query: %d vectors, Top-%d\n", test_queries, top_k);

    double avg_us = 0;
    serial_flat_search(base, query, top_k, results.data(), &avg_us);

    // 召回率
    std::vector<IntType> result_flat(test_queries * top_k);
    for (int q = 0; q < test_queries; q++)
        memcpy(&result_flat[q * top_k], results[q].indices, top_k * sizeof(IntType));

    double recall = compute_recall(result_flat.data(), h_gt, test_queries, top_k, gt_k);

    printf("\n  Results:\n");
    printf("  ┌─────────────────────────────┐\n");
    printf("  │ Per Query:  %10.1f us    │\n", avg_us);
    printf("  │ Total:      %10.1f ms    │\n", avg_us * test_queries / 1000.0);
    printf("  │ Recall@%d:   %10.4f      │\n", top_k, recall);
    printf("  │ GFLOPS:     %10.3f       │\n",
           (double)(base.n * base.dim * test_queries * 2) / (avg_us * test_queries * 1e3));
    printf("  └─────────────────────────────┘\n");

    free(h_base); free(h_query); free(h_gt);
    return 0;
}
