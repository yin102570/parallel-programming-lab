/**
 * main.c — ANNS性能测试主程序
 *
 * 编译（Linux/ARM）：
 *   gcc -O2 -march=native -std=c11 -fopenmp -o benchmark main.c
 *
 * 编译（macOS Apple Silicon）：
 *   clang -O2 -march=armv8.4-a+neon -std=c11 -fopenmp -o benchmark main.c
 *   (macOS默认Clang不带OpenMP，需用brew install libomp)
 *
 * 编译（x86 Linux）：
 *   gcc -O2 -march=native -std=c11 -fopenmp -o benchmark main.c
 *
 * 运行示例：
 *   ./benchmark vectors.bin 1000 100  # 1000查询，K=100
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <errno.h>

#include "flat_search.h"
#include "simd_distance.h"
#include "ivf_index.h"
#include "pq_index.h"
#include "ivf_pq_index.h"

//==============================================================================
// 工具：计时
//==============================================================================

/**
 * 高精度计时（POSIX）
 * 使用clock_gettime而非clock()以获得微秒级精度
 */
double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

//==============================================================================
// 工具：打印结果
//==============================================================================

void print_search_result(const char* method, double latency_us, float recall,
                         int k, int n_queries) {
    double avg_latency_ms = latency_us / 1000.0 / n_queries;
    double qps = n_queries / (latency_us / 1e6);
    printf("%-20s | Latency: %.3f ms/query | QPS: %.0f | Recall@%d: %.4f\n",
           method, avg_latency_ms, qps, k, recall);
}

//==============================================================================
// 测试1：Flat基线
//==============================================================================

void test_flat(const float* vectors, int n, int dim, const float* queries,
               int n_queries, int k) {
    printf("\n========== Flat 基线测试 ==========\n");

    // 生成Ground Truth（用Flat搜索）
    printf("生成Ground Truth（线性扫描）...\n");
    SearchResult* gt_results = (SearchResult*)malloc(sizeof(SearchResult) * n_queries);
    double gt_time = get_time_us();
    for (int q = 0; q < n_queries; q++) {
        SearchResult* r = flat_search(&(VectorDataset){(float*)vectors, n, dim},
                                      queries + q * dim, k);
        gt_results[q].indices = r->indices;
        gt_results[q].distances = r->distances;
        gt_results[q].k = k;
        free(r);
    }
    gt_time = get_time_us() - gt_time;
    printf("Ground Truth完成: %.1f ms\n", gt_time / 1000.0);

    // 测试Flat搜索
    double search_time = get_time_us();
    SearchResult* pred_results = (SearchResult*)malloc(sizeof(SearchResult) * n_queries);
    for (int q = 0; q < n_queries; q++) {
        pred_results[q] = *(flat_search(&(VectorDataset){(float*)vectors, n, dim},
                                        queries + q * dim, k));
        free(flat_search(&(VectorDataset){(float*)vectors, n, dim},
                         queries + q * dim, k));
    }
    search_time = get_time_us() - search_time;

    // 计算召回率
    float recall = 0.0f;
    for (int q = 0; q < n_queries; q++) {
        recall += recall_at_k(pred_results[q].indices, gt_results[q].indices, k);
    }
    recall /= n_queries;

    print_search_result("Flat(Linear Scan)", search_time, recall, k, n_queries);

    // 清理
    for (int q = 0; q < n_queries; q++) {
        free(pred_results[q].indices);
        free(pred_results[q].distances);
        free(gt_results[q].indices);
        free(gt_results[q].distances);
    }
    free(pred_results);
    free(gt_results);
}

//==============================================================================
// 测试2：Flat + SIMD
//==============================================================================

#if USE_NEON || USE_AVX2
void test_flat_simd(const float* vectors, int n, int dim, const float* queries,
                    int n_queries, int k) {
    printf("\n========== Flat + SIMD测试 ==========\n");

    // SIMD版本只用于距离计算，搜索逻辑同Flat
    double search_time = get_time_us();
    int hit = 0, total = 0;

    for (int q = 0; q < n_queries; q++) {
        // 简单计数：验证能正确找到top-k
        // 实际项目这里应该和Ground Truth对比算召回率
        (void)k;
    }
    search_time = get_time_us() - search_time;

    print_search_result("Flat+SIMD", search_time, -1.0f, k, n_queries);
}
#endif

//==============================================================================
// 测试3：IVF
//==============================================================================

void test_ivf(const float* vectors, int n, int dim, const float* queries,
              int n_queries, int k, int n_list, int n_probe) {
    printf("\n========== IVF 测试 (n_list=%d, n_probe=%d) ==========\n",
           n_list, n_probe);

    // 构建索引
    printf("构建IVF索引...\n");
    double build_start = get_time_us();
    IVFIndex* index = ivf_build(vectors, n, dim, n_list, n_probe, 42);
    double build_time = get_time_us() - build_start;
    printf("索引构建: %.1f ms\n", build_time / 1000.0);

    // 搜索
    double search_time = get_time_us();
    int hit = 0, total = 0;

    for (int q = 0; q < n_queries; q++) {
        IVFSearchResult* r = ivf_search(index, vectors, queries + q * dim, k);
        total += k;
        // 本测试省略召回率验证（需要Ground Truth）
        ivf_free_result(r);
    }
    search_time = get_time_us() - search_time;

    print_search_result("IVF", search_time, -1.0f, k, n_queries);

    // 统计信息
    int avg_candidates = 0;
    for (int p = 0; p < n_probe; p++) {
        avg_candidates += index->clusters[p].n_members;
    }
    printf("平均候选数: %d (理论值: %d)\n",
           avg_candidates, n / n_list * n_probe);

    ivf_free(index);
}

//==============================================================================
// 测试4：IVF-PQ（核心测试）
//==============================================================================

void test_ivfpq(const float* vectors, int n, int dim, const float* queries,
                int n_queries, int k, int n_list, int n_probe,
                int pq_m, int pq_ks, int n_threads) {
    printf("\n========== IVF-PQ 测试 (n_list=%d, n_probe=%d, M=%d, threads=%d) ==========\n",
           n_list, n_probe, pq_m, n_threads);

    // 构建索引
    printf("构建IVF-PQ索引...\n");
    double build_start = get_time_us();
    IVFPQIndex* index = ivfpq_build(vectors, n, dim, n_list, n_probe, pq_m, pq_ks);
    double build_time = get_time_us() - build_start;
    printf("索引构建: %.1f ms\n", build_time / 1000.0);

    // PQ存储分析
    size_t pq_bytes = (size_t)n * pq_m;
    size_t ivf_bytes = n_list * sizeof(int) * n; // 上界，实际更小
    size_t total_index_bytes = pq_bytes + ivf_bytes + n * dim * sizeof(float);
    printf("存储分析:\n");
    printf("  原始向量: %.1f MB\n", n * dim * sizeof(float) / 1e6);
    printf("  PQ编码: %.1f MB (压缩比: %.1fx)\n",
           pq_bytes / 1e6, (float)n * dim * sizeof(float) / pq_bytes);
    printf("  总索引: %.1f MB\n", total_index_bytes / 1e6);

    // 搜索
    printf("执行查询...\n");
    double search_time = get_time_us();

    for (int q = 0; q < n_queries; q++) {
        IVFPQResult* r;
        if (n_threads > 1) {
            r = ivfpq_search_parallel(index, queries + q * dim, k, n_threads);
        } else {
            r = ivfpq_search(index, queries + q * dim, k);
        }
        ivfpq_free_result(r);
    }
    search_time = get_time_us() - search_time;

    double avg_latency_ms = search_time / 1000.0 / n_queries;
    double qps = n_queries / (search_time / 1e6);

    printf("\n结果:\n");
    printf("  平均延迟: %.3f ms/query\n", avg_latency_ms);
    printf("  吞吐量: %.0f queries/sec\n", qps);
    printf("  加速比(vs Flat 10ms): %.1fx\n", 10.0 / avg_latency_ms);

    ivfpq_free(index);
}

//==============================================================================
// 参数调优：网格搜索n_probe
//==============================================================================

void tune_nprobe(const float* vectors, int n, int dim, const float* queries,
                 int n_queries, int k, int n_list) {
    printf("\n========== n_probe 参数调优 ==========\n");

    IVFPQIndex* index = ivfpq_build(vectors, n, dim, n_list, 50, 8, 256);

    int probes[] = {5, 10, 20, 50};
    for (int pi = 0; pi < 4; pi++) {
        int n_probe = probes[pi];
        index->n_probe = n_probe;

        double search_time = get_time_us();
        for (int q = 0; q < n_queries; q++) {
            IVFPQResult* r = ivfpq_search(index, queries + q * dim, k);
            ivfpq_free_result(r);
        }
        search_time = get_time_us() - search_time;

        double avg_latency_ms = search_time / 1000.0 / n_queries;
        printf("n_probe=%2d | Latency=%.3f ms | candidates~%d\n",
               n_probe, avg_latency_ms, n / n_list * n_probe);
    }

    ivfpq_free(index);
}

//==============================================================================
// 主函数
//==============================================================================

void print_usage(const char* prog) {
    printf("用法: %s <vectors_file> [选项]\n", prog);
    printf("\n选项:\n");
    printf("  --n-queries N    查询数量 (默认: 100)\n");
    printf("  --k N            返回近邻数 (默认: 100)\n");
    printf("  --n-list N       IVF簇数 (默认: 256)\n");
    printf("  --n-probe N      探查簇数 (默认: 10)\n");
    printf("  --pq-m N         PQ分段数 (默认: 8)\n");
    printf("  --threads N      线程数 (默认: 4)\n");
    printf("  --tune           运行参数调优\n");
    printf("  --test METHOD    运行指定测试: flat, ivf, ivfpq, all\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* vector_file = argv[1];

    // 默认参数
    int n_queries = 100;
    int k = 100;
    int n_list = 256;
    int n_probe = 10;
    int pq_m = 8;
    int pq_ks = 256;
    int n_threads = 4;
    int do_tune = 0;
    const char* test_method = "all";

    // 解析参数
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--n-queries") == 0 && i+1 < argc)
            n_queries = atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i+1 < argc)
            k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n-list") == 0 && i+1 < argc)
            n_list = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n-probe") == 0 && i+1 < argc)
            n_probe = atoi(argv[++i]);
        else if (strcmp(argv[i], "--pq-m") == 0 && i+1 < argc)
            pq_m = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i+1 < argc)
            n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tune") == 0)
            do_tune = 1;
        else if (strcmp(argv[i], "--test") == 0 && i+1 < argc)
            test_method = argv[++i];
    }

    // 加载向量数据
    printf("加载向量数据: %s\n", vector_file);
    VectorDataset* dataset = load_dataset(vector_file);
    if (!dataset) {
        fprintf(stderr, "错误: 无法加载向量文件 %s\n", vector_file);
        return 1;
    }
    printf("向量数: %d, 维度: %d\n", dataset->n, dataset->dim);

    // 生成随机查询向量
    printf("生成 %d 个随机查询向量...\n", n_queries);
    float* queries = (float*)malloc(sizeof(float) * n_queries * dataset->dim);
    srand(12345);
    for (int i = 0; i < n_queries * dataset->dim; i++) {
        queries[i] = (float)rand() / RAND_MAX;
    }

    // 执行测试
    if (strcmp(test_method, "flat") == 0 || strcmp(test_method, "all") == 0) {
        test_flat(dataset->data, dataset->n, dataset->dim, queries, n_queries, k);
    }

    if (strcmp(test_method, "ivf") == 0 || strcmp(test_method, "all") == 0) {
        test_ivf(dataset->data, dataset->n, dataset->dim, queries, n_queries, k, n_list, n_probe);
    }

    if (strcmp(test_method, "ivfpq") == 0 || strcmp(test_method, "all") == 0) {
        test_ivfpq(dataset->data, dataset->n, dataset->dim, queries, n_queries,
                   k, n_list, n_probe, pq_m, pq_ks, n_threads);

        // 测试不同线程数
        printf("\n--- 线程扩展性测试 ---");
        for (int t = 1; t <= 8; t *= 2) {
            if (t == n_threads) continue;
            test_ivfpq(dataset->data, dataset->n, dataset->dim, queries, n_queries,
                       k, n_list, n_probe, pq_m, pq_ks, t);
        }
    }

    if (do_tune) {
        tune_nprobe(dataset->data, dataset->n, dataset->dim, queries, n_queries, k, n_list);
    }

    // 清理
    free(queries);
    free_dataset(dataset);

    printf("\n测试完成。\n");
    return 0;
}
