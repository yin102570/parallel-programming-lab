/**
 * main_ann_gpu.cc
 * ANN GPU加速 — 主实验程序
 *
 * 5种运行模式:
 *   Mode 0: 单查询GPU优化 (Level 1)
 *   Mode 1: 持久化内存单查询
 *   Mode 2: Batch矩阵乘法 (Level 2, 推荐)
 *   Mode 3: rocBLAS对比 (Level 3)
 *   Mode 4: Block Size调优
 *
 * 用法:
 *   ./ann_gpu [mode] [--batch N] [--queries N] [--data PATH]
 */

#include "flat_scan_gpu_hip.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>

// 默认路径 — 根据平台调整
#ifdef _WIN32
#define DEFAULT_DATA_DIR "D:\\HuaweiMoveData\\Users\\asdf1\\Desktop\\ann数据集"
#else
#define DEFAULT_DATA_DIR "./data"
#endif

#define DEFAULT_BASE_FILE  "DEEP100K.base.100k.fbin.larkcache"
#define DEFAULT_QUERY_FILE "DEEP100K.query.fbin"
#define DEFAULT_GT_FILE    "DEEP100K.gt.query.100k.top100.bin"

// ============================================================
// CPU Baseline (Level 0)
// ============================================================

void cpu_flat_search(
    const float* base, const float* query,
    int n, int dim, int nq, int top_k,
    int* result,
    double* avg_time_us = nullptr)
{
    std::vector<float> distances(n);
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int q = 0; q < nq; q++) {
        const float* q_vec = &query[q * dim];

        // 计算所有距离
        for (int i = 0; i < n; i++) {
            float dot = 0.0f;
            const float* b_vec = &base[i * dim];
            for (int d = 0; d < dim; d++) {
                dot += q_vec[d] * b_vec[d];
            }
            distances[i] = 1.0f - dot;  // 内积距离
        }

        // CPU Top-K (partial sort)
        std::vector<std::pair<float, int>> idx(n);
        for (int i = 0; i < n; i++) idx[i] = {distances[i], i};
        std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end());

        for (int k = 0; k < top_k; k++)
            result[q * top_k + k] = idx[k].second;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    if (avg_time_us) {
        *avg_time_us = std::chrono::duration<double, std::micro>(
            t_end - t_start).count() / nq;
    }
}

// ============================================================
// 报告打印
// ============================================================

void print_header(const char* title) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  %-56s  ║\n", title);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
}

void print_result(const GpuBenchResult& r, const char* label,
                  double cpu_us = 0, double other_us = 0) {
    printf("┌─ %s\n", label);
    printf("│  Total Time:    %10.1f us\n", r.total_time_us);
    printf("│  Per Query:     %10.1f us\n", r.per_query_us);
    printf("│  Compute:       %10.1f us\n", r.compute_time_us);
    printf("│  H2D Transfer:  %10.1f us\n", r.h2d_time_us);
    printf("│  D2H Transfer:  %10.1f us\n", r.d2h_time_us);
    printf("│  QPS:           %10.1f\n", r.qps);
    printf("│  TFLOPS(est):   %10.3f\n", r.tflops);
    printf("│  Recall@%d:      %10.4f\n", r.top_k, r.recall_at_k);
    if (cpu_us > 0) {
        printf("│  Speedup vs CPU:%8.1fx\n", cpu_us / r.per_query_us);
    }
    if (other_us > 0) {
        printf("│  Speedup vs Other:%6.1fx\n", other_us / r.per_query_us);
    }
    printf("└────────────────────────────────\n");
}

void print_summary_table_header() {
    printf("\n┌──────────────────┬───────────┬──────────┬──────────┬──────────┐\n");
    printf("│ %-16s │ %9s │ %8s │ %8s │ %8s │\n",
           "Method", "PerQuery(us)", "Recall@10", "QPS", "TFLOPS");
    printf("├──────────────────┼───────────┼──────────┼──────────┼──────────┤\n");
}

void print_summary_row(const char* name, double per_query, double recall,
                       double qps, double tflops) {
    printf("│ %-16s │ %9.1f │ %7.4f  │ %8.0f │ %7.3f  │\n",
           name, per_query, recall, qps, tflops);
}

void print_summary_table_footer() {
    printf("└──────────────────┴───────────┴──────────┴──────────┴──────────┘\n");
}

// ============================================================
// 使用说明
// ============================================================

void print_usage(const char* prog) {
    printf("Usage: %s [mode] [options]\n\n", prog);
    printf("Modes:\n");
    printf("  0  单查询GPU优化 (Level 1) — 逐query执行, 共享内存+自适应block\n");
    printf("  1  持久化内存单查询 — 优化内存管理版本\n");
    printf("  2  Batch矩阵乘法 (Level 2, 推荐) — Tiled GEMM + wavefront感知\n");
    printf("  3  rocBLAS对比 (Level 3) — AMD官方库对比\n");
    printf("  4  Block Size调优 — 扫描最优block size\n");
    printf("  5  CPU Baseline — 原始CPU实现\n");
    printf("  all  运行全部模式\n\n");
    printf("Options:\n");
    printf("  --batch N     Batch大小 (默认64, 仅Mode 2/3有效)\n");
    printf("  --queries N   测试query数 (默认100)\n");
    printf("  --topk N      Top-K值 (默认10)\n");
    printf("  --data PATH   数据集目录\n");
    printf("  --help        显示帮助\n");
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    // 解析参数
    int mode = 2;            // 默认Batch模式
    int batch_size = 64;
    int test_queries = 100;
    int top_k = 10;
    std::string data_dir = DEFAULT_DATA_DIR;

    bool run_all = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--queries") == 0 && i + 1 < argc) {
            test_queries = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--topk") == 0 && i + 1 < argc) {
            top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "all") == 0) {
            run_all = true;
        } else {
            mode = atoi(argv[i]);
        }
    }

    // 加载数据
    std::string base_path  = data_dir + "/" + DEFAULT_BASE_FILE;
    std::string query_path = data_dir + "/" + DEFAULT_QUERY_FILE;
    std::string gt_path    = data_dir + "/" + DEFAULT_GT_FILE;

    printf("[INIT] Loading datasets from: %s\n", data_dir.c_str());

    int n, d, nq, dq, gt_nq, gt_k;
    float* h_base  = load_fbin_file(base_path.c_str(), &n, &d);
    float* h_query = load_fbin_file(query_path.c_str(), &nq, &dq);
    int*   h_gt    = load_gt_bin_file(gt_path.c_str(), &gt_nq, &gt_k);

    if (d != dq) {
        fprintf(stderr, "[ERROR] Dimension mismatch: base=%d, query=%d\n", d, dq);
        return 1;
    }

    test_queries = std::min(test_queries, nq);
    printf("[INIT] n=%d, dim=%d, queries=%d (using %d), top-%d\n",
           n, d, nq, test_queries, top_k);
    printf("[INIT] GPU: Using HIP\n\n");

    // 分配结果内存
    std::vector<int> h_result(test_queries * top_k);

    double cpu_per_query = 0;
    double other_gpu_per_query = 12038.9;  // 他人GPU baseline

    // ==================== 运行实验 ====================

    auto run_experiment = [&](int exp_mode, const char* label) {
        GpuBenchResult bench = {};
        memset(h_result.data(), 0, test_queries * top_k * sizeof(int));

        int ret = gpu_flat_search(
            h_base, h_query, n, d, test_queries, top_k,
            h_result.data(), (RunMode)exp_mode,
            DIST_INNER_PRODUCT, batch_size, &bench);

        if (ret != 0) {
            printf("[ERROR] %s failed\n", label);
            return bench;
        }

        bench.recall_at_k = compute_recall(
            h_result.data(), h_gt, test_queries, top_k, gt_k);

        print_result(bench, label, cpu_per_query, other_gpu_per_query);
        return bench;
    };

    if (run_all) {
        // ============ 全模式运行 ============
        print_header("ANN GPU 加速 — 全模式性能对比");

        // Level 0: CPU Baseline
        print_header("Level 0: CPU Baseline");
        std::vector<int> cpu_result(test_queries * top_k);
        cpu_flat_search(h_base, h_query, n, d, test_queries, top_k,
                        cpu_result.data(), &cpu_per_query);
        double cpu_recall = compute_recall(
            cpu_result.data(), h_gt, test_queries, top_k, gt_k);
        printf("  CPU Baseline: %.1f us/query, Recall@%d=%.4f\n",
               cpu_per_query, top_k, cpu_recall);

        // Level 1: 单查询GPU
        print_header("Level 1: 单查询GPU优化");
        GpuBenchResult r1 = run_experiment(MODE_SINGLE_QUERY, "GPU Single Query");

        // Level 2: Batch GEMM (多种batch)
        print_header("Level 2: Batch矩阵乘法 — Batch Size 扫描");
        printf("| Batch | Per Query (us) | Recall@10 | QPS | TFLOPS |\n");
        printf("| :--- | :--- | :--- | :--- | :--- |\n");

        int batch_sizes[] = {1, 8, 16, 32, 64, 128, 200};
        GpuBenchResult best_batch = {};
        double best_qps = 0;
        int best_bs = 0;

        for (int bs : batch_sizes) {
            if (bs > test_queries) continue;
            GpuBenchResult br = {};
            memset(h_result.data(), 0, test_queries * top_k * sizeof(int));
            gpu_flat_search(h_base, h_query, n, d, test_queries, top_k,
                h_result.data(), MODE_BATCH_GEMM, DIST_INNER_PRODUCT,
                bs, &br);
            br.recall_at_k = compute_recall(
                h_result.data(), h_gt, test_queries, top_k, gt_k);
            printf("| %d | %.1f | %.4f | %.0f | %.3f |\n",
                   bs, br.per_query_us, br.recall_at_k, br.qps, br.tflops);
            if (br.qps > best_qps) {
                best_qps = br.qps;
                best_batch = br;
                best_bs = bs;
            }
        }

        print_header("Level 2: 最优Batch配置");
        print_result(best_batch, "Best Batch GEMM", cpu_per_query, other_gpu_per_query);

        // Level 3: rocBLAS
        print_header("Level 3: rocBLAS 对比");
        GpuBenchResult r3 = run_experiment(MODE_ROCBLAS, "rocBLAS SGEMM");

        // 汇总表
        print_header("性能对比汇总");
        print_summary_table_header();
        print_summary_row("CPU Baseline", cpu_per_query, cpu_recall,
            (test_queries / cpu_per_query) * 1e6, 0);
        print_summary_row("Other GPU", other_gpu_per_query, 0.9995,
            (test_queries / other_gpu_per_query) * 1e6, 0);
        print_summary_row("Our Level 1", r1.per_query_us, r1.recall_at_k,
            r1.qps, r1.tflops);
        print_summary_row("Our Level 2", best_batch.per_query_us,
            best_batch.recall_at_k, best_batch.qps, best_batch.tflops);
        print_summary_row("Our Level 3", r3.per_query_us, r3.recall_at_k,
            r3.qps, r3.tflops);
        print_summary_table_footer();

    } else if (mode == 5) {
        // CPU Baseline only
        print_header("Level 0: CPU Baseline");
        cpu_flat_search(h_base, h_query, n, d, test_queries, top_k,
                        h_result.data(), &cpu_per_query);
        double cpu_recall = compute_recall(
            h_result.data(), h_gt, test_queries, top_k, gt_k);
        printf("  Time: %.1f us/query\n", cpu_per_query);
        printf("  Recall@%d: %.4f\n", top_k, cpu_recall);

    } else {
        // 单模式运行
        const char* labels[] = {
            "Level 1: 单查询GPU优化",
            "Level 1: 持久化内存单查询",
            "Level 2: Batch矩阵乘法",
            "Level 3: rocBLAS对比",
            "Level 4: Block Size调优"
        };

        print_header(labels[mode]);

        if (mode == 4) {
            // Block Size Tuning
            gpu_block_tuning(nullptr, h_query, n, d, test_queries, top_k,
                             h_result.data());
        } else {
            GpuBenchResult bench = run_experiment(mode, labels[mode]);
            print_summary_table_header();
            print_summary_row(labels[mode], bench.per_query_us,
                bench.recall_at_k, bench.qps, bench.tflops);
            print_summary_table_footer();
        }
    }

    // 清理
    free(h_base);
    free(h_query);
    free(h_gt);

    printf("\n[DONE] Experiment complete.\n");
    return 0;
}
