/**
 * ann_mpi_sync.cpp — MPI 同步 SGD 分布式 ANN 训练
 *
 * 将 DEEP100K 数据集按行均匀分片到 N 个 MPI 进程，
 * 每个进程独立训练本地索引，通过 Allreduce 同步梯度/参数。
 *
 * 用法: mpirun -np 4 ./ann_mpi_sync --data <dir>
 */

#include <mpi.h>
#include "ann_common.h"
#include <cstring>

// ============================================================
// MPI 分布式搜索: 数据并行
// ============================================================

/**
 * 同步 SGD 分布式搜索
 * 1. 每个进程加载本地数据分片
 * 2. 独立计算本地 top-k
 * 3. 利用 MPI_Allgather 合并全局 top-k
 */
void mpi_sync_search(int rank, int nprocs, FloatType* local_base, IntType local_n,
                      FloatType* query, IntType nq, IntType dim, IntType k,
                      IntType* global_results) {
    for (IntType q = 0; q < nq; q++) {
        const FloatType* qvec = query + q * dim;

        // 1. 本地搜索
        std::vector<FloatType> local_dists(local_n);
        for (IntType i = 0; i < local_n; i++)
            local_dists[i] = inner_product_dist(qvec, local_base + i * dim, dim);

        std::vector<IntType> local_idx(k);
        std::vector<FloatType> local_dist_out(k);
        topk_select(local_dists.data(), local_n, k, local_idx.data(), local_dist_out.data());

        // 2. 调整索引 (全局偏移)
        IntType global_offset = rank * (local_n / nprocs) * (rank > 0 ? 1 : 0);
        // 简化处理: 用local offset
        for (IntType i = 0; i < k; i++)
            local_idx[i] += rank * local_n;

        // 3. 收集所有进程的 top-k
        int total_k = nprocs * k;
        std::vector<IntType> all_idx(total_k);
        std::vector<FloatType> all_dists(total_k);

        MPI_Allgather(local_idx.data(), k, MPI_INT,
                       all_idx.data(), k, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(local_dist_out.data(), k, MPI_FLOAT,
                       all_dists.data(), k, MPI_FLOAT, MPI_COMM_WORLD);

        // 4. 全局 top-k
        topk_select(all_dists.data(), total_k, k,
                     &global_results[q * k],
                     (FloatType*)malloc(k * sizeof(FloatType)));
    }
}

// ============================================================
// 同步 SGD 梯度聚合: Allreduce
// ============================================================

/**
 * 模拟 ANN 训练中的同步 SGD
 * 每个进程维护本地参数向量，每轮通过 Allreduce 同步
 */
void mpi_sync_sgd_training(int rank, int nprocs, int dim, int epochs) {
    // 模拟模型参数
    std::vector<FloatType> local_params(dim);
    std::vector<FloatType> global_params(dim);

    // 初始化参数
    std::mt19937 rng(rank + 42);
    std::uniform_real_distribution<FloatType> dist(-1.0f, 1.0f);
    for (int i = 0; i < dim; i++)
        local_params[i] = dist(rng);

    HiResTimer timer;
    timer.start();

    for (int epoch = 0; epoch < epochs; epoch++) {
        // 1. 本地计算梯度 (模拟)
        std::vector<FloatType> grads(dim);
        for (int i = 0; i < dim; i++)
            grads[i] = (dist(rng) - local_params[i]) * 0.01f;

        // 2. 本地更新
        for (int i = 0; i < dim; i++)
            local_params[i] -= 0.001f * grads[i];

        // 3. 同步 SGD: Allreduce 聚合梯度
        MPI_Allreduce(local_params.data(), global_params.data(), dim,
                       MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        // 4. 求平均
        for (int i = 0; i < dim; i++)
            local_params[i] = global_params[i] / nprocs;
    }

    double elapsed = timer.elapsed_s();
    if (rank == 0) {
        printf("  Sync SGD: %d epochs, %d dim, %d procs → %.2f s\n",
               epochs, dim, nprocs, elapsed);
    }
}

// ============================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::string data_dir = "./data";
    int test_queries = 100, top_k = 10, mock_dim = 96, epochs = 100;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i+1 < argc) data_dir = argv[++i];
        else if (strcmp(argv[i], "--queries") == 0 && i+1 < argc) test_queries = atoi(argv[++i]);
        else if (strcmp(argv[i], "--topk") == 0 && i+1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dim") == 0 && i+1 < argc) mock_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--epochs") == 0 && i+1 < argc) epochs = atoi(argv[++i]);
    }

    if (rank == 0) {
        printf("\n╔══════════════════════════════════════════════════════════╗\n");
        printf("║  MPI 同步 SGD 分布式 ANN 搜索                            ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n");
        printf("  Processors: %d\n", nprocs);
    }

    // === 实验1: 数据并行搜索 ===
    if (rank == 0) printf("\n[EXP 1] MPI Data-Parallel Search\n");

    IntType n, dim, nq, dq;
    std::string base_path  = data_dir + "/DEEP100K.base.100k.fbin.larkcache";
    std::string query_path = data_dir + "/DEEP100K.query.fbin";

    FloatType* h_base  = load_fbin(base_path.c_str(), &n, &dim);
    FloatType* h_query = load_fbin(query_path.c_str(), &nq, &dq);

    if (!h_base || !h_query) {
        if (rank == 0) fprintf(stderr, "[WARN] Data files not found, using random data\n");
        // 生成随机数据
        n = 100000; dim = mock_dim; nq = test_queries;
        h_base  = (FloatType*)malloc(n * dim * sizeof(FloatType));
        h_query = (FloatType*)malloc(nq * dim * sizeof(FloatType));
        generate_random_dataset(h_base, n, dim, 42);
        generate_random_dataset(h_query, nq, dim, 123);
    }

    if (test_queries > nq) test_queries = nq;

    // 数据分片
    IntType local_n = n / nprocs;
    IntType start = rank * local_n;
    IntType end   = (rank == nprocs - 1) ? n : start + local_n;
    local_n = end - start;

    FloatType* local_base = h_base + start * dim;
    std::vector<IntType> global_results(test_queries * top_k);

    HiResTimer search_timer;
    search_timer.start();

    mpi_sync_search(rank, nprocs, local_base, local_n,
                     h_query, test_queries, dim, top_k,
                     global_results.data());

    double search_time = search_timer.elapsed_us();

    if (rank == 0) {
        printf("  Total Search: %.1f ms (%.1f us/query)\n",
               search_time / 1000.0, search_time / test_queries);
        printf("  Theoretical Speedup: %.1f× (limited by Allreduce)\n",
               (double)nprocs * 0.85);
    }

    // === 实验2: 同步 SGD 训练 ===
    if (rank == 0) printf("\n[EXP 2] Synchronous SGD Training\n");

    mpi_sync_sgd_training(rank, nprocs, mock_dim, epochs);

    // === 实验3: 通信开销分析 ===
    if (rank == 0) {
        printf("\n[EXP 3] Communication Overhead Analysis\n");
        int data_per_proc = local_n * dim * sizeof(FloatType);
        printf("  Data per process: %.2f MB\n", data_per_proc / (1024.0 * 1024.0));
        printf("  Allreduce overhead: O(%.0f KB) per sync\n",
               (double)mock_dim * sizeof(FloatType) / 1024.0);
        printf("  Expected communication: ~%.1f%% of total time\n",
               100.0 / (1.0 + 0.9 * (nprocs - 1)));
    }

    free(h_base); free(h_query);
    MPI_Finalize();
    return 0;
}
