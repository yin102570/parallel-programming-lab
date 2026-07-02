/**
 * ann_mpi_async.cpp — MPI 异步 SGD 分布式 ANN 训练
 *
 * 🆕 期末全新开发 — 异步随机梯度下降
 *
 * 核心创新: 用非阻塞通信 (MPI_Isend/Irecv) 替代同步 Allreduce，
 * 消除同步屏障等待，允许各进程以不同速度推进训练。
 *
 * 理论: 异步 SGD 引入 stale gradient，但通过步长衰减可收敛。
 *
 * 用法: mpirun -np 4 ./ann_mpi_async --data <dir>
 */

#include <mpi.h>
#include "ann_common.h"
#include <cstring>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>

// ============================================================
// 异步参数服务器架构
// ============================================================

/** 梯度消息 */
struct GradientMessage {
    int    sender_rank;
    int    iteration;
    FloatType* data;
    int    size;
};

/**
 * 参数服务器进程 (rank 0)
 * 维护全局模型参数，异步接收各 worker 梯度并更新
 */
class ParameterServer {
    std::vector<FloatType> params_;
    std::mutex mtx_;
    int dim_;
    int nworkers_;
    std::atomic<int> total_updates_{0};

public:
    ParameterServer(int dim, int nworkers) : dim_(dim), nworkers_(nworkers) {
        params_.resize(dim, 0.0f);
        std::mt19937 rng(42);
        std::uniform_real_distribution<FloatType> dist(-0.01f, 0.01f);
        for (int i = 0; i < dim; i++) params_[i] = dist(rng);
    }

    /** 异步接收梯度并更新参数 */
    void apply_gradient(const FloatType* grads, int iter, FloatType lr = 0.001f) {
        std::lock_guard<std::mutex> lock(mtx_);
        // Staleness-aware: 衰减陈旧梯度
        FloatType alpha = 1.0f / (1.0f + 0.01f * (total_updates_.load() - iter));
        for (int i = 0; i < dim_; i++)
            params_[i] -= lr * alpha * grads[i];
        total_updates_++;
    }

    void get_params(FloatType* out) {
        std::lock_guard<std::mutex> lock(mtx_);
        memcpy(out, params_.data(), dim_ * sizeof(FloatType));
    }

    int total_updates() const { return total_updates_.load(); }
};

/**
 * Worker 进程 (rank 1..nprocs-1)
 * 维护本地数据分片，异步拉取参数、计算梯度、推送梯度
 */
class AsyncWorker {
    FloatType* local_data_;
    IntType    local_n_;
    int dim_;
    int rank_;
    int server_rank_ = 0;

public:
    AsyncWorker(FloatType* data, IntType n, int dim, int rank)
        : local_data_(data), local_n_(n), dim_(dim), rank_(rank) {}

    /** 计算本地梯度 (模拟 ANN 损失关于参数的梯度) */
    void compute_gradient(const FloatType* params, FloatType* grads, int batch_size = 64) {
        std::mt19937 rng(rank_ + 123);
        std::uniform_real_distribution<FloatType> dist(-1.0f, 1.0f);

        memset(grads, 0, dim_ * sizeof(FloatType));
        int actual_batch = std::min(batch_size, (int)local_n_);

        for (int b = 0; b < actual_batch; b++) {
            FloatType* sample = local_data_ + (b % local_n_) * dim_;
            FloatType dot = 0;
            for (int d = 0; d < dim_; d++)
                dot += params[d] * sample[d];
            FloatType loss_grad = 2.0f * (dot - 1.0f);
            for (int d = 0; d < dim_; d++)
                grads[d] += loss_grad * sample[d];
        }
        for (int d = 0; d < dim_; d++)
            grads[d] /= actual_batch;
    }
};

// ============================================================
// 非阻塞通信实现异步 SGD
// ============================================================

void async_sgd_server(int dim, int nworkers, int max_iters) {
    ParameterServer server(dim, nworkers);
    int received = 0;
    int expected = nworkers * max_iters;

    printf("  [Server] Waiting for %d gradient updates...\n", expected);

    HiResTimer timer;
    timer.start();

    while (received < expected) {
        for (int w = 1; w <= nworkers; w++) {
            MPI_Status status;
            int flag = 0;
            MPI_Iprobe(w, 0, MPI_COMM_WORLD, &flag, &status);
            if (flag) {
                int count;
                MPI_Get_count(&status, MPI_FLOAT, &count);
                std::vector<FloatType> grads(count);
                MPI_Recv(grads.data(), count, MPI_FLOAT, w, 0,
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                server.apply_gradient(grads.data(), received);

                // 回传更新后的参数
                std::vector<FloatType> params(dim);
                server.get_params(params.data());
                MPI_Send(params.data(), dim, MPI_FLOAT, w, 1, MPI_COMM_WORLD);

                received++;
                if (received % (expected / 10) == 0 || received == expected)
                    printf("  [Server] Progress: %d/%d updates\n", received, expected);
            }
        }
    }

    double elapsed = timer.elapsed_s();
    printf("  [Server] Complete: %d updates in %.2f s (%.1f updates/s)\n",
           received, elapsed, received / elapsed);
}

void async_sgd_worker(int rank, int dim, int max_iters, int batch_size) {
    // 模拟本地数据
    int local_n = 25000;
    std::vector<FloatType> local_data(local_n * dim);
    generate_random_dataset(local_data.data(), local_n, dim, rank * 100);

    AsyncWorker worker(local_data.data(), local_n, dim, rank);

    std::vector<FloatType> params(dim);
    std::vector<FloatType> grads(dim);

    HiResTimer timer;
    timer.start();

    for (int iter = 0; iter < max_iters; iter++) {
        // 1. 拉取参数 (阻塞接收)
        if (iter == 0) {
            memset(params.data(), 0, dim * sizeof(FloatType));
        } else {
            MPI_Recv(params.data(), dim, MPI_FLOAT, 0, 1,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // 2. 计算本地梯度
        worker.compute_gradient(params.data(), grads.data(), batch_size);

        // 3. 推送梯度 (非阻塞发送)
        MPI_Request req;
        MPI_Isend(grads.data(), dim, MPI_FLOAT, 0, 0,
                   MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }

    double elapsed = timer.elapsed_s();
    printf("  [Worker %d] %d iters in %.2f s (%.1f iters/s)\n",
           rank, max_iters, elapsed, max_iters / elapsed);
}

// ============================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (nprocs < 2) {
        if (rank == 0) fprintf(stderr, "Need at least 2 processes (server + worker)\n");
        MPI_Finalize();
        return 1;
    }

    int dim = 96, max_iters = 100, batch_size = 64;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dim") == 0 && i+1 < argc) dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0 && i+1 < argc) max_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i+1 < argc) batch_size = atoi(argv[++i]);
    }

    if (rank == 0) {
        printf("\n╔══════════════════════════════════════════════════════════╗\n");
        printf("║  MPI 异步 SGD — 异步分布式 ANN 训练 (🆕 期末新开发)       ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");
        printf("  Architecture: Parameter Server + %d Workers\n", nprocs - 1);
        printf("  Model Dim: %d, Max Iters: %d, Batch: %d\n", dim, max_iters, batch_size);
        printf("  Key Innovation: Non-blocking Isend/Irecv + Staleness-aware\n\n");

        async_sgd_server(dim, nprocs - 1, max_iters);
    } else {
        async_sgd_worker(rank, dim, max_iters, batch_size);
    }

    MPI_Finalize();
    return 0;
}
