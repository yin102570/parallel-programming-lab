/**
 * ann_cuda.cu — CUDA Flat ANN 搜索 (Tiled GEMM)
 *
 * 将 ANN 搜索映射为矩阵乘法: Scores = 1.0 - Base × Query^T
 * 核心: Tiled GEMM + Shared Memory + Wavefront/Warp 对齐
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// ============================================================
// 配置
// ============================================================
#define TILE_M   16
#define TILE_N   64   // 对齐 warp size=32, wavefront=64
#define TILE_K   32
#define BLOCK_SIZE 256
#define WARP_SIZE  32

// ============================================================
// CUDA Kernel: Tiled GEMM for distance matrix
// ============================================================

__global__ void kernel_tiled_gemm(
    const float* __restrict__ base,     // [n × dim]
    const float* __restrict__ query,    // [m × dim]
    float* __restrict__ scores,         // [n × m]
    int n, int m, int dim)
{
    __shared__ float Bs[TILE_M][TILE_K];
    __shared__ float Qs[TILE_K][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float acc = 0.0f;

    for (int k = 0; k < dim; k += TILE_K) {
        // 协作加载 tiles
        if (row < n && (k + threadIdx.x) < dim)
            Bs[threadIdx.y][threadIdx.x] = base[row * dim + k + threadIdx.x];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < m && (k + threadIdx.y) < dim)
            Qs[threadIdx.y][threadIdx.x] = query[col * dim + k + threadIdx.y];
        else
            Qs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // 规约累加
        #pragma unroll
        for (int t = 0; t < TILE_K; t++)
            acc += Bs[threadIdx.y][t] * Qs[t][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < m)
        scores[row * m + col] = 1.0f - acc;
}

// ============================================================
// CPU Top-K (Host side postprocessing)
// ============================================================

void cpu_topk_per_query(const float* scores, int n, int m, int k,
                         int* out_indices, float* out_distances) {
    #pragma omp parallel for
    for (int q = 0; q < m; q++) {
        std::vector<std::pair<float, int>> pairs(n);
        for (int i = 0; i < n; i++)
            pairs[i] = {scores[i * m + q], i};
        std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end());
        for (int j = 0; j < k; j++) {
            out_indices[q * k + j] = pairs[j].second;
            out_distances[q * k + j] = pairs[j].first;
        }
    }
}

// ============================================================
// CUDA Flat Search Driver
// ============================================================

double cuda_flat_search(const float* h_base, const float* h_query,
                         int n, int dim, int nq, int k,
                         int* h_result, float* h_distances,
                         int batch_size = 64) {
    float *d_base, *d_query, *d_scores;
    size_t size_base  = (size_t)n * dim * sizeof(float);
    size_t size_query = (size_t)batch_size * dim * sizeof(float);
    size_t size_score = (size_t)n * batch_size * sizeof(float);

    cudaMalloc(&d_base,  size_base);
    cudaMalloc(&d_query, size_query);
    cudaMalloc(&d_scores, size_score);

    cudaMemcpy(d_base, h_base, size_base, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 分批处理查询
    for (int b = 0; b < nq; b += batch_size) {
        int batch = std::min(batch_size, nq - b);

        cudaMemcpy(d_query, h_query + b * dim,
                    batch * dim * sizeof(float), cudaMemcpyHostToDevice);

        dim3 grid((batch + TILE_N - 1) / TILE_N,
                   (n + TILE_M - 1) / TILE_M);
        dim3 block(TILE_N, TILE_M);

        kernel_tiled_gemm<<<grid, block>>>(d_base, d_query, d_scores, n, batch, dim);
        cudaDeviceSynchronize();

        // 拷贝 scores 回 host 做 top-k
        float* h_scores_part = (float*)malloc(n * batch * sizeof(float));
        cudaMemcpy(h_scores_part, d_scores, n * batch * sizeof(float),
                    cudaMemcpyDeviceToHost);

        cpu_topk_per_query(h_scores_part, n, batch, k,
                            h_result + b * k, h_distances + b * k);
        free(h_scores_part);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(d_base); cudaFree(d_query); cudaFree(d_scores);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return ms;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    std::string data_dir = "./data";
    int test_queries = 100, top_k = 10, batch_size = 64;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i+1 < argc) data_dir = argv[++i];
        else if (strcmp(argv[i], "--queries") == 0 && i+1 < argc) test_queries = atoi(argv[++i]);
        else if (strcmp(argv[i], "--topk") == 0 && i+1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i+1 < argc) batch_size = atoi(argv[++i]);
    }

    // 加载数据
    int n, dim, nq, dq;
    std::string base_path  = data_dir + "/DEEP100K.base.100k.fbin.larkcache";
    std::string query_path = data_dir + "/DEEP100K.query.fbin";

    FILE* fbase = fopen(base_path.c_str(), "rb");
    FILE* fquery = fopen(query_path.c_str(), "rb");

    if (!fbase || !fquery) {
        fprintf(stderr, "[ERROR] Data files not found. Generating random data...\n");
        n = 100000; dim = 96; nq = test_queries;
    } else {
        fread(&n, sizeof(int), 1, fbase);
        fread(&dim, sizeof(int), 1, fbase);
        fread(&nq, sizeof(int), 1, fquery);
        fread(&dq, sizeof(int), 1, fquery);
        fclose(fbase); fclose(fquery);
    }

    if (test_queries > nq) test_queries = nq;

    float* h_base  = (float*)malloc(n * dim * sizeof(float));
    float* h_query = (float*)malloc(test_queries * dim * sizeof(float));

    if (fbase) {
        fbase = fopen(base_path.c_str(), "rb");
        fseek(fbase, 8, SEEK_SET);
        fread(h_base, sizeof(float), n * dim, fbase);
        fclose(fbase);
    } else {
        for (int i = 0; i < n * dim; i++)
            h_base[i] = (float)rand() / RAND_MAX;
    }

    if (fquery) {
        fquery = fopen(query_path.c_str(), "rb");
        fseek(fquery, 8, SEEK_SET);
        fread(h_query, sizeof(float), test_queries * dim, fquery);
        fclose(fquery);
    } else {
        for (int i = 0; i < test_queries * dim; i++)
            h_query[i] = (float)rand() / RAND_MAX;
    }

    // 分配结果
    int*   h_result    = (int*)malloc(test_queries * top_k * sizeof(int));
    float* h_distances = (float*)malloc(test_queries * top_k * sizeof(float));

    printf("\n=== CUDA GPU Flat ANN Search (Tiled GEMM) ===\n");
    printf("  Base: %d x %d, Queries: %d, Top-%d, Batch: %d\n",
           n, dim, test_queries, top_k, batch_size);

    // GPU 信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  GPU: %s, %d SMs, %.1f GB VRAM\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    double ms = cuda_flat_search(h_base, h_query, n, dim,
                                  test_queries, top_k,
                                  h_result, h_distances, batch_size);

    printf("\n  Results:\n");
    printf("  ┌─────────────────────────────┐\n");
    printf("  │ Total Time:  %10.1f ms   │\n", ms);
    printf("  │ Per Query:   %10.1f us   │\n", ms * 1000.0 / test_queries);
    printf("  │ QPS:         %10.1f       │\n", test_queries / (ms / 1000.0));
    printf("  └─────────────────────────────┘\n");

    free(h_base); free(h_query); free(h_result); free(h_distances);
    return 0;
}
