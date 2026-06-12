#!/bin/bash
# ============================================================
# AUP平台一键部署 + 运行脚本
# 用法: bash setup_and_run.sh [数据集目录路径]
# ============================================================

set -e

echo "============================================="
echo "  ANN GPU 加速 — AUP 一键部署脚本"
echo "============================================="

# ---- 参数 ----
DATA_DIR="${1:-}"
if [ -z "$DATA_DIR" ]; then
    echo "[INFO] 未指定数据集路径, 自动搜索..."
    # 常见路径
    CANDIDATES=(
        "$HOME/data"
        "$HOME/ann_data"
        "./data"
        "/data"
        "/home/jovyan/data"
        "/workspace/data"
    )
    for c in "${CANDIDATES[@]}"; do
        if [ -d "$c" ] && (ls "$c"/DEEP100K* 2>/dev/null | head -1); then
            DATA_DIR="$c"
            break
        fi
    done
    if [ -z "$DATA_DIR" ]; then
        echo ""
        echo "[!] 未找到数据集! 请手动指定路径:"
        echo "    bash $0 /path/to/dataset"
        exit 1
    fi
fi

echo "[OK] 数据集路径: $DATA_DIR"

# 验证数据集
for f in "DEEP100K.base.100k.fbin.larkcache" "DEEP100K.query.fbin"; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        echo "[ERROR] 缺少文件: $DATA_DIR/$f"
        echo "请确认数据集已上传到服务器!"
        exit 1
    fi
done
echo "[OK] 数据集验证通过"

# ---- 工作目录 ----
WORK_DIR="$HOME/gpu_ann"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo ""
echo "==== 创建源代码文件 ===="

# ==================== flat_scan_gpu_hip.h ====================
cat > flat_scan_gpu_hip.h << 'HEADER_EOF'
#ifndef FLAT_SCAN_GPU_HIP_H
#define FLAT_SCAN_GPU_HIP_H

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double total_time_us;
    double compute_time_us;
    double h2d_time_us;
    double d2h_time_us;
    double per_query_us;
    double recall_at_k;
    double qps;
    double tflops;
    int    num_queries;
    int    top_k;
} GpuBenchResult;

typedef enum {
    MODE_SINGLE_QUERY = 0,
    MODE_PERSISTENT   = 1,
    MODE_BATCH_GEMM   = 2,
    MODE_ROCBLAS      = 3,
    MODE_BLOCK_TUNING = 4
} RunMode;

typedef enum {
    DIST_INNER_PRODUCT = 0,
    DIST_L2            = 1
} DistMetric;

int gpu_flat_search(
    const float* base_data,
    const float* query_data,
    int n, int vecdim, int num_queries, int top_k,
    int* result_indices,
    RunMode mode, DistMetric metric, int batch_size,
    GpuBenchResult* result
);

float* load_fbin_file(const char* filename, int* n_out, int* d_out);
int* load_gt_bin_file(const char* filename, int* nq_out, int* k_out);
double compute_recall(const int* result_ids, const int* gt_ids, int nq, int K, int gt_K);
int get_adaptive_block_size(int vecdim);

#define HIP_CHECK(cmd) do { \
    hipError_t _err = cmd; \
    if (_err != hipSuccess) { \
        fprintf(stderr, "HIP Error [%s]: %s at %s:%d\n", \
            hipGetErrorName(_err), hipGetErrorString(_err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define GPU_MALLOC(ptr, size) HIP_CHECK(hipMalloc((void**)(ptr), (size)))
#define GPU_FREE(ptr)         HIP_CHECK(hipFree(ptr))

#ifdef __cplusplus
}
#endif
#endif
HEADER_EOF
echo "  [OK] flat_scan_gpu_hip.h"

# ==================== flat_scan_gpu_hip.cpp ====================
cat > flat_scan_gpu_hip.cpp << 'CPP_EOF'
#include "flat_scan_gpu_hip.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <chrono>

#define TILE_M      16
#define TILE_N      64
#define TILE_K      32
#define WARP_SIZE   64
#define MAX_K       128
#define TOPK_BLOCK  256

// ===== 工具函数 =====

float* load_fbin_file(const char* filename, int* n_out, int* d_out) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) { fprintf(stderr, "[ERROR] Cannot open: %s\n", filename); exit(1); }
    int n, d;
    input.read((char*)&n, 4); input.read((char*)&d, 4);
    float* data = (float*)malloc((size_t)n * d * sizeof(float));
    input.read((char*)data, (size_t)n * d * sizeof(float));
    input.close();
    if (n_out) *n_out = n; if (d_out) *d_out = d;
    printf("[LOAD] %s: %d x %d (%.2f MB)\n", filename, n, d, (double)n*d*4/(1024*1024));
    return data;
}

int* load_gt_bin_file(const char* filename, int* nq_out, int* k_out) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) { fprintf(stderr, "[ERROR] Cannot open: %s\n", filename); exit(1); }
    int nq, k;
    input.read((char*)&nq, 4); input.read((char*)&k, 4);
    int* data = (int*)malloc((size_t)nq * k * sizeof(int));
    input.read((char*)data, (size_t)nq * k * sizeof(int));
    input.close();
    if (nq_out) *nq_out = nq; if (k_out) *k_out = k;
    printf("[LOAD] %s: %d queries x top-%d\n", filename, nq, k);
    return data;
}

double compute_recall(const int* result_ids, const int* gt_ids, int nq, int K, int gt_K) {
    long long hits = 0;
    for (int i = 0; i < nq; i++)
        for (int j = 0; j < K; j++)
            for (int p = 0; p < gt_K; p++)
                if (result_ids[i*K+j] == gt_ids[i*gt_k+p]) { hits++; break; }
    return (double)hits / (double)(nq * K);
}

int get_adaptive_block_size(int vecdim) {
    if (vecdim <= 16)  return 1024;
    if (vecdim <= 32)  return 512;
    if (vecdim <= 128) return 256;
    return 128;
}

// ===== Level 1: 单查询内核 =====

__global__ void __launch_bounds__(256)
kernel_single_dot_product(
    const float* __restrict__ base,
    const float* __restrict__ query,
    float* __restrict__ scores,
    int n, int dim)
{
    extern __shared__ float s_query[];
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        s_query[i] = query[i];
    __syncthreads();

    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    float acc = 0.0f;
    const float* base_row = &base[j * dim];
    #pragma unroll 4
    for (int d = 0; d < dim; d++)
        acc += base_row[d] * s_query[d];
    scores[j] = 1.0f - acc;
}

__global__ void __launch_bounds__(TOPK_BLOCK)
kernel_single_topk(const float* __restrict__ scores, int* __restrict__ indices, int n, int K) {
    if (blockIdx.x > 0) return;
    float top_vals[MAX_K]; int top_ids[MAX_K];
    for (int i = 0; i < K; i++) top_vals[i] = 1e30f;

    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float val = scores[j];
        if (val < top_vals[K-1]) {
            int p = K-1;
            while (p > 0 && val < top_vals[p-1]) {
                top_vals[p] = top_vals[p-1]; top_ids[p] = top_ids[p-1]; p--;
            }
            top_vals[p] = val; top_ids[p] = j;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        for (int i = 0; i < K; i++) indices[i] = top_ids[i];
}

// ===== Level 2: Tiled GEMM 内核 (稳健版) =====

__global__ void __launch_bounds__(256)
kernel_tiled_gemm_simple(
    const float* __restrict__ base,
    const float* __restrict__ query,
    float* __restrict__ scores,
    int n, int m, int dim)
{
    __shared__ float Bs[16][32];
    __shared__ float Qs[32][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    float acc = 0.0f;

    for (int k = 0; k < dim; k += 32) {
        Bs[threadIdx.y][threadIdx.x] = (row < n && (k+threadIdx.x) < dim)
            ? base[row*dim+k+threadIdx.x] : 0.0f;
        Qs[threadIdx.y][threadIdx.x] = (col < m && (k+threadIdx.y) < dim)
            ? query[col*dim+k+threadIdx.y] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < 32; t++)
            acc = fmaf(Bs[threadIdx.y][t], Qs[t][threadIdx.x], acc);
        __syncthreads();
    }

    if (row < n && col < m)
        scores[row*m + col] = 1.0f - acc;
}

// ===== Batch Top-K 内核 =====

__global__ void __launch_bounds__(TOPK_BLOCK)
kernel_batch_topk(const float* __restrict__ scores, int* __restrict__ indices,
                  int n, int m, int K) {
    int q_idx = blockIdx.x;
    if (q_idx >= m) return;

    const float* query_col = &scores[q_idx];
    float top_vals[MAX_K]; int top_ids[MAX_K];
    for (int i = 0; i < K; i++) top_vals[i] = 1e30f;

    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float val = query_col[j * m];
        if (val < top_vals[K-1]) {
            int p = K-1;
            while (p > 0 && val < top_vals[p-1]) {
                top_vals[p] = top_vals[p-1]; top_ids[p] = top_ids[p-1]; p--;
            }
            top_vals[p] = val; top_ids[p] = j;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        for (int i = 0; i < K; i++) indices[q_idx*K+i] = top_ids[i];
}

// ===== Block Size Tuning 模板内核 =====

template<int BLOCK_SIZE>
__global__ void kernel_dot_product_bs(
    const float* __restrict__ base,
    const float* __restrict__ query,
    float* __restrict__ scores,
    int n, int dim)
{
    extern __shared__ float s_q[];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s_q[i] = query[i];
    __syncthreads();

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float acc = 0.0f;
    const float* b = &base[j * dim];
    #pragma unroll 4
    for (int d = 0; d < dim; d++) acc += b[d] * s_q[d];
    scores[j] = 1.0f - acc;
}

template __global__ void kernel_dot_product_bs<16>(const float*,const float*,float*,int,int);
template __global__ void kernel_dot_product_bs<32>(const float*,const float*,float*,int,int);
template __global__ void kernel_dot_product_bs<64>(const float*,const float*,float*,int,int);
template __global__ void kernel_dot_product_bs<128>(const float*,const float*,float*,int,int);
template __global__ void kernel_dot_product_bs<256>(const float*,const float*,float*,int,int);
template __global__ void kernel_dot_product_bs<512>(const float*,const float*,float*,int,int);
template __global__ void kernel_dot_product_bs<1024>(const float*,const float*,float*,int,int);

// ===== 实现函数 =====

int gpu_single_query_search(const float* d_base, const float* h_query,
    int n, int vecdim, int num_queries, int top_k, int* h_result,
    GpuBenchResult* result) {

    int bs = get_adaptive_block_size(vecdim);
    int grid = (n + bs - 1) / bs;
    size_t shmem = vecdim * sizeof(float);

    float *d_sc, *d_qb; int *d_idx;
    GPU_MALLOC(&d_sc, (size_t)n*sizeof(float));
    GPU_MALLOC(&d_qb, (size_t)vecdim*sizeof(float));
    GPU_MALLOC(&d_idx, (size_t)top_k*sizeof(int));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    auto t0 = std::chrono::high_resolution_clock::now();
    double th2d=0, tcomp=0, td2h=0;

    for (int i = 0; i < num_queries; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        HIP_CHECK(hipMemcpyAsync(d_qb, &h_query[i*vecdim], (size_t)vecdim*4, hipMemcpyHostToDevice, stream));
        auto t2 = std::chrono::high_resolution_clock::now();
        th2d += std::chrono::duration<double,std::micro>(t2-t1).count();

        kernel_single_dot_product<<<grid,bs,shmem,stream>>>(d_base,d_qb,d_sc,n,vecdim);
        kernel_single_topk<<<1,TOPK_BLOCK,0,stream>>>(d_sc,d_idx,n,top_k);
        HIP_CHECK(hipStreamSynchronize(stream));

        auto t3 = std::chrono::high_resolution_clock::now();
        tcomp += std::chrono::duration<double,std::micro>(t3-t2).count();

        HIP_CHECK(hipMemcpyAsync(&h_result[i*top_k],d_idx,(size_t)top_k*4,hipMemcpyDeviceToHost,stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        td2h += std::chrono::duration<double,std::micro>(
            std::chrono::high_resolution_clock::now()-t3).count();
    }

    double total = std::chrono::duration<double,std::micro>(
        std::chrono::high_resolution_clock::now()-t0).count();

    HIP_CHECK(hipStreamDestroy(stream));
    GPU_FREE(d_sc); GPU_FREE(d_qb); GPU_FREE(d_idx);

    if (result) {
        result->total_time_us=total; result->compute_time_us=tcomp;
        result->h2d_time_us=th2d; result->d2h_time_us=td2h;
        result->per_query_us=total/num_queries; result->num_queries=num_queries;
        result->top_k=top_k; result->qps=(num_queries/total)*1e6;
        result->tflops=(2.0*num_queries*n*vecdim)/(tcomp*1e6);
    }
    return 0;
}

int gpu_batch_gemm_search(const float* d_base, const float* h_query,
    int n, int vecdim, int num_queries, int top_k, int batch_size,
    int* h_result, GpuBenchResult* result) {

    if (batch_size <= 0 || batch_size > num_queries) batch_size = num_queries;

    float *d_qb, *d_sc; int *d_idx;
    GPU_MALLOC(&d_qb, (size_t)batch_size*vecdim*4);
    GPU_MALLOC(&d_sc, (size_t)n*batch_size*4);
    GPU_MALLOC(&d_idx, (size_t)batch_size*top_k*4);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    auto t0 = std::chrono::high_resolution_clock::now();
    double th2d=0, tcomp=0, td2h=0;
    int nb = (num_queries + batch_size - 1) / batch_size;

    for (int b = 0; b < nb; b++) {
        int cur = std::min(batch_size, num_queries - b*batch_size);
        int off = b*batch_size;

        auto t1 = std::chrono::high_resolution_clock::now();
        HIP_CHECK(hipMemcpyAsync(d_qb, &h_query[off*vecdim], (size_t)cur*vecdim*4, hipMemcpyHostToDevice, stream));
        th2d += std::chrono::duration<double,std::micro>(std::chrono::high_resolution_clock::now()-t1).count();

        t1 = std::chrono::high_resolution_clock::now();
        dim3 blk(16,16), grd((cur+15)/16, (n+15)/16);
        kernel_tiled_gemm_simple<<<grd,blk,0,stream>>>(d_base,d_qb,d_sc,n,cur,vecdim);
        kernel_batch_topk<<<cur,TOPK_BLOCK,0,stream>>>(d_sc,d_idx,n,cur,top_k);
        HIP_CHECK(hipStreamSynchronize(stream));
        tcomp += std::chrono::duration<double,std::micro>(std::chrono::high_resolution_clock::now()-t1).count();

        t1 = std::chrono::high_resolution_clock::now();
        HIP_CHECK(hipMemcpyAsync(&h_result[off*top_k],d_idx,(size_t)cur*top_k*4,hipMemcpyDeviceToHost,stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        td2h += std::chrono::duration<double,std::micro>(std::chrono::high_resolution_clock::now()-t1).count();
    }

    double total = std::chrono::duration<double,std::micro>(
        std::chrono::high_resolution_clock::now()-t0).count();

    HIP_CHECK(hipStreamDestroy(stream));
    GPU_FREE(d_qb); GPU_FREE(d_sc); GPU_FREE(d_idx);

    if (result) {
        result->total_time_us=total; result->compute_time_us=tcomp;
        result->h2d_time_us=th2d; result->d2h_time_us=td2h;
        result->per_query_us=total/num_queries; result->num_queries=num_queries;
        result->top_k=top_k; result->qps=(num_queries/total)*1e6;
        result->tflops=(2.0*(double)num_queries*n*vecdim)/(tcomp*1e6);
    }
    return 0;
}

int gpu_rocblas_search(const float* d_base, const float* h_query,
    int n, int vecdim, int num_queries, int top_k,
    int batch_size, int* h_result, GpuBenchResult* result) {
    printf("[WARN] rocBLAS not compiled, falling back to tiled GEMM\n");
    return gpu_batch_gemm_search(d_base,h_query,n,vecdim,num_queries,top_k,batch_size,h_result,result);
}

#define LAUNCH_BS(BS) kernel_dot_product_bs<BS><<<grid,BS,shmem,stream>>>(d_base,d_qb,d_sc,n,dim)

int gpu_block_tuning(const float* d_base, const float* h_query,
    int n, int vecdim, int num_queries, int top_k, int* h_result) {

    printf("\n# Block Size Tuning (dim=%d, queries=%d)\n", vecdim, num_queries);
    printf("| BlockSize | Time(ms) | PerQuery(us) | Top-3 Results |\n");
    printf("| :--- | :--- | :--- | :--- |\n");

    int bss[] = {16,32,64,128,256,512,1024};
    float *d_sc, *d_qb; int *d_idx;
    GPU_MALLOC(&d_sc,(size_t)n*4);
    GPU_MALLOC(&d_qb,(size_t)vecdim*4);
    GPU_MALLOC(&d_idx,(size_t)top_k*4);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    for (int bi = 0; bi < 7; bi++) {
        int BS = bss[bi], grid = (n+BS-1)/BS;
        size_t shmem = vecdim*sizeof(float);

        HIP_CHECK(hipMemcpyAsync(d_qb,h_query,(size_t)vecdim*4,hipMemcpyHostToDevice,stream));

        hipEvent_t ev1,ev2;
        hipEventCreate(&ev1); hipEventCreate(&ev2);
        hipEventRecord(ev1,stream);

        switch(BS) {
            case 16: LAUNCH_BS(16); break; case 32: LAUNCH_BS(32); break;
            case 64: LAUNCH_BS(64); break; case 128: LAUNCH_BS(128); break;
            case 256: LAUNCH_BS(256); break; case 512: LAUNCH_BS(512); break;
            case 1024: LAUNCH_BS(1024); break;
        }
        kernel_single_topk<<<1,TOPK_BLOCK,0,stream>>>(d_sc,d_idx,n,top_k);

        hipEventRecord(ev2,stream);
        hipEventSynchronize(ev2);
        float ms; hipEventElapsedTime(&ms,ev1,ev2);
        hipEventDestroy(ev1); hipEventDestroy(ev2);

        int* hr = (int*)malloc(top_k*4);
        HIP_CHECK(hipMemcpy(hr,d_idx,(size_t)top_k*4,hipMemcpyDeviceToHost));
        printf("| %5d | %.3f | %.1f | [%d,%d,%d] |\n",
               BS, ms, ms*1000.0/num_queries,
               hr[0], hr[min(1,top_k-1)], hr[min(2,top_k-1)]);
        free(hr);
    }

    HIP_CHECK(hipStreamDestroy(stream));
    GPU_FREE(d_sc); GPU_FREE(d_qb); GPU_FREE(d_idx);
    return 0;
}
#undef LAUNCHBS

int gpu_flat_search(const float* base_data, const float* query_data,
    int n, int vecdim, int num_queries, int top_k,
    int* result_indices, RunMode mode, DistMetric metric,
    int batch_size, GpuBenchResult* result) {

    float* d_base;
    GPU_MALLOC(&d_base, (size_t)n*vecdim*4);
    HIP_CHECK(hipMemcpy(d_base, base_data, (size_t)n*vecdim*4, hipMemcpyHostToDevice));

    int ret = 0;
    switch(mode) {
        case MODE_SINGLE_QUERY:
        case MODE_PERSISTENT:
            ret = gpu_single_query_search(d_base,query_data,n,vecdim,num_queries,top_k,result_indices,result);
            break;
        case MODE_BATCH_GEMM:
            ret = gpu_batch_gemm_search(d_base,query_data,n,vecdim,num_queries,top_k,
                batch_size>0?batch_size:64, result_indices, result);
            break;
        case MODE_ROCBLAS:
            ret = gpu_rocblas_search(d_base,query_data,n,vecdim,num_queries,top_k,
                batch_size>0?batch_size:64, result_indices, result);
            break;
        case MODE_BLOCK_TUNING:
            ret = gpu_block_tuning(d_base,query_data,n,vecdim,num_queries,top_k,result_indices);
            break;
        default: fprintf(stderr,"[ERROR] Unknown mode %d\n",mode); ret=-1;
    }

    GPU_FREE(d_base);
    return ret;
}
CPP_EOF
echo "  [OK] flat_scan_gpu_hip.cpp"

# ==================== main_ann_gpu.cc ====================
cat > main_ann_gpu.cc << 'MAIN_EOF'
#include "flat_scan_gpu_hip.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

#define DEFAULT_DATA_DIR "./data"
#define DEFAULT_BASE_FILE  "DEEP100K.base.100k.fbin.larkcache"
#define DEFAULT_QUERY_FILE "DEEP100K.query.fbin"
#define DEFAULT_GT_FILE    "DEEP100K.gt.query.100k.top100.bin"

void cpu_baseline(const float* base, const float* query, int n, int dim,
    int nq, int k, int* result, double* avg_us=nullptr) {
    std::vector<float> dist(n);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int q=0;q<nq;q++){
        for (int i=0;i<n;i++){
            float dot=0; for (int d=0;d<dim;d++) dot+=query[q*dim+d]*base[i*dim+d];
            dist[i]=1.0f-dot;
        }
        std::vector<std::pair<float,int>> idx(n);
        for (int i=0;i<n;i++) idx[i]={dist[i],i};
        std::partial_sort(idx.begin(),idx.begin()+k,idx.end());
        for (int i=0;i<k;i++) result[q*k+i]=idx[i].second;
    }
    if(avg_us) *avg_us=std::chrono::duration<double,std::micro>(
        std::chrono::high_resolution_clock::now()-t0).count()/nq;
}

void print_header(const char* t) {
    printf("\n%s\n", std::string(60,'=').c_str());
    printf("  %s\n", t);
    printf("%s\n\n", std::string(60,'=').c_str());
}

void print_result(const GpuBenchResult& r, const char* label, double cpu_us=0, double other_us=0) {
    printf("--- %s ---\n", label);
    printf("  Total: %.1f us | PerQuery: %.1f us | Compute: %.1f us\n",
           r.total_time_us, r.per_query_us, r.compute_time_us);
    printf("  H2D: %.1f us | D2H: %.1f us | QPS: %.0f | TFLOPS: %.3f\n",
           r.h2d_time_us, r.d2h_time_us, r.qps, r.tflops);
    printf("  Recall@%d: %.4f", r.top_k, r.recall_at_k);
    if(cpu_us>0) printf(" | SpeedupvsCPU: %.1fx", cpu_us/r.per_query_us);
    if(other_us>0) printf(" | SpeedupvsOther: %.1fx", other_us/r.per_query_us);
    printf("\n");
}

void print_usage(const char* p) {
    printf("Usage: %s [mode] [options]\nModes: 0=single, 1=persistent, 2=batch(reco), 3=rocblas, 4=tuning, 5=cpu, all\nOptions: --batch N --queries N --topk N --data PATH\n",p);
}

int main(int argc, char** argv[]) {
    int mode=2, batch=64, queries=100, topk=10;
    std::string data_dir = DEFAULT_DATA_DIR;
    bool run_all=false;

    for (int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--help")||!strcmp(argv[i],"-h")){print_usage(argv[0]);return 0;}
        else if(!strcmp(argv[i],"--batch")&&i+1<argc) batch=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--queries")&&i+1<argc) queries=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--topk")&&i+1<argc) topk=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--data")&&i+1<argc) data_dir=argv[++i];
        else if(!strcmp(argv[i],"all")) run_all=true;
        else mode=atoi(argv[i]);
    }

    std::string bp=data_dir+"/"+DEFAULT_BASE_FILE;
    std::string qp=data_dir+"/"+DEFAULT_QUERY_FILE;
    std::string gp=data_dir+"/"+DEFAULT_GT_FILE;

    printf("[INIT] Data dir: %s\n", data_dir.c_str());

    int n,d,nq,dq,gn,gk;
    float* hb=load_fbin_file(bp.c_str(),&n,&d);
    float* hq=load_fbin_file(qp.c_str(),&nq,&dq);
    int*  hg=load_gt_bin_file(gp.c_str(),&gn,&gk);
    if(d!=dq){fprintf(stderr,"Dim mismatch\n");return 1;}

    queries=std::min(queries,nq);
    printf("[INIT] n=%d dim=%d nq=%d(use%d) top-%d\n",n,d,nq,queries,topk);

    std::vector<int> res(queries*topk);
    double cpu_us=0;
    const double OTHER_US=12038.9;

    auto run_exp=[&](int m,const char* lab)->GpuBenchResult{
        GpuBenchResult br={};
        memset(res.data(),0,queries*topk*4);
        gpu_flat_search(hb,hq,n,d,queries,topk,res.data(),(RunMode)m,DIST_INNER_PRODUCT,batch,&br);
        br.recall_at_k=compute_recall(res.data(),hg,queries,topk,gk);
        print_result(br,lab,cpu_us,OTHER_US);
        return br;
    };

    if(run_all){
        // CPU baseline
        print_header("Level 0: CPU Baseline");
        cpu_baseline(hb,hq,n,d,queries,topk,res.data(),&cpu_us);
        printf("CPU: %.1f us/query Recall@%d=%.4f\n\n",cpu_us,topk,
               compute_recall(res.data(),hg,queries,topk,gk));

        // Level 1
        print_header("Level 1: Single Query GPU");
        GpuBenchResult r1=run_exp(0,"Single Query");

        // Level 2 batch scan
        print_header("Level 2: Batch GEMM — Batch Size Scan");
        printf("|Batch|PerQ(us)|Recall|QPS|TFLOPS|\n|:---|:---|:---|:---|:---|\n");
        int bss[]={1,8,16,32,64,128,200};
        GpuBenchResult best={}; double bestq=0; int bestbs=0;
        for(int bs:bss){ if(bs>queries) continue;
            GpuBenchResult br={};
            memset(res.data(),0,queries*topk*4);
            gpu_flat_search(hb,hq,n,d,queries,topk,res.data(),MODE_BATCH_GEMM,DIST_INNER_PRODUCT,bs,&br);
            br.recall_at_k=compute_recall(res.data(),hg,queries,topk,gk);
            printf("|%d|%.1f|%.4f|%.0f|%.3f|\n",bs,br.per_query_us,br.recall_at_k,br.qps,br.tflops);
            if(br.qps>bestq){bestq=br.qps;best=br;bestbs=bs;}
        }
        print_header("Level 2: Best Batch");
        print_result(best,"Best Batch",cpu_us,OTHER_US);

        // Level 3
        print_header("Level 3: rocBLAS comparison");
        GpuBenchResult r3=run_exp(3,"rocBLAS");

        // Summary
        print_header("Summary Table");
        printf("%-18s %10s %8s %10s %8s\n","Method","PerQ(us)","Recall","QPS","TFLOPS");
        printf("%-18s %10.1f %8.4f %10.0f %8.3f\n","CPU Baseline",cpu_us,
               compute_recall(res.data(),hg,queries,topk,gk),(queries/cpu_us)*1e6,0.0);
        printf("%-18s %10.1f %8.4f %10.0f %8.3f\n","Other GPU",OTHER_US,0.9995,(queries/OTHER_US)*1e6,0.0);
        printf("%-18s %10.1f %8.4f %10.0f %8.3f\n","Our L1-Single",r1.per_query_us,r1.recall_at_k,r1.qps,r1.tflops);
        printf("%-18s %10.1f %8.4f %10.0f %8.3f\n","Our L2-Batch(best)",best.per_query_us,best.recall_at_k,best.qps,best.tflops);
        printf("%-18s %10.1f %8.4f %10.0f %8.3f\n","Our L3-rocblas",r3.per_query_us,r3.recall_at_k,r3.qps,r3.tflops);
    } else if(mode==5){
        print_header("CPU Baseline only");
        cpu_baseline(hb,hq,n,d,queries,topk,res.data(),&cpu_us);
        printf("Time: %.1f us/query Recall@%d: %.4f\n",cpu_us,topk,compute_recall(res.data(),hg,queries,topk,gk));
    } else {
        const char* labs[]={"L1-Single","L1-Persistent","L2-BatchGEMM","L3-rocBLAS","L4-BlockTuning"};
        print_header(labs[mode]);
        if(mode==4){
            gpu_block_tuning(nullptr,hq,n,d,queries,topk,res.data());
        }else{
            run_exp(mode,labs[mode]);
        }
    }

    free(hb);free(hq);free(hg);
    printf("\n[DONE]\n"); return 0;
}
MAIN_EOF
echo "  [OK] main_ann_gpu.cc"

# ==================== Makefile ====================
cat > Makefile << 'MAKEEOF'
HIPCC ?= hipcc
TARGET = ann_gpu
HIPFLAGS = -std=c++17 -O3 -Wno-unused-result -Wall
ROCM_PATH ?= /opt/rocm
HIP_LIB = -L$(ROCM_PATH)/lib -lamdhip64
DATA_DIR ?= ./data

.PHONY: all clean run run0 run1 run2 run3 run4 runall tune

all: $(TARGET)

flat_scan_gpu_hip.o: flat_scan_gpu_hip.cpp flat_scan_gpu_hip.h
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

main_ann_gpu.o: main_ann_gpu.cc flat_scan_gpu_hip.h
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

$(TARGET): main_ann_gpu.o flat_scan_gpu_hip.o
	$(HIPCC) $(HIPFLAGS) $^ $(HIP_LIB) -o $@

clean:
	rm -f $(TARGET) *.o

RUN_BASE = ./$(TARGET)
RUN_ARGS = --data "$(DATA_DIR)" --queries 100

run: $(TARGET)
	$(RUN_BASE) 2 --batch 64 $(RUN_ARGS)

run0: $(TARGET); $(RUN_BASE) 0 $(RUN_ARGS)
run1: $(TARGET); $(RUN_BASE) 1 $(RUN_ARGS)
run2: $(TARGET); $(RUN_BASE) 2 --batch 64 $(RUN_ARGS)
run3: $(TARGET); $(RUN_BASE) 3 --batch 64 $(RUN_ARGS)
run4: $(TARGET); $(RUN_BASE) 4 $(RUN_ARGS)
runall: $(TARGET); $(RUN_BASE) all --batch 64 $(RUN_ARGS)
tune: $(TARGET);  $(RUN_BASE) 4 $(RUN_ARGS)
MAKEEOF
echo "  [OK] Makefile"

echo ""
echo "==== 编译 ===="

# 确保ROCm环境
if [ -f /etc/profile.d/rocm.sh ]; then
    source /etc/profile.d/rocm.sh 2>/dev/null || true
fi
if command -v module &>/dev/null; then
    module load rocm 2>/dev/null || true
fi

# 检查hipcc
which hipcc 2>/dev/null || { echo "[ERROR] hipcc not found. Please ensure ROCm is loaded."; exit 1; }
hipcc --version 2>&1 | head -1

# 设置数据集路径
export DATA_DIR="$DATA_DIR"

# 编译
make clean 2>/dev/null || true
make all 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================="
    echo "  ✅ 编译成功!"
    echo "============================================="
    echo ""
    echo "接下来可以运行实验:"
    echo ""
    echo "  # 推荐模式: Batch矩阵乘法"
    echo "  make run          # Mode 2 (推荐)"
    echo ""
    echo "  # 其他模式:"
    echo "  make run0         # 单查询GPU"
    echo "  make run1         # 持久化内存"
    echo "  make run2         # Batch GEMM"
    echo "  make run3         # rocBLAS对比"
    echo "  make run4         # Block Size调优"
    echo "  make runall       # 全模式对比"
    echo ""
    echo "或者直接运行:"
    echo "  ./ann_gpu 2 --batch 64 --data \"$DATA_DIR\" --queries 100"
    echo ""
    
    # 自动运行推荐模式
    if [ "${AUTO_RUN:-}" = "1" ]; then
        echo ">>> Auto-running Mode 2 (Batch GEMM)..."
        make run
    fi
else
    echo "[ERROR] 编译失败!"
    exit 1
fi
