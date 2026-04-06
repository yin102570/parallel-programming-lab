/**
 * pq_index.h — PQ乘积量化实现
 *
 * 前沿独到洞察（这是整个ANNS优化中最深刻的技巧）：
 *
 * 1. PQ的本质：把"计算"变成"查表"
 *    传统做法：距离 = Σ(query[d] - vec[d])² = Σ query[d]² + Σ vec[d]² - 2Σ query[d]*vec[d]
 *             每次计算需要3次数组访问 + dim次乘加
 *    PQ做法：把dim维向量分成m段，每段独立聚类256个中心。
 *            距离 = Σ (|query段 - center段|²) ，每段只需一次查表
 *            96维/8段 = 每段12维，256个中心，每个中心预计算距离并存入LUT
 *
 * 2. 内存访问的质变：
 *    原始向量：384字节/向量（96 * 4字节）
 *    PQ编码：8字节/向量（8段 * 1字节/段），压缩比 48:1
 *    这意味着在同样的内存带宽下，PQ能在单位时间内"扫过"48倍的向量数量。
 *    这不是算法优化，这是数据结构对硬件特性的精准适配。
 *
 * 3. ADC (Asymmetric Distance Computation) 的核心洞察：
 *    非对称性体现在：查询向量是完整精度（float），但数据库向量是压缩的（uint8）。
 *    这允许我们：查询可以完整计算每个子向量到256个中心的距离（查LUT），
 *    而数据库向量只需存储"属于哪个中心"的ID。
 *    对称量化（SQ）的问题：量化误差同时影响查询和数据库，双重损失。
 *    ADC只让数据库端承受量化损失，是更优的工程折中。
 *
 * 4. m（分段数）的选取：
 *    m越大：压缩比越高（每段维度越小，中心数256不变）
 *    m越小：每段的子空间维度越大，量化误差越小（召回率更高）
 *    96维的最佳分段方案：m=8时每段12维，m=12时每段8维。
 *    经验规律：每段16-32维时PQ效果最好（过低->量化误差大，过高->LUT太小收益少）
 *    所以m=8（每段12维）比m=16（每段6维）召回率更好。
 *
 * 5. 召回率损失的根本原因：
 *    PQ把连续空间强行映射到离散网格（256^m个网格点）。
 *    如果向量的真实最近邻恰好落在相邻网格，则必定被错误排序。
 *    解决方向：OPQ（优化乘积量化）通过在学习阶段就让向量均匀分布在网格中，
 *    减少"跨网格邻居"的问题。这是Faiss中OPQ比PQ召回率高的原因。
 */

#ifndef PQ_INDEX_H
#define PQ_INDEX_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

//==============================================================================
// 数据结构
//==============================================================================

// PQ编码器参数
#define PQ_M 8           // 分段数（本项目选8，每段12维）
#define PQ_KS 256        // 每段中心数（必须 ≤ 256，因为用uint8存储）
#define PQ_NBITS 8       // 中心编码位数（log2(PQ_KS)）

// PQ子空间描述
typedef struct {
    float* centers;     // [M * KS * Ds] 各子空间的聚类中心，Ds=dim/M
    int M;               // 分段数
    int KS;              // 每段中心数
    int dim;             // 原始维度
    int Ds;              // 每段子维度
} PQModel;

// PQ编码表：存储数据库中每个向量的PQ编码
typedef struct {
    uint8_t* codes;     // [n * M] 每个向量的PQ编码（压缩存储）
    int n;               // 向量数量
    int M;               // 分段数
} PQCodes;

// PQ完整索引
typedef struct {
    PQModel* model;      // PQ模型（中心点）
    PQCodes* codes;      // 向量编码
    int n;               // 向量数量
    int dim;             // 原始维度
} PQIndex;

// LUT预计算表
typedef struct {
    float* tables;       // [M * KS] 查询到各中心的距离表
    int M;
    int KS;
} PQLUT;

//==============================================================================
// PQ训练（从向量集合学习PQ模型）
//==============================================================================

/**
 * 训练PQ模型 — 乘积量化的核心
 *
 * 与标准K-Means的区别：
 * - 标准K-Means：在完整D维空间聚类
 * - PQ K-Means：在D/M维子空间分别聚类（各段独立）
 *
 * 这个"独立"至关重要：意味着我们可以并行训练每个子空间的聚类，
 * 且存储的中心表大小是 (M * KS * Ds)，而非 (KS * D)。
 * 当M=8, D=96, Ds=12, KS=256时：M*KS*Ds = 24,576floats ≈ 98KB
 * 而标准K-Means需要 KS*D = 24,576floats，看起来相同。
 * 但关键差异：PQ的每个子空间可以独立使用更小的KS达到同等精度！
 *
 * @param vectors   训练向量 [n * dim]
 * @param n         向量数量
 * @param dim       维度
 * @param M         分段数
 * @param KS        每段中心数（默认256）
 * @param n_iter    训练迭代次数
 */
PQModel* pq_train(const float* vectors, int n, int dim, int M, int KS, int n_iter) {
    int Ds = dim / M;  // 每段维度，必须整除

    PQModel* model = (PQModel*)malloc(sizeof(PQModel));
    model->M = M;
    model->KS = KS;
    model->dim = dim;
    model->Ds = Ds;
    model->centers = (float*)malloc(sizeof(float) * M * KS * Ds);

    // 对每个子空间独立训练K-Means
    for (int m = 0; m < M; m++) {
        float* sub_vectors = (float*)malloc(sizeof(float) * n * Ds);

        // 提取第m段子向量
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < Ds; d++) {
                sub_vectors[i * Ds + d] = vectors[i * dim + m * Ds + d];
            }
        }

        // 在子空间上训练K-Means（简化版：随机采样初始化）
        float* centers = model->centers + m * KS * Ds;

        // 随机初始化中心
        srand(42 + m);
        int* rand_indices = (int*)malloc(sizeof(int) * KS);
        for (int k = 0; k < KS; k++) {
            rand_indices[k] = rand() % n;
            memcpy(centers + k * Ds, sub_vectors + rand_indices[k] * Ds, sizeof(float) * Ds);
        }

        // 迭代优化
        for (int iter = 0; iter < n_iter; iter++) {
            int* assignments = (int*)malloc(sizeof(int) * n);
            int* counts = (int*)calloc(KS, sizeof(int));
            float* sum = (float*)calloc(KS * Ds, sizeof(float));

            // E步：分配
            for (int i = 0; i < n; i++) {
                float min_dist = FLT_MAX;
                int best = 0;
                const float* v = sub_vectors + i * Ds;
                for (int k = 0; k < KS; k++) {
                    float dist = 0.0f;
                    const float* c = centers + k * Ds;
                    for (int d = 0; d < Ds; d++) {
                        float diff = v[d] - c[d];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        best = k;
                    }
                }
                assignments[i] = best;
                counts[best]++;
                for (int d = 0; d < Ds; d++) {
                    sum[best * Ds + d] += v[d];
                }
            }

            // M步：更新中心
            for (int k = 0; k < KS; k++) {
                if (counts[k] > 0) {
                    float inv = 1.0f / counts[k];
                    for (int d = 0; d < Ds; d++) {
                        centers[k * Ds + d] = sum[k * Ds + d] * inv;
                    }
                }
            }

            free(assignments);
            free(counts);
            free(sum);
        }

        free(sub_vectors);
    }

    return model;
}

/**
 * 用训练好的模型编码向量
 *
 * @param model     PQ模型
 * @param vectors   待编码向量 [n * dim]
 * @param n         向量数量
 * @return          PQ编码表
 */
PQCodes* pq_encode(const PQModel* model, const float* vectors, int n) {
    PQCodes* codes = (PQCodes*)malloc(sizeof(PQCodes));
    codes->codes = (uint8_t*)malloc(sizeof(uint8_t) * n * model->M);
    codes->n = n;
    codes->M = model->M;

    for (int i = 0; i < n; i++) {
        const float* v = vectors + i * model->dim;
        for (int m = 0; m < model->M; m++) {
            // 找最近的中心
            float min_dist = FLT_MAX;
            int best = 0;
            const float* sub_v = v + m * model->Ds;

            for (int k = 0; k < model->KS; k++) {
                float dist = 0.0f;
                const float* center = model->centers + m * model->KS * model->Ds + k * model->Ds;
                for (int d = 0; d < model->Ds; d++) {
                    float diff = sub_v[d] - center[d];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best = k;
                }
            }
            codes->codes[i * model->M + m] = (uint8_t)best;
        }
    }

    return codes;
}

//==============================================================================
// PQ搜索（ADC距离计算）
//==============================================================================

/**
 * 构建查询的LUT（查找表）
 *
 * 这是ADC的核心：查询子向量到所有中心的距离预计算一次，后续每条编码只查表累加。
 *
 * 计算量：M * KS次距离计算（8 * 256 = 2048次/查询）
 * vs 原始线性扫描：n * dim次（10万 * 96 = 960万次/查询）
 * 加速比：960万 / 2048 ≈ 4688倍！（理论）
 * 实际受限于：LUT表的内存访问、编码遍历的开销
 *
 * @param model     PQ模型
 * @param query     查询向量 [dim]
 * @return          预计算的LUT [M * KS]
 */
PQLUT* pq_build_lut(const PQModel* model, const float* query) {
    PQLUT* lut = (PQLUT*)malloc(sizeof(PQLUT));
    lut->M = model->M;
    lut->KS = model->KS;
    lut->tables = (float*)malloc(sizeof(float) * model->M * model->KS);

    for (int m = 0; m < model->M; m++) {
        const float* query_sub = query + m * model->Ds;
        float* lut_m = lut->tables + m * model->KS;

        // 计算查询子向量到第m段子空间所有KS个中心的距离
        for (int k = 0; k < model->KS; k++) {
            float dist = 0.0f;
            const float* center = model->centers + m * model->KS * model->Ds + k * model->Ds;
            for (int d = 0; d < model->Ds; d++) {
                float diff = query_sub[d] - center[d];
                dist += diff * diff;
            }
            lut_m[k] = dist;
        }
    }

    return lut;
}

/**
 * 用PQ编码和LUT计算近似距离
 *
 * 查表代替计算的本质：把子空间的距离计算"离线化"
 * |q - v|² = |q_m - c_m(v)|² ≈ LUT[m][code_m(v)]
 * 编码只存储"属于哪个中心"，距离查表获得
 *
 * @param lut       查表表
 * @param codes     PQ编码 [M] (uint8)
 * @return          近似L2距离
 */
static inline float pq_distance_from_lut(const PQLUT* lut, const uint8_t* codes) {
    float dist = 0.0f;
    for (int m = 0; m < lut->M; m++) {
        dist += lut->tables[m * lut->KS + codes[m]];
    }
    return dist;
}

/**
 * PQ搜索：扫描所有编码，找top-k
 *
 * 性能瓶颈分析：
 * 1. LUT已预计算，访问模式是顺序读（M * KS个float，cache友好）
 * 2. 编码访问是跳跃的（codes[i*M+m]，对每个i，m维度不连续）
 *    优化：将编码重新排列为 [M][n] 而非 [n][M]，让m维度连续访问
 * 3. 代码外层循环遍历n个候选，内层循环查M次表
 *    优化：内层循环展开（因为M=8是常量），减少分支开销
 */
typedef struct {
    int* indices;
    float* distances;
    int k;
} PQSearchResult;

PQSearchResult* pq_search(const PQIndex* index, const PQLUT* lut, int k) {
    PQSearchResult* result = (PQSearchResult*)malloc(sizeof(PQSearchResult));
    result->indices = (int*)malloc(sizeof(int) * k);
    result->distances = (float*)malloc(sizeof(float) * k);
    result->k = k;

    for (int i = 0; i < k; i++) {
        result->indices[i] = -1;
        result->distances[i] = FLT_MAX;
    }

    // 扫描所有编码
    for (int i = 0; i < index->n; i++) {
        const uint8_t* code = index->codes->codes + i * index->codes->M;

        // PQ距离：查表累加（已内联展开优化）
        float dist = 0.0f;
        // 手动展开M=8的查表过程，避免循环开销
        dist += lut->tables[0 * lut->KS + code[0]];
        dist += lut->tables[1 * lut->KS + code[1]];
        dist += lut->tables[2 * lut->KS + code[2]];
        dist += lut->tables[3 * lut->KS + code[3]];
        dist += lut->tables[4 * lut->KS + code[4]];
        dist += lut->tables[5 * lut->KS + code[5]];
        dist += lut->tables[6 * lut->KS + code[6]];
        dist += lut->tables[7 * lut->KS + code[7]];

        // Top-k插入
        for (int j = 0; j < k; j++) {
            if (dist < result->distances[j]) {
                for (int m = k - 1; m > j; m--) {
                    result->distances[m] = result->distances[m - 1];
                    result->indices[m] = result->indices[m - 1];
                }
                result->distances[j] = dist;
                result->indices[j] = i;
                break;
            }
        }
    }

    return result;
}

//==============================================================================
// PQ索引构建
//==============================================================================

PQIndex* pq_build(const float* vectors, int n, int dim, int M, int KS, int n_iter) {
    PQIndex* index = (PQIndex*)malloc(sizeof(PQIndex));
    index->n = n;
    index->dim = dim;

    // 训练PQ模型
    index->model = pq_train(vectors, n, dim, M, KS, n_iter);

    // 编码所有向量
    index->codes = pq_encode(index->model, vectors, n);

    return index;
}

void pq_free_lut(PQLUT* lut) {
    if (lut) {
        free(lut->tables);
        free(lut);
    }
}

void pq_free_codes(PQCodes* codes) {
    if (codes) {
        free(codes->codes);
        free(codes);
    }
}

void pq_free_index(PQIndex* index) {
    if (index) {
        free(index->model->centers);
        free(index->model);
        pq_free_codes(index->codes);
        free(index);
    }
}

void pq_free_result(PQSearchResult* result) {
    if (result) {
        free(result->indices);
        free(result->distances);
        free(result);
    }
}

#endif // PQ_INDEX_H
