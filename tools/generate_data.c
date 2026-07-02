/**
 * generate_data.c — 生成测试向量数据集
 *
 * 生成格式：n(int32) + dim(int32) + float32[n*dim]
 * 与load_dataset()兼容
 *
 * 编译：gcc -O2 -o generate_data generate_data.c
 * 运行：./generate_data 100000 96 > vectors.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(int argc, char** argv) {
    int n = 100000;
    int dim = 96;

    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) dim = atoi(argv[2]);

    // 确保dim是PQ分段数的整数倍
    if (dim % 8 != 0) {
        fprintf(stderr, "警告: dim=%d 不是8的倍数, PQ可能不整齐\n", dim);
    }

    printf("生成 %d x %d 随机向量...\n", n, dim);

    // 文件头：n, dim
    fwrite(&n, sizeof(int), 1, stdout);
    fwrite(&dim, sizeof(int), 1, stdout);

    // 生成服从均匀分布的随机向量
    // 注意：生产环境应使用真实数据集（DEEP100K, SIFT1B等）
    srand(42);
    for (int i = 0; i < n * dim; i++) {
        float v = (float)rand() / RAND_MAX;
        fwrite(&v, sizeof(float), 1, stdout);
    }

    fprintf(stderr, "完成: %.1f MB\n", n * dim * sizeof(float) / 1e6);
    return 0;
}
