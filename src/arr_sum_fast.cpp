#include "../include/base.h"


TYPE arr_sum_fast(const vector<TYPE>&arr, int n) {
    // 4路并行累加
    TYPE sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    int i = 0;

    // 每次处理4个元素
    for (; i + 3 < n; i += 4) {
        sum0 += arr[i];
        sum1 += arr[i + 1];
        sum2 += arr[i + 2];
        sum3 += arr[i + 3];
    }

    // 处理剩余元素
    for (; i < n; ++i) {
        sum0 += arr[i];
    }

    return sum0 + sum1 + sum2 + sum3;
}