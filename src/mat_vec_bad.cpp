#include "../include/base.h"

void mat_vec_bad(const vector<TYPE>& A, const vector<TYPE>& x, vector<TYPE>& y, int n) {
    for (int i = 0; i < n; ++i) {
        TYPE sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += A[j * n + i] * x[j]; // 这里设定为列优先访问
        }
        y[i] = sum;
    }
}