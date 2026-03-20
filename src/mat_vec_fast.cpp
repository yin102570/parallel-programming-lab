#include "../include/base.h"
void mat_vec_fast(const vector<TYPE>& A, const vector<TYPE>& x, vector<TYPE>& y, int n) {
    const int BLOCK = 64;
    fill(y.begin(), y.end(), 0.0);

    for (int ii = 0; ii < n; ii += BLOCK) {
        for (int jj = 0; jj < n; jj += BLOCK) {
            int imax = min(ii + BLOCK, n);
            int jmax = min(jj + BLOCK, n);
            for (int i = ii; i < imax; ++i) {
                TYPE xi = x[i];
                int j = jj;
                for (; j + 3 < jmax; j += 4) {
                    y[j] += A[i * n + j] * xi;
                    y[j + 1] += A[i * n + j + 1] * xi;
                    y[j + 2] += A[i * n + j + 2] * xi;
                    y[j + 3] += A[i * n + j + 3] * xi;
                }
                for (; j < jmax; ++j) {
                    y[j] += A[i * n + j] * xi;
                }
            }
        }
    }
}