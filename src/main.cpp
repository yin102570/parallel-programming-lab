#include "../include/base.h"


void mat_vec_bad(const vector<TYPE>& A, const vector<TYPE>& x, vector<TYPE>& y, int n);
void mat_vec_fast(const vector<TYPE>& A, const vector<TYPE>& x, vector<TYPE>& y, int n);
TYPE arr_sum_slow(const vector<TYPE>& arr, int n);
TYPE arr_sum_fast(vector<TYPE> arr, int n);
int main() {
    cout << "n,mat_bad_us,mat_fast_us,mat_speedup,sum_slow_us,sum_fast_us,sum_speedup\n";

    for (int n = N_START; n <= N_MAX; n += STEP) {
        vector<TYPE> A(n * n), x(n), y1(n), y2(n), arr(n);
        // 初始化固定值，保证结果可复现
        for (int i = 0; i < n * n; ++i) A[i] = (i % 17 - 8.0);
        for (int i = 0; i < n; ++i) {
            x[i] = (i % 13 - 6.0);
            arr[i] = (i % 19 - 9.0);
        }

        // 矩阵平凡版计时
        auto t1 = chrono::high_resolution_clock::now();
        for (int r = 0; r < 20; ++r) mat_vec_bad(A, x, y1, n);
        double t_mat_bad = chrono::duration<double, micro>(chrono::high_resolution_clock::now() - t1).count() / 20;

        // 矩阵优化版计时
        auto t2 = chrono::high_resolution_clock::now();
        for (int r = 0; r < 20; ++r) mat_vec_fast(A, x, y2, n);
        double t_mat_fast = chrono::duration<double, micro>(chrono::high_resolution_clock::now() - t2).count() / 20;

        // 求和平凡版计时
        auto t3 = chrono::high_resolution_clock::now();
        TYPE s1 = 0;
        for (int r = 0; r < 50; ++r) s1 = arr_sum_slow(arr, n);
        double t_sum_slow = chrono::duration<double, micro>(chrono::high_resolution_clock::now() - t3).count() / 50;

        // 求和优化版计时
        auto t4 = chrono::high_resolution_clock::now();
        TYPE s2 = 0;
        for (int r = 0; r < 50; ++r) s2 = arr_sum_fast(arr, n);
        double t_sum_fast = chrono::duration<double, micro>(chrono::high_resolution_clock::now() - t4).count() / 50;

        // 输出CSV格式数据
        cout << n << "," << fixed << setprecision(2)
             << t_mat_bad << "," << t_mat_fast << "," << t_mat_bad/t_mat_fast << ","
             << t_sum_slow << "," << t_sum_fast << "," << t_sum_slow/t_sum_fast << "\n";
    }
    return 0;
}