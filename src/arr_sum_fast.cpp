#include "../include/base.h"

TYPE arr_sum_fast(vector<TYPE> arr, int n) {
    while (n > 1) {
        int half = n / 4;
        for (int i = 0; i < half; ++i) {
            arr[i] += arr[i + half];
            arr[i] += arr[i + 2*half];
            arr[i] += arr[i + 3*half];
        }
        if (n % 4 != 0) {
            for (int i = 4*half; i < n; ++i) {
                arr[0] += arr[i];
            }
        }
        n = (n + 3) / 4;
    }
    return arr[0];
}