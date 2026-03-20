#include "../include/base.h"

TYPE arr_sum_slow(const vector<TYPE>& arr, int n) {
    TYPE s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += arr[i];
    }
    return s;
}