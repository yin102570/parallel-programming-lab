cmake_minimum_required(VERSION 3.14)
project(CPU_OPT)

set(CMAKE_CXX_STANDARD 17)
# 最高优化级别，适配本地CPU架构
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -funroll-loops -ffast-math")

add_executable(lab 
    src/main.cpp 
    src/mat_vec_bad.cpp 
    src/mat_vec_fast.cpp 
    src/arr_sum_slow.cpp 
    src/arr_sum_fast.cpp
)