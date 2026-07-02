cmake_minimum_required(VERSION 3.16)
project(ANN_Parallel_System VERSION 1.0 LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 全局优化标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -Wall")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O3 -march=native -ffast-math -Wall")

# OpenMP
find_package(OpenMP REQUIRED)

# MPI
find_package(MPI)

# CUDA (可选)
find_package(CUDAToolkit QUIET)

# 公共头文件
include_directories(${CMAKE_SOURCE_DIR}/include)

# 子目录
add_subdirectory(01_serial)
add_subdirectory(02_simd)
add_subdirectory(03_openmp)

if(MPI_FOUND)
    add_subdirectory(04_mpi)
    add_subdirectory(06_hybrid)
endif()

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    add_subdirectory(05_cuda)
endif()
