cmake_minimum_required(VERSION 3.16)
project(ann_cuda LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86")

find_package(CUDAToolkit REQUIRED)

add_executable(ann_cuda ann_cuda.cu flat_scan_gpu.cpp)
target_include_directories(ann_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ann_cuda PRIVATE CUDA::cudart CUDA::cuda_driver)

if(OpenMP_FOUND)
    target_link_libraries(ann_cuda PRIVATE OpenMP::OpenMP_CXX)
endif()
