cmake_minimum_required(VERSION 3.16)
project(ann_openmp LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -ffast-math -fopenmp")

find_package(OpenMP REQUIRED)

add_executable(ann_openmp main_openmp.c)
target_include_directories(ann_openmp PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ann_openmp PRIVATE OpenMP::OpenMP_C)
