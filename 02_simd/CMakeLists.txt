cmake_minimum_required(VERSION 3.16)
project(ann_simd LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -mfma")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O3 -march=native -ffast-math -mfma")

add_executable(ann_simd main_simd.cpp)
target_include_directories(ann_simd PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
