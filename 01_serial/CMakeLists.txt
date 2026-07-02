cmake_minimum_required(VERSION 3.16)
project(ann_serial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")

add_executable(ann_serial ann_serial.cpp)
target_include_directories(ann_serial PRIVATE ${CMAKE_SOURCE_DIR}/include)
